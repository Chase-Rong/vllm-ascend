# Streaming KV Tiering：图模式（ACLGraph）实现设计

> 本文是 `hybrid_linear_decode_kv_tiering_design.md` 的下钻文档，聚焦一个目标：
> **把图模式下的实现写到函数抽象与时序级，供可行性评审。**
> 对应的 OpenSpec change：`openspec/changes/hybrid-linear-decode-kv-tiering/`。
>
> 阅读前提：已读过主设计文档 §1–§3（动机、三足组结构、D1–D7 决策）。

---

## 0. 本文回答的问题

1. decode 一步在 FULL ACLGraph 模式下到底怎么执行，我们的 hook 插在哪；
2. 每个新组件的**具体函数抽象**（签名级）；
3. eager / 捕获 / replay 三种状态下的**逐事件时序**；
4. 正确性论证：为什么预取**不允许跨步**（这是与权重 prefetch 的本质差异）；
5. staging buffer 的内存账目（它不是免费的，policy 必须把它计入）。

---

## 1. 现状解剖：FULL graph 模式下的 decode 一步

```
CPU (python)                                  NPU
────────────────────────────────────────────────────────────────
scheduler 决策 (本步跑哪些请求, 分配新 block)
  │
model_runner._update_states()
  │  block_table 原地更新 (persistent device tensor, 固定地址) ──→ BlockTable .tensor
  │  slot_mapping / seq_lens 等写入固定 buffer                    (地址在捕获时固定)
  │
update_attn_params (update_stream, ExternalEvent 同步)
  │
ACLGraphWrapper.__call__()                                   graph.replay()
  │  (replay 前按需 sync, acl_graph.py:253-266)            ──→ 整张图一次发射:
  │                                                            93 层 forward + sampling
  │◄─────────────────────────────────────────────────────── 完成 (事件/同步点)
sampler 后处理
```

对我们的设计成立的两个事实：

- **block table 是固定地址的持久 tensor，replay 前原地更新**——这正是 sparse offload 间接寻址依赖的机制，我们复用；
- **CPU 侧 scheduler 决策（新请求、新 block、抢占恢复）发生在 replay 之前**。一旦 replay 开始，block table 在本步内不再变化。**这决定了预取调度的不变量（§5.2）。**

---

## 2. 机制地基：三个先例的精确语义

### 2.1 权重 PrefetchOffloader（fork/join 图内捕获）—— overlap 的模板

上游 `vllm/model_executor/offloader/prefetch.py` + Ascend 适配（NZ buffer）+ 捕获生命周期处理（`vllm_ascend/compilation/acl_graph.py:180-194`）。核心语义：

```python
# 每个被 offload 的层, forward 被 _hook_module_forward 包装:
def forward(*args, **kwargs):
    torch.ops.vllm.wait_prefetch(input_tensor, index)      # join: 等本层拷贝完成
    output = original_forward(*args, **kwargs)             # 原 forward
    torch.ops.vllm.start_prefetch(output, next_index)      # fork: 启动 next 层拷贝
    return output

# start_onload_to_static() —— fork 的具体动作:
fork_event = torch.cuda.Event()
torch.cuda.current_stream().record_event(fork_event)       # 在 compute stream 上打点
copy_stream.wait_event(fork_event)                         # copy stream 等到该点
with torch.cuda.stream(copy_stream):
    gpu_buffer.copy_(cpu_storage, non_blocking=True)       # pinned→HBM 异步拷贝
self._copy_done_event.record(copy_stream)                  # 完成事件
self._prefetch_in_capture = torch.cuda.is_current_stream_capturing()

# _wait_for_layer() —— join:
if capturing:
    if not offloader._prefetch_in_capture: return          # 跳过捕获前的预取
    torch.cuda.current_stream().wait_event(offloader._copy_done_event)
    offloader._prefetch_in_capture = False
```

关键工程事实（全部是已踩过的坑，直接继承）：

- `torch.ops.vllm.wait_prefetch/start_prefetch` 是带 `mutates_args` 的 custom op——给 torch.compile 制造数据依赖，使事件操作进入图且不被优化掉；
- **捕获中 record 的 event 在捕获结束后失效**（`_event_valid_for_eager` 标记，eager 回退 `wait_stream`）；
- 图尾部 dangling fork（最后一层为下一 forward 发起的预取）由 `join_after_forward()` 在捕获结束前补齐 join，否则报 unjoined stream error；
- 捕获前 `sync_prev_onload()` 清在途拷贝；
- **静态 buffer 轮转**：`slot = layer_idx % prefetch_step`，地址固定，图安全。

### 2.2 sparse KV offload——图内间接寻址与 host 可寻址性

`sparse_kv_offload_manager.py:onload_topk_kv(capturing=True)`：图内 async D2H 元数据 → `_launch_host_func` host 节点算拷贝计划 → 图内 async H2D 计划 → device 侧 `offload.sparse_copy(gvas, addr, size, ...)` 按地址表搬数。证明：

- host pinned 内存 device 可寻址（gvas 全局虚拟地址）；
- 逐层调用点就在 attention impl 的 forward 里（`sfa_kv_offload.py`，attention impl 子类化是标准拦截位置）；
- 图内可以含 host 函数节点（我们需要时可用，dense 场景尽量不用）。

### 2.3 我们要拼的东西

| 组件 | 取自 | 改动 |
|---|---|---|
| fork/join 事件 + custom op 包装 | 2.1 | 源地址从固定权重存储 → block table 间接 |
| 静态 staging buffer 轮转 | 2.1 StaticBufferPool | slot 语义改为"在途层"，容量按 batch×W 算 |
| 图内间接 gather（源地址 replay 时解析） | 2.2 | 无 miss 逻辑，计划 = block table 的纯函数 |
| host 池（TP 共享/复制） | 2.2 | 一期 per-rank 复制即可 |
| **预取调度不跨步** | 无（权重无此问题） | **新增约束，§5.2 论证** |

---

## 3. 数据结构设计

### 3.1 Host 池（persistent，CPU pinned）

```
host_pool: 每 streaming 层一块 pinned 内存
  layout: [num_host_blocks, block_size, 576] bf16     # 与 spec page 对齐
  每 DP rank 独立; TP 一期按 per-rank 复制 (二期可 rank0 建池 + gvas 共享)

容量: num_host_blocks × page_bytes ≤ cpu_budget (config)
block table (streaming 组): scheduler 照常分配/维护, block id 即 host_pool 行号
```

### 3.2 Staging buffer（HBM，固定地址，轮转）

```
staging[slot]: [max_graph_batch, W_max, 576] bf16
  slot ∈ {0, 1}                       # slot_capacity=2 (SLOTS)
  W_max = 本层允许的最大在飞 token 数/请求 (config, 例 512k)

layout 选择: 按请求紧密排列 (dense-per-request), 不是 paged!
  → attention 消费时无需 block table, 只需 seq_lens
  → gather kernel 负责 "paged host 块 → 按请求紧密" 的压实
```

**内存账目（policy 必须计入）**：staging 不是免费的。

```
每层在飞量/序列 @512k        = 590 MB
staging 总量 = SLOTS × max_batch × 590MB
  例: 2 × 32 × 590MB ≈ 37.7 GB

净 HBM 节省 = (f − SLOTS) × max_batch × 590MB
  f=6: (6−2) × 32 × 590MB ≈ 75 GB   （不是裸算的 6/24）
```

推论：`f > SLOTS` 才有净收益；`SLOTS` 取 2（再深预取收益递减，见 §5.3 调度）。

### 3.3 新增/复用的固定地址 device tensor

| tensor | 形状 | 更新时机 | 用途 |
|---|---|---|---|
| `block_table[streaming 组]` | 复用现有，per-request | replay 前原地 | gather 源地址解析 |
| `staging[s]` | §3.2 | 图内 gather 写 | attention 消费 |
| `seq_lens` | 复用现有 | replay 前原地 | attention 消费 staging 的有效长度 |
| `slot_mapping` | 复用现有 | replay 前原地 | writeback 目标槽位解析 |

---

## 4. 函数抽象（签名级）

新包 `vllm_ascend/distributed/kv_transfer/streaming_kv/`。

### 4.1 StreamingKVManager（单例，worker 级）

```python
class StreamingKVManager:
    # ── 初始化 ──────────────────────────────────────────
    @classmethod
    def init(cls, vllm_config, kv_cache_config, plan: ResidencyPlan) -> "StreamingKVManager":
        """校验 plan 中的层确为 streaming 组; 分配 host 池/staging/copy_stream/events。"""

    # ── worker 注册（model_runner 初始化 kv cache 时调用, 与 sparse offload 同点位）──
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """对 streaming 层: 不分配持久 NPU cache, 返回 attention impl 可见的
        staging 视图描述; 记录 host 池基址 (gvas)。"""

    # ── 调度表（启动时从层型生成, 之后只读）─────────────────
    def schedule(self) -> "OnloadSchedule":
        """返回 [(issue_after_layer_idx, streaming_layer_idx, slot), ...]
        由 full_attn 层位自动生成, 处理 92/93 相邻等特例。"""

    # ── 图内/ eager 共用的事件原语（被 custom op 包装调用）──
    def start_onload(self, s: int, dep: torch.Tensor) -> None:
        """fork: compute stream 打点 → copy_stream.wait_event
        → gather(s, slot=s%2) → done_event[s].record(copy_stream)
        dep 仅为制造数据依赖 (mutates_args)。"""

    def wait_onload(self, s: int, dep: torch.Tensor) -> None:
        """join: compute stream wait done_event[s]; eager/捕获分支语义同 2.1。"""

    def gather(self, s: int, slot: int) -> None:
        """copy_stream 上执行: 逐请求按 block_table 把层 s 全部块
        从 host_pool 压实拷入 staging[slot]。实现见 §4.3。"""

    def writeback(self, s: int, dep: torch.Tensor) -> None:
        """层 s attention 之后: 新 KV 行 (staging[slot] 中 [req, seq_len] 位置)
        D2H 回 host_pool 对应块槽位; copy_stream 上, 图内可捕获。"""

    # ── 捕获生命周期（acl_graph.py 同点位调用）──────────────
    def sync_before_capture(self) -> None: ...   # 清在途拷贝
    def join_after_forward(self) -> None: ...     # 补齐图尾 dangling join
```

### 4.2 层 hook（包装每个 decoder layer，对应 2.1 的 `_hook_module_forward`）

```python
def wrap_decoder_layers(model, manager: StreamingKVManager):
    """对每个 decoder layer 包装 forward:
    on entry: 若本层是 streaming 层 s → manager.wait_onload(s, hidden)
    on exit : 若 schedule 在本层之后有 issue 点 → manager.start_onload(s', hidden)
              若本层是 streaming 层 s → manager.writeback(s, hidden)
    KDA 层同样有 hook (作为 issue 点载体), 但不改其计算路径。
    custom op: torch.ops.vllm_ascend.{wait_onload,start_onload,writeback}
    均声明 mutates_args=[dep], 保证进图且序正确。"""
```

为什么 hook 在 **decoder layer 粒度**而不是 attention impl 粒度：issue 点落在线性层（KDA 层没有 attention impl 的 kv 消费路径可挂），fork 必须发生在"前一个 full 层的 attention 完成之后"，只有 decoder-layer 边界能同时表达这两类事件点。attention impl 内部只需要知道"我的 kv_cache 是 staging[slot]"。

### 4.3 gather 的两种候选实现（spike 1.2 裁决）

```
候选 G1 — device 直读 gather kernel (首选):
  自定义 kernel, 输入 = block_table(device, 固定地址) + host_pool gvas 基址
  每个 thread block 负责 (请求, 块) → 从 host 地址 load → 写 staging
  本质: kernel 直接读 host 内存 (2.2 已证可寻址), 无需 SDMA 描述符
  优点: 单 kernel 进图, 无 host 往返; 读带宽=互联带宽
  风险: 小块随机读的效率 → spike 实测

候选 G2 — 间接 SDMA 批量拷贝 (保底):
  仿 sparse_copy: 先在图内由一个轻量 kernel/host_func 生成
  (src_gvas, dst, size) 描述符表 → SDMA batch 执行
  优点: 大块连续传输效率高; 缺点: 多一层计划生成
```

---

## 5. 时序设计（核心）

### 5.1 调度表生成（K3 实例）

层型：full 层 {4,8,…,88,92,93}，KDA 层其余。设 streaming 层为尾部 6 层 {68,72,76,80,84,88}（示例），SLOTS=2：

```
issue(s) 位置 = 层 s-3 (KDA 层) 的 exit hook       # 3 层计算窗口
join(s)  位置 = 层 s   (full 层) 的 entry hook
fork 安全约束: onload(s) 写 slot=s%2, 上一占用者是 streaming 层 s-2,
              故 fork 事件必须打在 s-2 的 attention 之后
              → 已天然满足: s-3 在 s-2(=s-8) 之后 ✓
writeback(s) 位置 = 层 s 的 exit hook (copy stream 上, 顺序在 onload(s) 之后)
```

### 5.2 正确性论证：预取为什么不能跨步

权重 prefetch 的循环调度（最后几层为**下一 forward** 的前几层预取）对我们是**错误**的：

```
若 onload(s=第一层streaming) 在 step N-1 的图尾部发射:
  gather 读 block_table ──→ 此时还是 step N-1 的值!
  而 step N 前 CPU 可能: 新请求进 batch / 追加新块 / 抢占恢复
  → step N 的 attention 读到陈旧/错误的 KV   ✗

权重无此问题 (源内容恒定); KV 的"哪些块"是逐步变化的。

结论: onload 调度必须满足
  issue(s) 的执行时点 > 本步 block_table 更新时点
  即: 所有 onload 都在本步 forward 内部发射, 窗口=层内距离, 不跨步。
  (K3 第一个 full 层在 4, 前面有 1-3 层做窗口, 足够; 窗口深度
   上限=本步内层间距离, 想更深只能放宽到 issue 全部在层 0 发射,
   copy stream 按序流完整个 step —— 见 5.3 变体 B)
```

### 5.3 时序图（一步内，eager 与 replay 同构）

变体 A（逐层 fork，默认）：

```
compute stream:  L65  L66  L67  │F68│  L69 L70 L71 │F72│  L73 L74 L75 │F76│ ...
                                wait68            wait72            wait76
                   ▲              ▲                 ▲                 ▲
                   │              │done68           │done72           │done76
copy stream:     fork68        gather68          gather72          gather76
                (L65 exit)      (slot0)           (slot1)           (slot0)
                                └─writeback68 (F68 exit 后, 同 stream 顺序)
attention F68 读 staging[0]; F72 读 staging[1]; F76 读 staging[0] (已被 gather76 重写,
安全性: gather76 的 fork 在 F72 之后? 否——slot0 上一占用者 F68, fork76 在 L73,
即 F72 之后 > F68 之后 ✓)
```

变体 B（步首集中发射，窗口最大化）：

```
层 0 hook: 一次性把 onload(s1..s6) 全部 issue 到 copy stream
  copy stream 顺序执行 gather(s1)→gather(s2)→… (流内天然有序)
  每个 gather 前有 wait_event(fork_s), fork_s 打在 slot 上一占用者之后
compute stream 在各 full 层前 wait_event(done_s)
→ copy stream 全步连续满载, 带宽利用率最高; 事件 pacing 保证安全
```

两变体共用同一组事件原语，只是 fork 打点位置不同；都进图。默认 A，B 作为带宽不足时的调优档。

### 5.4 捕获 / replay 时序

```
捕获 (每个 batch_descriptor 一次):
  sync_before_capture()                      # 清在途
  with torch.npu.graph(g):
    model.forward()                          # 包装层发射 wait/start/writeback
                                             # custom ops → 事件与拷贝全部入图
    manager.join_after_forward()             # 补 dangling join (无跨步fork后,
                                             #  仅剩最后一个 writeback 的 join)
replay (每步):
  CPU: scheduler → block_table/seq_lens/slot_mapping 原地更新
  NPU: graph.replay()                        # gather/writeback 按捕获时的事件
                                             # 依赖执行, 源地址 replay 时从
                                             # block_table 解析 → 本步正确数据
```

**继承的坑（来自 2.1，直接沿用其解法）**：捕获中 record 的 event 捕获后失效 → eager/捕获分支标记；非 pinned host 内存的 non_blocking 拷贝会隐式同步破坏 fork → host 池必须 pinned；TP 下禁止图内 barrier（用 broadcast，sparse offload 已有先例）。

### 5.5 MTP / spec decode

每步 token 数 = batch×(1+k)：`gather` 不变（按块搬历史）；`writeback` 按 slot_mapping 覆盖 1+k 行/请求；staging 的 W_max 按 `seq_len + 1 + k` 预留。attention 消费 staging 时 seq_lens 含 spec token——与现有 MLA decode 路径同一约定。

### 5.6 padding 与变 batch

图按固定 batch 捕获，不足时 pad：pad 请求的 block_table 行为 0，gather 会多拷一份块 0 的内容（无害，attention 对 pad 行 mask）。优化：gather kernel 内按 `num_actual_reqs`（固定地址 tensor）早退，跳过 pad 行——`num_actual_reqs` 已是图内可读的现有机制。

---

## 6. 与 model_runner 的接线点（对照现有代码位置）

| 点位 | 现有参照 | 我们挂什么 |
|---|---|---|
| spec 创建 | `store_on_host` 标记（model_runner_v1.py:4742 sparse 用法） | policy 选定层标 True |
| kv cache 注册 | `sparse_kv_offload_manager.register_kv_caches`（model_runner_v1.py:3675-3708） | `StreamingKVManager.register_kv_caches` |
| staging 分配 | `allocate_kv_cache_tensors_for_sparse_kv_offload`（model_runner_v1.py:4053-4066 同区域） | staging 双缓冲分配 |
| decoder layer 包装 | 权重 prefetch `_hook_module_forward` | `wrap_decoder_layers`（模型加载后） |
| metadata 更新 | `update_sparse_kv_offload_metadata`（model_runner_v1.py:2844） | 无需新增（复用 block_table/seq_lens 原地更新） |
| 捕获生命周期 | `acl_graph.py:180-194` `get_offloader().sync_prev_onload()/join_after_forward()` | 同点位调 manager 对应方法 |
| attention 消费 | `sfa_kv_offload.py`（impl 子类化） | streaming MLA impl：kv_cache=staging[slot]，seq_lens 直读 |

---

## 7. 落地前必须实测的三个探针（对应 tasks 1.x）

| 探针 | 裁决什么 | 失败时的退路 |
|---|---|---|
| P1 host 读带宽（device 直读 vs SDMA） | G1/G2 选择；B_h2d 实测值 → policy 参数 | 无退路，是方案前提 |
| P2 MLA decode kernel 接受非 paged 连续 KV 缓冲 + seq_lens | staging 紧凑 layout 可行 | 退为 paged staging + 合成 block table（gather 多一步地址换算） |
| P3 单步内 fork/join 事件链 ×6 层的捕获开销与 replay 正确性 | 变体 A/B 选择 | 减 streaming 层数 / eager 档运行 |

---

## 8. 一句话可行性结论

图模式所需的**每一项**机制（fork/join 捕获、静态 buffer 轮转、图内间接寻址、host 可寻址、in-place metadata、捕获生命周期 hook）在仓库内都有生产级先例且本文已定位到行号；新增的实质性约束只有一条——**预取不跨步**（§5.2），它由调度表静态保证，不引入新机制。剩余不确定性收敛为 P1–P3 三个可独立执行的小探针。
