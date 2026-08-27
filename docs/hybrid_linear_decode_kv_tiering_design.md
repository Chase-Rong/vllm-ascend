# Hybrid Linear 模型 Decode 侧 KV 分层驻留设计文档

> 对应 OpenSpec change：`openspec/changes/hybrid-linear-decode-kv-tiering/`
> 本文档面向整体 review：包含背景推导、量化依据、架构设计、先例证据与风险清单。
> OpenSpec 的 proposal/design/spec/tasks 是正式契约，本文档是完整叙述版。

## 1. 背景与动机

### 1.1 问题

Hybrid linear-attention 模型（Kimi K3/KDA、Qwen3.5/GDN、MiniMax M3）把大部分序列状态放在常量大小的线性注意力 state 里，但 full attention 层的 dense KV 仍随序列长度线性增长。在 PD 分离部署中，Decode 节点要为每个请求**常驻**保存这份 KV 并在每个 decode step **全量重读**——它构成 Decode 侧 HBM 的主要占用，直接限制 batch size 与吞吐。

以 Kimi K3 真实配置（93 层 = 69 KDA + 24 MLA）在 512k 上下文估算：

| 项 | 数值 |
|---|---|
| MLA KV/token/层 | (512 kv_lora + 64 rope) × 2B = **1152 B** |
| 单序列单 full 层 @512k | **590 MB** |
| 单序列 24 个 full 层合计 | **≈ 14.2 GB** |
| KDA state（69 层，常量） | ≈ 210 MB/序列，不随序列增长 |

### 1.2 机会

昇腾 950 级别硬件的 host↔NPU 带宽远超传统 PCIe。若把一部分 full attention 层的 KV 放到 CPU DRAM，每步流式 H2D 并与相邻线性层的计算重叠，即可用 host 带宽换 HBM 容量：省出的 HBM 全部转化为更大的 batch。

**吞吐天花板公式**（f = offload 的 full 层比例，B_h2d = host 带宽）：

```
聚合 TPS ≤ B_h2d / (f × 14.2GB)        （与 batch 无关的硬上限）
单序列 HBM 占用 = (1−f) × 14.2GB
```

| B_h2d | f=1/4（卸 6 层，省 3.5GB/序列） | f=1/2（卸 12 层，省 7.1GB/序列） |
|---|---|---|
| 400 GB/s | ≈ 113 TPS | ≈ 56 TPS |
| 1 TB/s | ≈ 282 TPS | ≈ 141 TPS |
| 2 TB/s | ≈ 564 TPS | ≈ 282 TPS |

对照：全驻留 HBM（3.2TB/s）的 attention 读取上限 ≈ 226 TPS。**B_h2d 到 ~1TB/s 量级时，f=1/4 的吞吐天花板已高于 HBM 驻留基准**，方案从"容量逃逸阀"变成"吞吐正收益"。

### 1.3 为什么天然适合 hybrid linear 模型

K3 的层型是严格的 `KDA×3 → MLA×1` 循环（仅末尾 92/93 相邻）。每个 full 层前面有 3 个**无 KV 负担**的线性层 + MoE 计算，构成天然的 overlap 窗口；KDA state 常量大小、永远驻留 HBM，不参与卸载。纯 dense 模型没有这种窗口且 KV 全部需要卸载，性价比低——因此机制严格限定 hybrid linear 模型（判定条件：KV cache config 中 MambaSpec 组与 full attention 组共存，机器可判定，不绑任何模型名）。

### 1.4 与现有三种 offload 的关系

| 机制 | 场景 | 与本方案关系 |
|---|---|---|
| KV Cache CPU Offload（prefix L2） | 不活跃 prefix 块二级缓存 | 不同问题 |
| Recompute CPU Offload | Decode 侧**被抢占请求**的保全（按请求、全层） | 不同触发；可视为互补 |
| Sparse KV Offload | 稀疏注意力模型 decode，top-k 按需取块 | **通路先例**，但服务稀疏模型且无 overlap |
| **本方案（Streaming KV Tiering）** | hybrid linear 模型 decode，dense 层按层流式 + overlap | 第四种独立机制 |

## 2. 总体架构

### 2.1 三足组结构（方案 A：按驻留拆组）

启动时由静态 policy 选定 streaming 层集合，通过 spec 级 `store_on_host` 标记自然拆组：

```
kv_cache_groups:
  [0] MambaSpec group              → KDA 层        (HBM 常驻, 不动)
  [1] AscendMLAAttentionSpec, store_on_host=False → 18 个 MLA 层 (HBM 常驻)
  [2] AscendMLAAttentionSpec, store_on_host=True  →  6 个 MLA 层 (CPU 持久 + HBM 暂存)

scheduler 记账:  [0][1] 占 HBM num_blocks 预算; [2] 占 CPU 预算(bytes→blocks)
prefix cache:    [1][2] 同为 FullAttentionSpec, 跨组命中对齐
                 (DeepSeek-V4 C4/C128 双 full-attn 组已有先例)
```

选择拆组（而非组内层属性）的核心理由：**驻留本质上是容量记账问题**，而容量记账是组级的。streaming 组的 block table 照常维护（prefix cache 兼容），只是 worker 侧其"KV tensor"分配为 pinned host 内存，NPU 侧每层只有一个固定 staging buffer。

### 2.2 数据通路（形态 1：图内间接 gather + fork/join copy stream）

```
decode step (每个 streaming 层 i):

  层 i-k 的 attention 之后 (k = 预取深度, ≈ 3 KDA + 1):
    start_onload(i):  fork ─→ copy stream
                       gather kernel 读 block table[i] (固定地址, 原地更新)
                       按块从 host pool → HBM staging buffer[i % 2]
  层 i-1..i-k 的线性层计算 (compute stream)   ←─ 与 H2D 并发
  层 i 的 attention 之前:
    wait_onload(i):   join ─→ compute stream 等待拷贝完成
    attention kernel 消费 staging buffer (layout 与 paged cache 一致)

  层 i 的 attention 之后:
    writeback: 新 KV 行 → append buffer → copy stream D2H 回 host pool 槽位
```

### 2.3 KV 的一生（端到端不占持久 HBM）

```
Prefill 节点                     网络               Decode 节点
┌─────────────────┐                              ┌──────────────────┐
│ 层 i 算出 KV     │                              │                  │
│   ↓ layerwise   │   host-to-host                │  host DRAM 持久池 │
│   D2H (已有机制) │ ──────────────────────→      │   ↓ 每步流式 H2D  │
│ prefill host 池  │  (SfaRemoteD2HConnector 模式) │ staging 双缓冲    │
└─────────────────┘                              │   ↓              │
                                                 │ attention kernel  │
                                                 │   ↓ 新 KV D2H 回写│
                                                 │  回 host 池       │
                                                 └──────────────────┘
```

## 3. 关键设计决策（含备选方案与理由）

### D1 驻留标记用 spec 级 `store_on_host` 拆组
- 依据：`merge()` 要求组内 `store_on_host` 一致，标记不同的层**自然**落入不同组，无需调度器手术。
- 备选（组内层属性）：打破组内同构假设，scheduler/block pool/prefix cache 三处穿孔，否决。

### D2 streaming 组按 CPU 预算记账
- block table 粒度与常驻组一致（prefix cache 兼容）；容量检查走 CPU 预算；worker 侧分配 pinned host 池。
- prefix 命中块在 host 池中，后续请求命中后照常流式消费，无需特殊处理。

### D3 图内间接 gather + fork/join copy stream
- **间接寻址**（源地址每步变化）→ 借用 sparse offload 已验证的模式：block table 固定地址原地更新，device 侧 kernel replay 时解析；host 内存 device 可寻址（gvas）。dense 无 miss 逻辑，不需要 host-func 往返。
- **overlap** → 借用权重预取 offloader 已验证的模式：`start/wait` fork/join、`StaticBufferPool` 双缓冲轮转、`prefetch_step` 跨层预取；捕获生命周期 hook（`sync_prev_onload`/`join_after_forward`）在 `compilation/acl_graph.py` 已埋好。
- 备选：
  - 零拷贝直连（attention kernel 直接读 host 内存）——最简，待 kernel 探针验证（Open Question），若可行作为 fast path；
  - 同流内联拷贝——无 overlap，否决；
  - 分段 graph replay + 段间 eager 拷贝——fork/join 失效时的 fallback。

### D4 写回与 PD 交接
- decode 新 KV：append buffer → copy stream D2H（图内可捕获）。
- prefill 侧：复用 layerwise 写出 + host-to-host PD 传输（`SfaRemoteD2HConnector` 模式），streaming 层 KV 在两端都不占持久 HBM。

### D5 静态 residency policy
- 输入：每 rank HBM KV 预算、`max_model_len`、spec page size、B_h2d（配置或初始化微基准实测）、目标 TPS/batch。
- 策略：从网络尾部开始选最小 streaming 层集合，满足 `TPS_ceiling = B_h2d/(f×KV_bytes) ≥ target`；`stream_layers` 显式配置可覆盖。
- 运行时不变。动态 per-request 驻留与 ACLGraph 结构性冲突（内存计划变化、数据路径不固定），其收益（短序列不付流式代价）以"staging 作为 host 块缓存"的后续优化形式保留。

### D6 激活门控
- 条件：配置开关 + MambaSpec 组与 full attention 组共存 + 一期 streaming 层全为 `AscendMLAAttentionSpec`。
- 纯 dense 模型：启动报错；Qwen3.5（GQA full 层）：二期支持（通路同构，page layout 不同）。

### D7 模块划分
- 新增 `vllm_ascend/distributed/kv_transfer/streaming_kv/`：`policy.py`（驻留计划）、`manager.py`（host 池、staging、onload/writeback、生命周期）、`copy_pipeline.py`（fork/join、捕获集成）。
- spec 标记挂在模型 spec 创建路径；worker 钩子与现有 sparse offload 调用点并列。
- 与三种现有 offload connector 互斥校验（同时配置则启动报错）。

## 4. 先例证据汇总（可行性依据）

| 需要的能力 | 先例 | 状态 |
|---|---|---|
| 图内 host↔device 数据移动 | sparse offload：`_launch_host_func` + `sparse_copy`（gvas），`capturing=True` 下运行 | ✅ 生产在跑 |
| 源地址每步变化 | block table 固定地址原地更新 + 间接 kernel | ✅ 同上 |
| 图内双流 fork/join overlap | 权重预取 offloader（`prefetch.py` + `acl_graph.py` 的 join/sync 处理） | ✅ 生产在跑 |
| 静态 buffer 池轮转 | `StaticBufferPool`（NZ 格式适配都有） | ✅ 同上 |
| 多 full-attn 组的 prefix 命中 | DeepSeek-V4 C4/C128 | ✅ 已在 coordinator |
| host 池 TP 共享 | sparse offload "rank0 建池 + gvas 广播" | ✅ 可参考 |
| prefill 写路径分流 | layerwise prefill offload | ✅ 已有 |

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 950 host 带宽是假设非实测 | 独立 spike 先行；初始化微基准；policy 按实测值计算，低于目标则告警/拒绝 |
| MLA decode kernel 不接受 staging/host 指针 | staging 复刻 paged layout + 合成连续 block table；kernel 探针为任务 1.2 |
| 高 batch 下单 copy stream 超出 overlap 窗口 | 预取深度 k 可调；profile 驱动调优；必要时调 f 或降 batch |
| MTP/DSpark 改变每步 token 数与 slot 映射 | onload/writeback 按 1+k 行/请求设计；spec 场景覆盖；e2e 验证 |
| host 池成为新的准入维度，运维需 sizing | 启动日志输出每序列 host 字节与最大序列数；文档给公式；默认预算保守 |
| 与现有 offload connector 混配 | 启动互斥校验，明确报错 |

## 6. 实施分期

- **Spike（任务 1.x）**：950 host 带宽基准（SDMA vs device 直读）；MLA kernel 指针探针。
- **一期**：eager 模式跑通（policy/拆组/记账/host 池/gather/writeback/overlap/PD 交接），K3 长上下文正确性（与全 HBM 基线 token 级一致）+ 带宽账验证。
- **二期**：ACLGraph 集成（捕获安全化 + MTP），950 吞吐基准。
- **三期（本 change 仅跟踪）**：GQA `FullAttentionSpec` streaming（Qwen3.5）。

## 7. 已记录的后续优化（non-goals）

1. staging buffer 作为 host 块的 HBM 缓存：block 级命中策略（纯运行时行为，不改图结构），短序列请求自然免于流式代价；
2. resident 组的承压降级（复用 recompute offload 思路）；
3. prefill 侧读回流（chunked prefill 超长 prompt 回读 host KV）；
4. 零拷贝直连 fast path（待 kernel 探针结论）。
