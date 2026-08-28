# SPDX-License-Identifier: Apache-2.0
"""UT for the Kimi K3 Mooncake swap-memory receive path.

The tests are split so that they can run on a host without an Ascend NPU:

* ``TestSwapMemoryRegistry`` and ``TestSwapStagingPlanner`` use CPU tensors and
  a fake transfer engine.  They cover address planning, the 2 MiB split, batch
  windowing and error propagation.
* ``TestSwapStagingOnNPU`` is skipped unless a real NPU with
  ``torch_npu.empty_with_swapped_memory`` is present.  It covers the parts that
  can only be judged against real swap memory.
"""

import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_hybrid_connector as connector_module,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    SWAP_STAGING_ALIGNMENT,
    SWAP_STAGING_WINDOW,
    KVCacheRecvingThread,
    KVCacheTaskTracker,
    MooncakeConnector,
    MooncakeConnectorWorker,
)
from vllm_ascend.kv_offload.mooncake_swap_memory import (
    clear_swapped_tensors_for_testing,
    empty_swapped_memory,
    get_swapped_tensor,
    is_swapped_range,
    register_swapped_tensor,
)

ALIGN = SWAP_STAGING_ALIGNMENT


def _npu_swap_available() -> bool:
    try:
        import torch_npu
    except ImportError:
        return False
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        return False
    return bool(getattr(torch, "npu", None)) and torch.npu.is_available()


class FakeEngine:
    """Records every batch_transfer_sync_read call and replays scripted rets."""

    def __init__(self, rets=None):
        self.calls: list[tuple[str, list[int], list[int], list[int]]] = []
        self._rets = list(rets or [])

    def batch_transfer_sync_read(self, session_id, local_dst, remote_src, lengths):
        self.calls.append((session_id, list(local_dst), list(remote_src), list(lengths)))
        if self._rets:
            return self._rets.pop(0)
        return 0


class StagingThreadStub:
    """Binds the real staging methods onto a minimal, CPU-friendly object.

    Only the attributes the staging helpers touch are provided, so the tests
    exercise the production code without constructing a full connector or
    starting a thread.
    """

    _swap_staging_chunk_bytes = KVCacheRecvingThread._swap_staging_chunk_bytes
    _ensure_swap_staging = KVCacheRecvingThread._ensure_swap_staging
    _copy_staging_to_swapped = KVCacheRecvingThread._copy_staging_to_swapped
    _batch_transfer_sync_read_with_swap_staging = KVCacheRecvingThread._batch_transfer_sync_read_with_swap_staging

    def __init__(self, block_len_per_addr, engine, staging_numel=None):
        self.block_len_per_addr = list(block_len_per_addr)
        self.engine = engine
        self._swap_staging_pools: dict[threading.Thread, tuple[torch.Tensor, torch.Tensor]] = {}
        self._swap_staging_pool_lock = threading.Lock()
        # A CPU tensor stands in for the registered NPU staging buffer; the
        # planner only needs data_ptr arithmetic and a byte-addressable view.
        self._staging_backing = torch.zeros(staging_numel or 1, dtype=torch.int8)
        if staging_numel is not None:
            self._swap_staging_pools[threading.current_thread()] = (
                self._staging_backing,
                self._staging_backing,
            )
        self.kv_caches = {"layer0": self._staging_backing}
        self.copies: list[tuple[int, int, int]] = []

    def current_staging(self):
        return self._swap_staging_pools[threading.current_thread()][1]

    def record_copy(self, dst, length, staging, staging_offset):
        self.copies.append((dst, length, staging_offset))


class TestSwapMemoryRegistry(unittest.TestCase):
    def setUp(self):
        clear_swapped_tensors_for_testing()
        self.addCleanup(clear_swapped_tensors_for_testing)

    def test_resolves_exact_and_sub_ranges(self):
        tensor = torch.zeros(4096, dtype=torch.int8)
        register_swapped_tensor(tensor)
        base = tensor.data_ptr()

        self.assertTrue(is_swapped_range(base, 4096))
        owner, offset = get_swapped_tensor(base + 100, 200)
        self.assertIs(owner, tensor)
        self.assertEqual(offset, 100)

    def test_rejects_ranges_outside_registered_allocation(self):
        tensor = torch.zeros(4096, dtype=torch.int8)
        register_swapped_tensor(tensor)
        base = tensor.data_ptr()

        # One byte past the end must not resolve, otherwise the connector would
        # copy outside the allocation.
        self.assertFalse(is_swapped_range(base, 4097))
        self.assertFalse(is_swapped_range(base + 4096, 1))
        self.assertIsNone(get_swapped_tensor(base - 1, 8))

    def test_unregistered_pointer_is_not_swapped(self):
        other = torch.zeros(64, dtype=torch.int8)
        self.assertFalse(is_swapped_range(other.data_ptr(), 64))

    def test_registration_is_idempotent(self):
        tensor = torch.zeros(256, dtype=torch.int8)
        register_swapped_tensor(tensor)
        register_swapped_tensor(tensor)
        owner, offset = get_swapped_tensor(tensor.data_ptr(), 256)
        self.assertIs(owner, tensor)
        self.assertEqual(offset, 0)

    def test_rejects_empty_allocation(self):
        with self.assertRaises(ValueError):
            register_swapped_tensor(torch.zeros(0, dtype=torch.int8))


class TestSwapStagingPlanner(unittest.TestCase):
    """Address planning for _batch_transfer_sync_read_with_swap_staging."""

    def setUp(self):
        clear_swapped_tensors_for_testing()
        self.addCleanup(clear_swapped_tensors_for_testing)
        self.block_len = 4096

    def _make(self, engine, staging_slots=4):
        slot_bytes = ALIGN + ALIGN  # chunk_bytes rounds up to ALIGN, plus window
        return StagingThreadStub([self.block_len], engine, slot_bytes * staging_slots)

    def _register_swap_dst(self, numel=ALIGN):
        swap = torch.zeros(numel, dtype=torch.int8)
        register_swapped_tensor(swap)
        return swap

    def test_ordinary_destination_uses_direct_read_only(self):
        engine = FakeEngine()
        stub = self._make(engine)
        plain = torch.zeros(self.block_len, dtype=torch.int8)

        ret = stub._batch_transfer_sync_read_with_swap_staging("s1", [plain.data_ptr()], [0x7000], [self.block_len])

        self.assertEqual(ret, 0)
        self.assertEqual(len(engine.calls), 1)
        _, local, remote, lengths = engine.calls[0]
        self.assertEqual(local, [plain.data_ptr()])
        self.assertEqual(remote, [0x7000])
        self.assertEqual(lengths, [self.block_len])

    def test_swap_destination_reads_into_staging_not_destination(self):
        engine = FakeEngine()
        stub = self._make(engine)
        swap = self._register_swap_dst()
        staging_ptr = stub.current_staging().data_ptr()

        with mock.patch.object(StagingThreadStub, "_copy_staging_to_swapped", StagingThreadStub.record_copy):
            ret = stub._batch_transfer_sync_read_with_swap_staging(
                "s1", [swap.data_ptr()], [ALIGN * 3], [self.block_len]
            )

        self.assertEqual(ret, 0)
        self.assertEqual(len(engine.calls), 1)
        _, local, remote, lengths = engine.calls[0]
        # The destination handed to Mooncake must be the staging buffer.
        self.assertNotEqual(local, [swap.data_ptr()])
        self.assertEqual(local, [staging_ptr])
        self.assertEqual(remote, [ALIGN * 3])
        self.assertEqual(lengths, [self.block_len])
        # And the data must then be copied into the swap destination.
        self.assertEqual(stub.copies, [(swap.data_ptr(), self.block_len, 0)])

    def test_mixed_batch_splits_direct_and_staged(self):
        engine = FakeEngine()
        stub = self._make(engine)
        swap = self._register_swap_dst()
        plain = torch.zeros(self.block_len, dtype=torch.int8)

        with mock.patch.object(StagingThreadStub, "_copy_staging_to_swapped", StagingThreadStub.record_copy):
            ret = stub._batch_transfer_sync_read_with_swap_staging(
                "s1",
                [plain.data_ptr(), swap.data_ptr()],
                [ALIGN, ALIGN * 2],
                [self.block_len, self.block_len],
            )

        self.assertEqual(ret, 0)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(engine.calls[0][1], [plain.data_ptr()])
        self.assertEqual(engine.calls[1][1], [stub.current_staging().data_ptr()])
        self.assertEqual(len(stub.copies), 1)

    def test_misaligned_remote_source_split_preserves_total_length(self):
        engine = FakeEngine()
        stub = self._make(engine)
        # A destination large enough that one block spans an alignment boundary.
        swap = self._register_swap_dst(numel=ALIGN * 4)
        length = ALIGN + 8192
        remote_base = ALIGN * 5 + 4096  # deliberately not 2 MiB aligned

        with mock.patch.object(StagingThreadStub, "_copy_staging_to_swapped", StagingThreadStub.record_copy):
            ret = stub._batch_transfer_sync_read_with_swap_staging("s1", [swap.data_ptr()], [remote_base], [length])

        self.assertEqual(ret, 0)
        staged_lengths = [n for call in engine.calls for n in call[3]]
        self.assertGreater(len(staged_lengths), 1, "expected the span to be split")
        self.assertEqual(sum(staged_lengths), length)
        self.assertTrue(all(n > 0 for n in staged_lengths))

        # Remote offsets must stay contiguous and cover the range exactly once.
        remotes = [r for call in engine.calls for r in call[2]]
        expected = remote_base
        for remote, piece in zip(remotes, staged_lengths):
            self.assertEqual(remote, expected)
            expected += piece
        self.assertEqual(expected, remote_base + length)

        # Copy destinations must mirror the same contiguous layout.
        self.assertEqual(sum(c[1] for c in stub.copies), length)
        expected_dst = swap.data_ptr()
        for dst, piece, _ in stub.copies:
            self.assertEqual(dst, expected_dst)
            expected_dst += piece

    def test_staged_piece_never_exceeds_slot_capacity(self):
        engine = FakeEngine()
        stub = self._make(engine)
        swap = self._register_swap_dst(numel=ALIGN * 4)
        chunk = stub._swap_staging_chunk_bytes()

        with mock.patch.object(StagingThreadStub, "_copy_staging_to_swapped", StagingThreadStub.record_copy):
            stub._batch_transfer_sync_read_with_swap_staging(
                "s1", [swap.data_ptr()], [ALIGN * 7 + 1024], [ALIGN * 2 + 512]
            )

        for call in engine.calls:
            for local, piece in zip(call[1], call[3]):
                slot_offset = local - stub.current_staging().data_ptr()
                self.assertLessEqual(
                    slot_offset % (chunk + ALIGN) + piece,
                    chunk + ALIGN,
                    "staged piece must not run past its slot",
                )

    def test_negative_engine_return_skips_copy(self):
        engine = FakeEngine(rets=[-1])
        stub = self._make(engine)
        swap = self._register_swap_dst()

        with mock.patch.object(StagingThreadStub, "_copy_staging_to_swapped", StagingThreadStub.record_copy):
            ret = stub._batch_transfer_sync_read_with_swap_staging("s1", [swap.data_ptr()], [ALIGN], [self.block_len])

        self.assertEqual(ret, -1)
        self.assertEqual(stub.copies, [], "no copy may run after a failed transfer")

    def test_direct_failure_short_circuits_before_staging(self):
        engine = FakeEngine(rets=[-2])
        stub = self._make(engine)
        swap = self._register_swap_dst()
        plain = torch.zeros(self.block_len, dtype=torch.int8)

        with mock.patch.object(StagingThreadStub, "_copy_staging_to_swapped", StagingThreadStub.record_copy):
            ret = stub._batch_transfer_sync_read_with_swap_staging(
                "s1",
                [plain.data_ptr(), swap.data_ptr()],
                [ALIGN, ALIGN * 2],
                [self.block_len, self.block_len],
            )

        self.assertEqual(ret, -2)
        self.assertEqual(len(engine.calls), 1, "staging must not run after a direct failure")
        self.assertEqual(stub.copies, [])

    def test_batches_are_copied_before_window_is_reused(self):
        """Slot reuse must not overwrite data that has not been copied yet.

        The staging window is capped at 16 slots, so more than 16 staged
        transfers are needed to force a second batch through the same slots.
        """
        engine = FakeEngine()
        stub = self._make(engine, staging_slots=16)
        swap = self._register_swap_dst(numel=ALIGN * 64)
        n = 20
        dsts = [swap.data_ptr() + i * self.block_len for i in range(n)]
        remotes = [ALIGN * (i + 4) for i in range(n)]
        lengths = [self.block_len] * n

        order: list[tuple[str, int]] = []

        def record_read(session_id, local, remote, length):
            order.append(("read", len(local)))
            return FakeEngine.batch_transfer_sync_read(engine, session_id, local, remote, length)

        def record_copy(self_, dst, length, staging, staging_offset):
            order.append(("copy", dst))

        engine_wrapper = mock.Mock(side_effect=record_read)
        stub.engine = mock.Mock(batch_transfer_sync_read=engine_wrapper)

        with mock.patch.object(StagingThreadStub, "_copy_staging_to_swapped", record_copy):
            ret = stub._batch_transfer_sync_read_with_swap_staging("s1", dsts, remotes, lengths)

        self.assertEqual(ret, 0)
        self.assertEqual(sum(1 for kind, _ in order if kind == "copy"), n)
        reads = sum(1 for kind, _ in order if kind == "read")
        self.assertGreater(reads, 1, "expected the window to be reused across batches")

        # Every read must be followed by its copies before the next read, so a
        # reused slot can never hold uncopied data.
        pending = 0
        for kind, payload in order:
            if kind == "read":
                self.assertEqual(pending, 0, "a new read started with copies outstanding")
                pending = payload
            else:
                pending -= 1
        self.assertEqual(pending, 0)


class TestSwapStagingPools(unittest.TestCase):
    def setUp(self):
        clear_swapped_tensors_for_testing()
        self.addCleanup(clear_swapped_tensors_for_testing)

    @staticmethod
    def _fake_allocations():
        real_empty = torch.empty
        allocations: list[int] = []
        allocations_lock = threading.Lock()

        def allocate_on_cpu(numel, *, dtype, device):
            del device
            with allocations_lock:
                allocations.append(numel)
            return real_empty(numel, dtype=dtype, device="cpu")

        fake_npu_tensor = mock.Mock(device=types.SimpleNamespace(type="npu"))
        return allocations, allocate_on_cpu, fake_npu_tensor

    def test_same_worker_allocates_and_registers_full_pool_once(self):
        stub = StagingThreadStub([4096], FakeEngine())
        allocations, allocate_on_cpu, fake_npu_tensor = self._fake_allocations()
        expected_bytes = (4096 + 1) * SWAP_STAGING_WINDOW

        with (
            mock.patch.object(connector_module, "SWAP_STAGING_ALIGNMENT", 1),
            mock.patch.object(
                connector_module,
                "iter_tensors",
                side_effect=lambda _: iter([fake_npu_tensor]),
            ),
            mock.patch.object(
                connector_module.torch,
                "empty",
                side_effect=allocate_on_cpu,
            ),
            mock.patch.object(connector_module.global_te, "register_buffer") as register,
        ):
            first = stub._ensure_swap_staging()
            second = stub._ensure_swap_staging()

        self.assertIs(first, second)
        self.assertEqual(first.numel(), expected_bytes)
        self.assertEqual(allocations, [expected_bytes + 1])
        register.assert_called_once_with([first.data_ptr()], [expected_bytes])
        self.assertEqual(len(stub._swap_staging_pools), 1)

    def test_concurrent_workers_use_distinct_pools_without_overwrite(self):
        class ConcurrentEngine:
            def __init__(self):
                self.owner = None
                self.barrier = threading.Barrier(2)
                self.staging_ptrs: dict[str, int] = {}
                self.lock = threading.Lock()

            def batch_transfer_sync_read(self, session_id, local_dst, remote_src, lengths):
                del remote_src
                with self.lock:
                    self.staging_ptrs[session_id] = local_dst[0]
                # Both workers must own a pool before either staged copy runs.
                self.barrier.wait(timeout=5)
                assert self.owner is not None
                with self.owner._swap_staging_pool_lock:
                    pools = list(self.owner._swap_staging_pools.values())
                for _, staging in pools:
                    start = staging.data_ptr()
                    offset = local_dst[0] - start
                    if offset >= 0 and offset + lengths[0] <= staging.numel():
                        staging.narrow(0, offset, lengths[0]).fill_({"worker-1": 11, "worker-2": 22}[session_id])
                        break
                else:
                    raise AssertionError("staging pointer did not resolve to a worker pool")
                self.barrier.wait(timeout=5)
                return 0

        engine = ConcurrentEngine()
        stub = StagingThreadStub([4096], engine)
        engine.owner = stub
        first_dst = torch.zeros(4096, dtype=torch.int8)
        second_dst = torch.zeros(4096, dtype=torch.int8)
        register_swapped_tensor(first_dst)
        register_swapped_tensor(second_dst)
        allocations, allocate_on_cpu, fake_npu_tensor = self._fake_allocations()

        def copy_without_npu_sync(self_, dst, length, staging, staging_offset):
            resolved = get_swapped_tensor(dst, length)
            assert resolved is not None
            target, target_offset = resolved
            target.view(torch.int8).reshape(-1).narrow(0, target_offset, length).copy_(
                staging.narrow(0, staging_offset, length)
            )

        def transfer(session_id, destination):
            return stub._batch_transfer_sync_read_with_swap_staging(
                session_id,
                [destination.data_ptr()],
                [ALIGN * 3],
                [destination.numel()],
            )

        with (
            mock.patch.object(connector_module, "SWAP_STAGING_ALIGNMENT", 1),
            mock.patch.object(
                connector_module,
                "iter_tensors",
                side_effect=lambda _: iter([fake_npu_tensor]),
            ),
            mock.patch.object(
                connector_module.torch,
                "empty",
                side_effect=allocate_on_cpu,
            ),
            mock.patch.object(connector_module.global_te, "register_buffer") as register,
            mock.patch.object(
                StagingThreadStub,
                "_copy_staging_to_swapped",
                copy_without_npu_sync,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [
                executor.submit(transfer, "worker-1", first_dst),
                executor.submit(transfer, "worker-2", second_dst),
            ]
            self.assertEqual([future.result(timeout=10) for future in futures], [0, 0])

        self.assertEqual(len(stub._swap_staging_pools), 2)
        self.assertNotEqual(engine.staging_ptrs["worker-1"], engine.staging_ptrs["worker-2"])
        self.assertTrue(torch.equal(first_dst, torch.full_like(first_dst, 11)))
        self.assertTrue(torch.equal(second_dst, torch.full_like(second_dst, 22)))
        expected_allocation = (4096 + 1) * SWAP_STAGING_WINDOW + 1
        self.assertEqual(allocations, [expected_allocation, expected_allocation])
        self.assertEqual(register.call_count, 2)


class FailureThreadStub:
    _handle_request = KVCacheRecvingThread._handle_request
    get_and_clear_invalid_block_ids = KVCacheRecvingThread.get_and_clear_invalid_block_ids
    _is_failed_recv_request = KVCacheRecvingThread._is_failed_recv_request
    _mark_failed_recv_request = KVCacheRecvingThread._mark_failed_recv_request
    _clear_failed_recv_request = KVCacheRecvingThread._clear_failed_recv_request

    def __init__(self, task_done_results):
        self.failed_recv_requests: set[str] = set()
        self.invalid_block_ids: set[int] = set()
        self.failed_recv_requests_lock = threading.Lock()
        self.use_hybrid = False
        self._transfer_kv_cache = mock.Mock(side_effect=RuntimeError("transfer failed"))
        self._transfer_kv_cache_all_groups = mock.Mock()
        self._mark_request_task_done = mock.Mock(side_effect=task_done_results)
        self._send_done_signal_to_free_remote_port = mock.Mock()
        self._send_done_recv_signal = mock.Mock()
        self.task_tracker = KVCacheTaskTracker()
        self.proc_not_transfer_request: dict[str, bool] = {}
        self.proc_not_transfer_request_lock = threading.Lock()
        self.request_queue = mock.Mock()


def _recv_task(local_block_ids, *, all_task_done):
    return {
        "request_id": "request-1",
        "remote_request_id": "remote-1",
        "remote_host": "127.0.0.1",
        "remote_handshake_port": 1234,
        "remote_port_send_num": {},
        "local_block_ids": local_block_ids,
        "all_task_done": all_task_done,
    }


class TestMooncakeLoadFailureReporting(unittest.TestCase):
    def test_failed_blocks_are_returned_and_cleared_through_connector_api(self):
        recv_thread = FailureThreadStub([True])
        recv_thread.task_tracker.add_req_to_process("request-1")
        recv_thread._handle_request(_recv_task(((3, 7), (11,)), all_task_done=True))

        worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_consumer"
        worker.kv_recv_thread = recv_thread
        connector = MooncakeConnector.__new__(MooncakeConnector)
        connector.connector_worker = worker

        self.assertEqual(connector.get_block_ids_with_load_errors(), {3, 7, 11})
        self.assertEqual(connector.get_block_ids_with_load_errors(), set())
        self.assertEqual(recv_thread.get_and_clear_finished_requests(), {"request-1"})

    def test_later_task_skips_transfer_after_request_failure(self):
        recv_thread = FailureThreadStub([False, True])
        recv_thread.task_tracker.add_req_to_process("request-1")

        recv_thread._handle_request(_recv_task(((1, 2), (3,)), all_task_done=False))
        recv_thread._handle_request(_recv_task(((4,), (5, 6)), all_task_done=True))

        recv_thread._transfer_kv_cache.assert_called_once()
        self.assertEqual(recv_thread.get_and_clear_invalid_block_ids(), {1, 2, 3, 4, 5, 6})
        self.assertEqual(recv_thread.get_and_clear_finished_requests(), {"request-1"})


class TestSwappedMemoryAllocation(unittest.TestCase):
    def test_allocator_result_is_explicitly_zeroed(self):
        allocated = torch.full((32,), 9, dtype=torch.int8)
        allocator = mock.Mock(return_value=allocated)
        fake_torch_npu = types.SimpleNamespace(empty_with_swapped_memory=allocator)

        with mock.patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            result = empty_swapped_memory((32,), dtype=torch.int8)

        self.assertIs(result, allocated)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        allocator.assert_called_once_with((32,), dtype=torch.int8, device="npu")


class TestGlobalTERegistration(unittest.TestCase):
    """Incremental (ptr, size, location) registration in GlobalTE."""

    def _make_te(self, with_location_api=False, ret=0):
        from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE

        te = GlobalTE()
        engine_attrs = {"register_memory.return_value": ret}
        engine = mock.Mock(**engine_attrs)
        if with_location_api:
            engine.register_memory_with_location = mock.Mock(return_value=ret)
        else:
            # Mock auto-creates attributes, so remove it explicitly.
            del engine.register_memory_with_location
        te.transfer_engine = engine
        return te, engine

    def test_first_registration_calls_register_memory(self):
        te, engine = self._make_te()
        te.register_buffer([0x1000, 0x2000], [64, 128])

        engine.register_memory.assert_has_calls([mock.call(0x1000, 64), mock.call(0x2000, 128)], any_order=False)
        self.assertEqual(engine.register_memory.call_count, 2)

    def test_duplicate_registration_is_skipped(self):
        te, engine = self._make_te()
        te.register_buffer([0x1000], [64])
        te.register_buffer([0x1000], [64])

        self.assertEqual(engine.register_memory.call_count, 1)

    def test_staging_buffer_registers_after_kv_caches(self):
        """The old is_register_buffer flag short-circuited later buffers."""
        te, engine = self._make_te()
        te.register_buffer([0x1000], [64])
        self.assertTrue(te.is_register_buffer)

        # A staging buffer allocated later must still be registered.
        te.register_buffer([0x9000], [ALIGN])

        engine.register_memory.assert_called_with(0x9000, ALIGN)
        self.assertEqual(engine.register_memory.call_count, 2)

    def test_same_pointer_different_size_registers_again(self):
        te, engine = self._make_te()
        te.register_buffer([0x1000], [64])
        te.register_buffer([0x1000], [128])

        self.assertEqual(engine.register_memory.call_count, 2)

    def test_mismatched_size_count_raises(self):
        te, _ = self._make_te()
        with self.assertRaises(ValueError):
            te.register_buffer([0x1000, 0x2000], [64])

    def test_mismatched_location_count_raises(self):
        te, _ = self._make_te()
        with self.assertRaises(ValueError):
            te.register_buffer([0x1000, 0x2000], [64, 128], ["cpu"])

    def test_location_registration_uses_location_api(self):
        te, engine = self._make_te(with_location_api=True)
        te.register_buffer([0x1000], [64], ["cpu:0"])

        engine.register_memory_with_location.assert_called_once_with(0x1000, 64, "cpu:0")
        engine.register_memory.assert_not_called()

    def test_missing_location_api_raises_runtime_error(self):
        te, _ = self._make_te(with_location_api=False)
        with self.assertRaises(RuntimeError):
            te.register_buffer([0x1000], [64], ["cpu:0"])

    def test_location_and_default_are_tracked_separately(self):
        te, engine = self._make_te(with_location_api=True)
        te.register_buffer([0x1000], [64])
        te.register_buffer([0x1000], [64], ["cpu:0"])

        engine.register_memory.assert_called_once_with(0x1000, 64)
        engine.register_memory_with_location.assert_called_once_with(0x1000, 64, "cpu:0")

    def test_failed_registration_raises_and_is_not_cached(self):
        te, engine = self._make_te(ret=-1)
        with self.assertRaises(RuntimeError):
            te.register_buffer([0x1000], [64])

        # A failed range must not be remembered as registered.
        engine.register_memory.return_value = 0
        te.register_buffer([0x1000], [64])
        self.assertEqual(engine.register_memory.call_count, 2)


def _load_gate_function():
    """Compile _use_kimi_k3_pd_swap_memory straight out of model_runner_v1.py.

    ``model_runner_v1`` pulls in the whole worker stack, which cannot be
    imported in a bare UT environment.  The gate is self-contained, so the
    function is extracted from the real source file and compiled on its own.
    That keeps the test bound to the shipped code rather than a copy.
    """
    import ast
    import pathlib

    import vllm_ascend

    source_path = pathlib.Path(vllm_ascend.__file__).parent / "worker" / "model_runner_v1.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    target = "_use_kimi_k3_pd_swap_memory"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == target:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace: dict = {}
            exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
            return namespace[target]
    raise AssertionError(f"{target} not found in {source_path}")


class _GateStub:
    def __init__(self, kv_transfer_config, model_type):
        self.vllm_config = mock.Mock(kv_transfer_config=kv_transfer_config)
        self.model_config = mock.Mock(hf_config=mock.Mock(model_type=model_type))


def _kv_cfg(role="kv_consumer", connector="MooncakeHybridConnector", extra=None):
    return mock.Mock(
        kv_role=role,
        kv_connector=connector,
        kv_connector_extra_config=extra if extra is not None else {},
    )


class TestKimiK3SwapGate(unittest.TestCase):
    """Risk 7.4: the gate must not widen to other models or to Prefill."""

    @classmethod
    def setUpClass(cls):
        cls.gate = staticmethod(_load_gate_function())

    def _call(self, kv_transfer_config, model_type):
        return type(self).gate(_GateStub(kv_transfer_config, model_type))

    def test_enabled_for_k3_consumer_with_hybrid_connector(self):
        self.assertTrue(self._call(_kv_cfg(), "kimi_k3"))

    def test_disabled_without_kv_transfer_config(self):
        self.assertFalse(self._call(None, "kimi_k3"))

    def test_disabled_for_producer_role(self):
        self.assertFalse(self._call(_kv_cfg(role="kv_producer"), "kimi_k3"))

    def test_disabled_for_other_connectors(self):
        self.assertFalse(self._call(_kv_cfg(connector="MooncakeConnector"), "kimi_k3"))

    def test_disabled_for_kimi_linear_text_config(self):
        """The K3 text sub-config keeps model_type kimi_linear."""
        self.assertFalse(self._call(_kv_cfg(), "kimi_linear"))

    def test_disabled_for_unrelated_model(self):
        self.assertFalse(self._call(_kv_cfg(), "deepseek_v3"))

    def test_enabled_via_nested_connector_list(self):
        extra = {"connectors": [{"kv_connector": "MooncakeHybridConnector"}]}
        cfg = _kv_cfg(connector="MultiConnector", extra=extra)
        self.assertTrue(self._call(cfg, "kimi_k3"))

    def test_nested_connector_list_without_hybrid_is_disabled(self):
        extra = {"connectors": [{"kv_connector": "SomethingElse"}]}
        cfg = _kv_cfg(connector="MultiConnector", extra=extra)
        self.assertFalse(self._call(cfg, "kimi_k3"))

    def test_connector_name_match_is_case_insensitive(self):
        self.assertTrue(self._call(_kv_cfg(connector="mooncakehybridconnector"), "kimi_k3"))

    def test_missing_hf_config_is_disabled(self):
        stub = _GateStub(_kv_cfg(), "kimi_k3")
        stub.model_config = mock.Mock(spec=[])
        self.assertFalse(type(self).gate(stub))


def _readout(swap_bytes, offset, length):
    """Read a swap-memory span without touching it from the host.

    Direct host access to swap memory (``.to("cpu")``, ``tensor[i].item()``,
    ``cpu_tensor.copy_(swap)``) faults, so verification must bounce through an
    ordinary device tensor first.
    """
    out = torch.empty(length, dtype=torch.int8, device=swap_bytes.device)
    out.copy_(swap_bytes.narrow(0, offset, length))
    torch.npu.synchronize()
    return out.to("cpu")


@unittest.skipUnless(_npu_swap_available(), "requires an Ascend NPU with swap-memory support")
class TestSwapStagingOnNPU(unittest.TestCase):
    """Checks that can only be judged against real swap memory."""

    def setUp(self):
        clear_swapped_tensors_for_testing()
        self.addCleanup(clear_swapped_tensors_for_testing)
        torch.npu.set_device(0)
        self.device = torch.device("npu:0")

    def _alloc_aligned(self, numel):
        """Mirrors NPUModelRunner._allocate_kimi_k3_swap_tensor."""
        from vllm_ascend.kv_offload.mooncake_swap_memory import empty_swapped_memory

        storage = empty_swapped_memory((numel + ALIGN,), dtype=torch.int8)
        offset = (-storage.data_ptr()) % ALIGN
        view = storage[offset : offset + numel]
        self.assertEqual(view.data_ptr() % ALIGN, 0)
        register_swapped_tensor(view)
        # Keep the backing allocation alive for the duration of the test.
        self._storage = storage
        return view

    def test_allocation_is_aligned_and_registered(self):
        view = self._alloc_aligned(4 * ALIGN)
        self.assertEqual(view.device.type, "npu")
        self.assertEqual(view.numel(), 4 * ALIGN)
        self.assertTrue(is_swapped_range(view.data_ptr(), 4 * ALIGN))
        self.assertTrue(is_swapped_range(view.data_ptr() + ALIGN, 4096))
        self.assertFalse(is_swapped_range(view.data_ptr() + 4 * ALIGN, 1))

    def test_fresh_allocation_reads_as_zero(self):
        """The replaced torch.zeros path guaranteed zeroed cache memory.

        Observed on torch_npu 2.10.0 / Ascend 910: a fresh allocation already
        reads back as zero.  This is not promised by the op documentation, so
        the check is here to catch a regression rather than to rely on it.
        """
        view = self._alloc_aligned(2 * ALIGN)
        self.assertEqual(int(_readout(view, 0, 64 * 1024).abs().sum()), 0)

    def test_zero_and_fill_are_supported(self):
        view = self._alloc_aligned(2 * ALIGN)
        view.fill_(5)
        torch.npu.synchronize()
        self.assertTrue(torch.equal(_readout(view, 0, 4096), torch.full((4096,), 5, dtype=torch.int8)))
        view.zero_()
        torch.npu.synchronize()
        self.assertEqual(int(_readout(view, 0, 4096).abs().sum()), 0)

    def test_copy_staging_to_swapped_writes_exact_bytes(self):
        view = self._alloc_aligned(4 * ALIGN)
        view.zero_()
        torch.npu.synchronize()

        length = 64 * 1024
        pattern = torch.arange(length, dtype=torch.int32).remainder(251).sub(125).to(torch.int8)
        staging = pattern.to(self.device)
        # A deliberately unaligned destination offset inside the allocation.
        dst_offset = ALIGN + 4096

        stub = object.__new__(KVCacheRecvingThread)
        KVCacheRecvingThread._copy_staging_to_swapped(stub, view.data_ptr() + dst_offset, length, staging, 0)

        got = _readout(view, dst_offset, length)
        self.assertTrue(torch.equal(got, pattern))

    def test_copy_respects_staging_offset_and_leaves_neighbours_intact(self):
        view = self._alloc_aligned(4 * ALIGN)
        view.zero_()
        torch.npu.synchronize()

        length = 32 * 1024
        staging_offset = 8192
        staging = torch.zeros(staging_offset + length, dtype=torch.int8, device=self.device)
        pattern = torch.full((length,), 42, dtype=torch.int8)
        staging.narrow(0, staging_offset, length).copy_(pattern.to(self.device))
        torch.npu.synchronize()

        dst_offset = 2 * ALIGN + 1024
        stub = object.__new__(KVCacheRecvingThread)
        KVCacheRecvingThread._copy_staging_to_swapped(
            stub, view.data_ptr() + dst_offset, length, staging, staging_offset
        )

        self.assertTrue(torch.equal(_readout(view, dst_offset, length), pattern))
        # Bytes on both sides of the written span must be untouched.
        self.assertEqual(int(_readout(view, dst_offset - 4096, 4096).abs().sum()), 0)
        self.assertEqual(int(_readout(view, dst_offset + length, 4096).abs().sum()), 0)

    def test_copy_to_unregistered_destination_raises(self):
        self._alloc_aligned(2 * ALIGN)
        staging = torch.zeros(4096, dtype=torch.int8, device=self.device)
        stub = object.__new__(KVCacheRecvingThread)

        with self.assertRaises(RuntimeError):
            KVCacheRecvingThread._copy_staging_to_swapped(stub, 0xDEAD0000, 4096, staging, 0)


if __name__ == "__main__":
    unittest.main()
