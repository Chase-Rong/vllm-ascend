import threading


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()
        self.registered_buffers: set[tuple[int, int, str | None]] = set()

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
        return self.transfer_engine

    def register_buffer(
        self,
        ptrs: list[int],
        sizes: list[int],
        locations: list[str | None] | None = None,
    ):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if len(ptrs) != len(sizes):
                raise ValueError("Mooncake register pointer/size counts differ.")

            if locations is None:
                # Registration is incremental: the receive path can allocate
                # an NPU staging buffer after the KV cache buffers have been
                # registered.  Keep the legacy call shape while consulting
                # the per-buffer registry below instead of short-circuiting
                # on the historical boolean flag.
                locations = [None] * len(ptrs)
            elif len(locations) != len(ptrs):
                raise ValueError("Mooncake register locations must match ptr count.")

            register_with_location = getattr(self.transfer_engine, "register_memory_with_location", None)
            for ptr, size, location in zip(ptrs, sizes, locations):
                key = (int(ptr), int(size), location)
                if key in self.registered_buffers:
                    continue
                if location is not None:
                    if register_with_location is None:
                        raise RuntimeError(
                            "Mooncake TransferEngine does not support location-aware memory registration."
                        )
                    ret_value = register_with_location(ptr, size, location)
                else:
                    ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError(
                        f"Mooncake memory registration failed. ptr={ptr} size={size} location={location}"
                    )
                self.registered_buffers.add(key)
            self.is_register_buffer = True


global_te = GlobalTE()
