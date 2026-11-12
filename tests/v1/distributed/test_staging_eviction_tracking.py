# SPDX-License-Identifier: Apache-2.0
"""Resident removal must keep remaining staging allocations evictable."""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import TTLLock
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus

pytestmark = pytest.mark.skipif(
    not hasattr(TTLLock, "lock"), reason="Requires native TTL locks"
)


@pytest.mark.parametrize("remaining_staging", [False, True])
def test_clear_preserves_staging_eviction_tracking(remaining_staging: bool) -> None:
    """Withdraw residency immediately, but track the key until staging is freed."""
    bus = init_event_bus(EventBusConfig(enabled=True))
    events: list[Event] = []
    bus.subscribe(EventType.L1_KEYS_EVICTED, events.append)
    manager = DramL1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=65536,
                use_lazy=False,
                init_size_in_bytes=65536,
                align_bytes=64,
                shm_name="",
            ),
            write_ttl_seconds=1,
        )
    )
    policy = LRUEvictionPolicy()
    manager.register_listener(L1EvictionPolicy(policy))
    key = ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(1),
        model_name="staging-eviction-tracking",
        kv_rank=0,
    )
    layout = MemoryLayoutDesc(shapes=[torch.Size([32])], dtypes=[torch.uint8])
    try:
        result = manager.reserve_write([key], [False], layout, tag="resident")
        assert result[key][0] == L1Error.SUCCESS
        if remaining_staging:
            result = manager.reserve_write([key], [False], layout, tag="pending")
            assert result[key][0] == L1Error.SUCCESS

        assert manager.finish_write([key], tag="resident")[key] == L1Error.SUCCESS
        manager.clear(force=False)

        assert manager.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST
        bus.stop()
        assert len(events) == 1
        assert events[0].metadata["keys"] == [key]
        assert events[0].metadata["meta"][0].size_bytes == 32

        if remaining_staging:
            assert manager.get_staging_memory_usage() == 32
            assert not manager.is_key_evictable(key)
            assert not policy.get_eviction_actions(
                1.0, key_eligible_filter=manager.is_key_evictable
            )
            time.sleep(1.2)
            assert manager.is_key_evictable(key)
            actions = policy.get_eviction_actions(
                1.0, key_eligible_filter=manager.is_key_evictable
            )
            assert [victim for action in actions for victim in action.keys] == [key]
            assert manager.delete([key])[key] == L1Error.SUCCESS

        assert manager.get_staging_memory_usage() == 0
        assert manager.get_memory_usage()[0] == 0
        assert not policy.get_eviction_actions(1.0)
    finally:
        manager.close()
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))
