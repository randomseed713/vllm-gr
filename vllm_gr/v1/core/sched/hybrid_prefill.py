# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Run native Mamba checkpoint chunks before the grouped Beam Prefill."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from vllm_gr.v1.core.sched.beam_session import advance_gr_progress

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

    from vllm_gr.v1.core.sched.beam_session import GRSessionState

HYBRID_PREFILL_ATTR = "gr_hybrid_prefill_chunk"


@contextmanager
def _defer_prefix_publication(coordinator: Any) -> Iterator[None]:
    """Let every member look up completed prefixes before publishing new ones."""
    cache_blocks = coordinator.cache_blocks
    pending: list[tuple[Request, int]] = []

    def defer(request: Request, num_computed_tokens: int) -> None:
        pending.append((request, num_computed_tokens))

    coordinator.cache_blocks = defer
    try:
        yield
    finally:
        coordinator.cache_blocks = cache_blocks
    for request, num_computed_tokens in pending:
        cache_blocks(request, num_computed_tokens)


@contextmanager
def _snapshot_prefix_hits(scheduler: Any, session: GRSessionState) -> Iterator[None]:
    """Keep committed prefix sources visible while running members copy their tails."""
    if not getattr(scheduler, "mamba_partial_cache_hit", False) or not scheduler.running:
        yield
        return

    manager = scheduler.kv_cache_manager
    get_computed_blocks = manager.get_computed_blocks
    hits: dict[str, tuple[KVCacheBlocks, int, int]] = {}
    for request_id in session.request_ids:
        request = scheduler.requests.get(request_id)
        if request is not None and request.num_computed_tokens == 0:
            hits[request_id] = get_computed_blocks(request)
    pinned = [block for blocks, _, _ in hits.values() for group in blocks.blocks for block in group]

    def get_prefix(request: Request) -> tuple[KVCacheBlocks, int, int]:
        if request.request_id in hits:
            return hits[request.request_id]
        return get_computed_blocks(request)

    # A running producer can move a partial Mamba hash to a block that the
    # Worker has yet to copy. Waiting members can instead use the original
    # state as their own copy source. The Worker runs both copies before forward.
    manager.block_pool.touch(pinned)
    manager.get_computed_blocks = get_prefix
    try:
        yield
    finally:
        manager.get_computed_blocks = get_computed_blocks
        manager.block_pool.free_blocks(reversed(pinned))


@contextmanager
def hybrid_prefill_scope(scheduler: Any, session: GRSessionState) -> Iterator[str | None]:
    """Admit one checkpoint producer until all requests can finish Prefill together."""
    if (
        not getattr(scheduler, "need_mamba_block_aligned_split", False)
        or not session.hybrid_beam_block_counts
        or session.issued_stages
    ):
        yield None
        return
    request_id = next(
        (r for r in session.request_ids if r not in session._hybrid_prefill_ready), None
    )
    if request_id is None:
        if session.batch_size > 1:
            # An internal checkpoint can publish a prefix while allocating the
            # first member. The native Mamba allocator rejects a same-step hit,
            # which would split the remaining members out of the GR stage.
            with (
                _defer_prefix_publication(scheduler.kv_cache_manager.coordinator),
                _snapshot_prefix_hits(scheduler, session),
            ):
                yield None
        else:
            yield None
        return

    original_split = scheduler._mamba_block_aligned_split
    original_running = scheduler.running
    paused = [request for request in original_running if request.request_id != request_id]
    scheduler.running = [
        request for request in original_running if request.request_id == request_id
    ]

    def split(
        request: Any,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        allowed = original_split(
            request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens
        )
        end = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
            + allowed
        )
        if end < request.num_prompt_tokens:
            return allowed
        # Park the final chunk until all batch members can emit their logits.
        # A waiting request has not yet claimed the cached blocks looked up by
        # native scheduling. Pin that exact prefix while other members warm up.
        if num_new_local_computed_tokens:
            manager = scheduler.kv_cache_manager
            groups, cached, _ = manager.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
            if cached != num_new_local_computed_tokens:
                raise RuntimeError("Hybrid Beam prefix changed during admission")
            manager.block_pool.touch([block for group in groups for block in group])
            session._hybrid_prefill_pinned.extend(tuple(reversed(group)) for group in groups)
        session._hybrid_prefill_ready.add(request_id)
        advance_gr_progress(scheduler)
        return 0

    scheduler._mamba_block_aligned_split = split
    try:
        yield request_id
    finally:
        original_running[:] = [*paused, *scheduler.running]
        scheduler.running = original_running
        scheduler._mamba_block_aligned_split = original_split


def attach_hybrid_prefill_chunk(
    scheduler: Any, output: Any, session: GRSessionState, request_id: str
) -> None:
    if not output.num_scheduled_tokens:
        return
    if (
        tuple(output.num_scheduled_tokens) != (request_id,)
        or not scheduler.requests[request_id].is_prefill_chunk
    ):
        raise RuntimeError("Hybrid Beam checkpoint dispatch must contain only a partial Prefill")
    if session.hybrid_prefill_dispatch is not None:
        raise RuntimeError("Hybrid Beam checkpoint already has an outstanding dispatch")
    manager = scheduler.kv_cache_manager
    groups = manager.get_blocks(request_id).blocks
    manager.block_pool.touch([block for group in groups for block in group])
    identity = (session.session_id, session.owner_generation, advance_gr_progress(scheduler))
    session._kv_dispatch_leases[identity] = tuple(tuple(reversed(group)) for group in groups)
    session._hybrid_prefill_dispatch = identity
    setattr(output, HYBRID_PREFILL_ATTR, identity)


def release_hybrid_prefill_pins(scheduler: Any, session: GRSessionState) -> None:
    if session._hybrid_prefill_pinned:
        pool = scheduler.kv_cache_manager.block_pool
        for blocks in session._hybrid_prefill_pinned:
            pool.free_blocks(blocks)
        session._hybrid_prefill_pinned.clear()
