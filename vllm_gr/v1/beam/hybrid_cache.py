# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Native KV block budgets for Beam sessions with recurrent cache groups."""

from collections.abc import Sequence
from typing import NamedTuple

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)


class HybridBlockBudget(NamedTuple):
    native_blocks: int
    beam_blocks_by_group: tuple[int, ...]
    decode_reserved_blocks: int


def hybrid_group_spec(group: KVCacheGroupSpec) -> MambaSpec | AttentionSpec:
    """Resolve the block-table geometry used by a native cache group."""
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        spec = spec.first_spec
    if not group.host_resident and spec.has_layer_views:
        if (
            isinstance(spec, MambaSpec)
            and spec.num_speculative_blocks == 0
            and spec.mamba_cache_mode in ("none", "align")
        ):
            return spec
        if (
            isinstance(spec, AttentionSpec)
            and not isinstance(
                spec, (CircularBufferSpec, CrossAttentionSpec, EncoderOnlyAttentionSpec)
            )
            and spec.tokens_per_state == 1
        ):
            return spec
    raise ValueError(f"Hybrid Beam cannot fork the native {type(spec).__name__} cache group")


def hybrid_block_budget(
    groups: Sequence[KVCacheGroupSpec],
    *,
    prompt_tokens: int,
    decode_steps: int,
    beam_width: int,
    hash_block_size: int | None = None,
) -> HybridBlockBudget:
    """Budget native capacity, private Beam blocks and the Decode reserve.

    Each beam owns a state per recurrent group and the attention pages
    starting at the first writable position. Earlier pages are shared read-only.
    """
    native_blocks = 0
    decode_reserve = 0
    branch_counts = []
    has_recurrent_state = False
    extent = prompt_tokens + decode_steps
    for group in groups:
        spec = hybrid_group_spec(group)
        native_count = cdiv(extent, spec.block_size)
        partial_hit = False
        decode_tail_copy = False
        if hash_block_size is not None and hash_block_size < spec.block_size:
            partial_hit = prompt_tokens > hash_block_size
            decode_tail_copy = (
                decode_steps > 0
                and prompt_tokens % hash_block_size == 0
                and prompt_tokens % spec.block_size != 0
            )
        if isinstance(spec, MambaSpec):
            has_recurrent_state = True
            if spec.mamba_cache_mode == "none":
                native_count = 1
            else:
                # Native align ignores lookahead. Keep future logical state
                # blocks free while allocating all members' Prefill slots.
                decode_reserve += native_count - cdiv(prompt_tokens, spec.block_size)
                if prompt_tokens > spec.block_size:
                    native_count += spec.num_prefill_checkpoint_blocks
                if partial_hit or decode_tail_copy:
                    # A fine-grained prefix can hold a source and a copied
                    # tail in one column. Native admission can also require
                    # an extra free block before it knows the copy fits there.
                    native_count += 2
                if decode_tail_copy:
                    # A prompt ending on a sub-block hash boundary can
                    # publish a tail that the first Decode must copy.
                    decode_reserve += 2
            writable_pages = 1
        else:
            writable_pages = native_count - prompt_tokens // spec.block_size
            if partial_hit:
                # The shared source can remain pinned beside the writable copy.
                native_count += 1
        native_blocks += native_count
        branch_counts.append(beam_width * writable_pages if decode_steps else 0)
    if not has_recurrent_state:
        raise ValueError("Hybrid Beam requires a native recurrent cache group")
    return HybridBlockBudget(native_blocks, tuple(branch_counts), decode_reserve)
