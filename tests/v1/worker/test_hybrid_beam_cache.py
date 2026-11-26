# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Coverage for native Hybrid Beam admission and cache reparenting."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_gr.config.execution import validate_gr_execution_runtime
from vllm_gr.ops.cuda.beam_kv_reorder import CacheBlockCopy
from vllm_gr.v1.beam.hybrid_cache import HybridBlockBudget, hybrid_block_budget
from vllm_gr.v1.worker.cuda.hybrid_beam_cache import (
    CacheBlockPlan,
    _attention_decode_blocks,
    _recurrent_decode_blocks,
)
from vllm_gr.v1.worker.cuda.hybrid_beam_graph import GroupedDecodeGraphInputs

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_gdn_decode_graph_keeps_query_offsets_across_native_padding() -> None:
    device = torch.device("cpu")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE, max_cudagraph_capture_size=16
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        model_config=SimpleNamespace(max_model_len=64, hf_text_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
        additional_config={},
    )
    spec = MambaSpec(
        block_size=16,
        shapes=((128,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    builder = GDNAttentionMetadataBuilder(spec, ["kda"], cast(VllmConfig, config), device)
    worker = SimpleNamespace(
        device=device,
        model_config=config.model_config,
        kernel_block_sizes=[16],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[KVCacheGroupSpec(["kda"], spec)]),
        attn_groups=[
            [
                SimpleNamespace(
                    kv_cache_spec=spec, get_metadata_builder=lambda: builder, layer_names=["kda"]
                )
            ]
        ],
    )
    captured = GroupedDecodeGraphInputs(worker, 8)
    offsets = captured.attention["kda"].non_spec_query_start_loc
    assert offsets.tolist() == list(range(9))

    padded_offsets = torch.tensor([0, 1] + [2] * 15, dtype=torch.int32)
    native = builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=padded_offsets,
            query_start_loc_cpu=padded_offsets,
            seq_lens=torch.ones(16, dtype=torch.int32),
            num_reqs=16,
            num_actual_tokens=16,
            max_query_len=1,
            max_seq_len=1,
            block_table_tensor=torch.zeros((16, 1), dtype=torch.int32),
            slot_mapping=torch.full((16,), -1, dtype=torch.int64),
        ),
    )
    assert native.non_spec_query_start_loc is not None
    assert torch.equal(native.non_spec_query_start_loc, padded_offsets)
    assert offsets.tolist() == list(range(9))


def test_hybrid_runtime_requires_native_cuda_cache_modes() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=True,
            architectures=["UnlistedHybridForCausalLM"],
            enforce_eager=True,
            dtype=torch.bfloat16,
        ),
        device_config=SimpleNamespace(device="cuda"),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False, mamba_cache_mode="none", cache_dtype="auto"
        ),
        use_v2_model_runner=False,
    )
    validate_gr_execution_runtime(config)
    config.use_v2_model_runner = True
    validate_gr_execution_runtime(config)
    config.device_config.device = "npu"
    with pytest.raises(ValueError, match="Hybrid Beam requires CUDA"):
        validate_gr_execution_runtime(config)
    config.device_config.device = "cuda"
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="mamba_cache_mode=align"):
        validate_gr_execution_runtime(config)
    config.cache_config.mamba_cache_mode = "align"
    config.scheduler_config.enable_chunked_prefill = True
    validate_gr_execution_runtime(config)


def test_hybrid_budget_includes_the_writable_tail_of_each_beam() -> None:
    groups = [
        KVCacheGroupSpec(
            ["mla"],
            MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.bfloat16),
        ),
        KVCacheGroupSpec(
            ["kda"],
            MambaSpec(
                block_size=32,
                shapes=((3, 4), (1, 2, 2)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            ),
        ),
    ]
    assert hybrid_block_budget(
        groups, prompt_tokens=7, decode_steps=2, beam_width=2
    ) == HybridBlockBudget(native_blocks=4, beam_blocks_by_group=(4, 2), decode_reserved_blocks=0)
    assert hybrid_block_budget(
        groups, prompt_tokens=8, decode_steps=2, beam_width=2
    ) == HybridBlockBudget(native_blocks=4, beam_blocks_by_group=(2, 2), decode_reserved_blocks=0)
    assert hybrid_block_budget(
        groups, prompt_tokens=7, decode_steps=0, beam_width=2
    ) == HybridBlockBudget(native_blocks=3, beam_blocks_by_group=(0, 0), decode_reserved_blocks=0)


@pytest.mark.parametrize(  # type: ignore[misc]
    "mamba_type",
    [
        MambaAttentionBackendEnum.MAMBA1,
        MambaAttentionBackendEnum.MAMBA2,
        MambaAttentionBackendEnum.SHORT_CONV,
        MambaAttentionBackendEnum.LINEAR,
        MambaAttentionBackendEnum.GDN_ATTN,
    ],
)
def test_hybrid_budget_accepts_native_recurrent_groups_with_paged_attention(
    mamba_type: MambaAttentionBackendEnum,
) -> None:
    spec = MambaSpec(
        block_size=4,
        shapes=((3, 4),),
        dtypes=(torch.bfloat16,),
        mamba_type=mamba_type,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=8, dtype=torch.bfloat16),
        ),
        KVCacheGroupSpec(
            ["recurrent"],
            UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"recurrent": spec}),
        ),
    ]
    assert hybrid_block_budget(
        groups, prompt_tokens=7, decode_steps=2, beam_width=2
    ) == HybridBlockBudget(native_blocks=6, beam_blocks_by_group=(4, 2), decode_reserved_blocks=1)
    cache = torch.arange(40).reshape(10, 4)
    private = torch.tensor([[6], [7]], dtype=torch.int32)
    plan = CacheBlockPlan()
    table = _recurrent_decode_blocks(
        plan,
        torch.tensor([0, 3], dtype=torch.int32),
        private,
        torch.zeros(2, dtype=torch.long),
        prompt_length=7,
        step=0,
        spec=spec,
    )
    plan.apply(CacheBlockCopy((cache,)))
    assert table.tolist() == [[6, 6], [7, 7]]
    torch.testing.assert_close(cache[6:8], cache[3].expand(2, -1))


def test_align_budget_keeps_decode_blocks_and_checkpoint_headroom() -> None:
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=16,
                shapes=((1024,),),
                dtypes=(torch.float16,),
                mamba_cache_mode="align",
                num_prefill_checkpoint_blocks=1,
                prefill_checkpoint_alignment=8,
            ),
        ),
    ]
    result = hybrid_block_budget(
        groups, prompt_tokens=31, decode_steps=2, beam_width=2, hash_block_size=8
    )
    assert result == HybridBlockBudget(
        native_blocks=10, beam_blocks_by_group=(4, 2), decode_reserved_blocks=1
    )
    assert hybrid_block_budget(
        groups, prompt_tokens=8, decode_steps=2, beam_width=2, hash_block_size=8
    ) == HybridBlockBudget(native_blocks=4, beam_blocks_by_group=(2, 2), decode_reserved_blocks=2)


def test_attention_decode_keeps_the_shared_page_and_forks_the_tail_across_boundary() -> None:
    cache = torch.arange(40).view(10, 4, 1)
    private = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    prompt_cache = cache[1:3].clone()
    parents = torch.tensor([0, 0])
    copier = CacheBlockCopy((cache,))
    plan = CacheBlockPlan()
    table, slots = _attention_decode_blocks(
        plan,
        torch.tensor([1, 2], dtype=torch.int32),
        private,
        parents,
        block_size=4,
        prompt_length=7,
        step=0,
    )
    plan.apply(copier)
    assert table.tolist() == [[1, 3, 4], [1, 5, 6]]
    assert slots.tolist() == [15, 23]
    torch.testing.assert_close(cache[3], prompt_cache[1])
    torch.testing.assert_close(cache[5], prompt_cache[1])
    cache[3, 3] = 80
    cache[5, 3] = 90

    plan = CacheBlockPlan()
    _, slots = _attention_decode_blocks(
        plan,
        torch.tensor([1, 2], dtype=torch.int32),
        private,
        torch.tensor([1, 1]),
        block_size=4,
        prompt_length=7,
        step=1,
    )
    plan.apply(copier)
    assert slots.tolist() == [16, 24]
    assert cache[3, 3].item() == cache[5, 3].item() == 90
    assert not cache[[4, 6]].any()
    torch.testing.assert_close(cache[1:3], prompt_cache)
