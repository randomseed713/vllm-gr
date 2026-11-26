# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import pytest
import torch

from vllm_gr.ops.cuda.beam_kv_reorder import CacheBlockCopy

pytestmark = [pytest.mark.l2, pytest.mark.gpu_test]


def test_grouped_reparent_only_writes_owned_elements() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Grouped Beam cache copy requires CUDA")

    # Models can share native storage across groups, with padding between blocks.
    shape = (8, 2576)
    elements_per_block = 2560
    bf16 = torch.arange(8 * 2576, dtype=torch.int32, device="cuda").view(shape).to(torch.bfloat16)
    layer = bf16[:, :elements_per_block].reshape(8, 32, 80).transpose(1, 2)
    next_layer = (bf16 + 16)[:, :elements_per_block].reshape(8, 32, 80).transpose(1, 2)
    recurrent = torch.randn(8, 4, 17, 17, device="cuda")
    strided = recurrent[..., ::2]
    # Test copies between physical destinations with repeated and swapped
    # parents. Slicing the source also exercises noncontiguous index input.
    sources = torch.tensor([2, 0, 1, 0, 1, 0, 4, 0], device="cuda")[::2]
    targets = torch.tensor([1, 2, 3, 5], device="cuda")
    tensors = (layer, next_layer, recurrent[..., 1::2], strided)
    original = bf16.clone()
    expected = [tensor.clone() for tensor in tensors]
    for reference in expected:
        reference[targets] = reference[sources]

    CacheBlockCopy(tensors).copy(sources, targets)

    for result, reference in zip(tensors, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        bf16[:, elements_per_block:], original[:, elements_per_block:], rtol=0, atol=0
    )
