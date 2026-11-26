# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""CUDA graph bindings for native single-token MLA and GDN Decode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_gr.v1.beam.hybrid_cache import hybrid_group_spec
from vllm_gr.v1.worker.cuda.hybrid_beam_cache import (
    HybridDecodeBatch,
    build_hybrid_attention,
    kernel_block_sizes,
)

if TYPE_CHECKING:
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import GPUBeamStageRunner


class GroupedDecodeGraphInputs:
    def __init__(self, worker: Any, rows: int) -> None:
        device = worker.device
        max_len = worker.model_config.max_model_len
        self.input_ids = torch.zeros(rows, dtype=torch.int32, device=device)
        self.positions = torch.zeros(rows, dtype=torch.int64, device=device)
        self.state_ids: dict[int, torch.Tensor] = {}
        tables = []
        for group_id, (group, kernel_size) in enumerate(
            zip(worker.kv_cache_config.kv_cache_groups, kernel_block_sizes(worker), strict=True)
        ):
            spec = hybrid_group_spec(group)
            recurrent = isinstance(spec, MambaSpec)
            columns = (
                1
                if recurrent
                else cdiv(max_len, spec.block_size) * (spec.block_size // kernel_size)
            )
            table = torch.zeros((rows, columns), dtype=torch.int32, device=device)
            tables.append(table)
            if recurrent:
                # The capture builds single-token Decode metadata at position zero.
                self.state_ids[group_id] = table[:, 0]
        self.batch = HybridDecodeBatch(
            seq_lens=torch.ones(rows, dtype=torch.int32, device=device),
            seq_lens_cpu=torch.ones(rows, dtype=torch.int32),
            tables=tuple(tables),
            slots=tuple(torch.full((rows,), -1, dtype=torch.int64, device=device) for _ in tables),
        )
        # Dummy state 0 and slot -1 disable GDN and MLA cache writes during
        # warmup and capture. A replay binds the admitted private block IDs.
        self.attention, self.slots = build_hybrid_attention(
            worker, self.batch, self.positions, max_seq_len=max_len, graph_state_ids=self.state_ids
        )

    def load(self, owner: GPUBeamStageRunner, batch: HybridDecodeBatch) -> None:
        self.input_ids.copy_(owner.input_ids)
        self.positions.copy_(owner.positions)
        self.batch.seq_lens.copy_(batch.seq_lens)
        for index, (target, source, slot, actual_slots) in enumerate(
            zip(self.batch.tables, batch.tables, self.batch.slots, batch.slots, strict=True)
        ):
            if index in self.state_ids:
                self.state_ids[index].copy_(source[:, 0])
                continue
            target.zero_()
            target[:, : source.shape[1]].copy_(source)
            slot.copy_(actual_slots)


def supports_grouped_decode_graph(worker: Any) -> bool:
    # The graph stages live lengths and GDN state indices in fixed device
    # buffers. Other native builders can require host-selected launch geometry
    # or different state metadata and use eager Decode.
    for group_id, group in enumerate(worker.kv_cache_config.kv_cache_groups):
        spec = hybrid_group_spec(group)
        if isinstance(spec, MambaSpec):
            if spec.mamba_type is not MambaAttentionBackendEnum.GDN_ATTN:
                return False
        elif any(
            attention.backend.get_name() != "TRITON_MLA"
            for attention in worker.attn_groups[group_id]
        ):
            return False
    return True
