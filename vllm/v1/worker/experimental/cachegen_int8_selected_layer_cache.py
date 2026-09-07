# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Selected-layer compact INT8 cache with native-block mapping."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.worker.experimental.cachegen_int8_block_page_map import (
    CacheGenInt8BlockPageMap,
)
from vllm.v1.worker.experimental.cachegen_int8_fixed_scale_page_store import (
    CacheGenInt8FixedScalePageStore,
)


@dataclass
class CacheGenInt8SelectedLayerCache:
    """One selected layer's compact cache and native block-ID mapping."""

    page_map: CacheGenInt8BlockPageMap
    page_store: CacheGenInt8FixedScalePageStore

    @classmethod
    def allocate(
        cls,
        *,
        max_pages: int,
        tokens_per_page: int,
        num_kv_heads: int,
        head_size: int,
        key_scales: torch.Tensor,
        value_scales: torch.Tensor,
        device: torch.device,
    ) -> "CacheGenInt8SelectedLayerCache":
        return cls(
            page_map=CacheGenInt8BlockPageMap(max_pages=max_pages),
            page_store=CacheGenInt8FixedScalePageStore.allocate(
                num_pages=max_pages,
                tokens_per_page=tokens_per_page,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                key_scales=key_scales,
                value_scales=value_scales,
                device=device,
            ),
        )

    def map_active_block_table(
        self,
        native_block_table: torch.Tensor,
    ) -> torch.Tensor | None:
        """Map a rank-1 native active block table into compact page IDs."""
        if native_block_table.ndim != 1:
            raise ValueError(
                "native_block_table must be rank 1; got "
                f"{tuple(native_block_table.shape)}"
            )
        if native_block_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("native_block_table must have integer dtype")
        if native_block_table.device != self.page_store.device:
            raise ValueError(
                f"native_block_table must be on {self.page_store.device}; got "
                f"{native_block_table.device}"
            )

        compact_ids = self.page_map.map_block_ids(
            native_block_table.tolist()
        )
        if compact_ids is None:
            return None
        return torch.tensor(
            compact_ids,
            dtype=native_block_table.dtype,
            device=native_block_table.device,
        )

    def write_current_tokens(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        native_page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
    ) -> bool:
        """Write current K/V tokens; return False if page-map capacity is full."""
        compact_page_ids = self.map_active_block_table(native_page_ids)
        if compact_page_ids is None:
            return False

        self.page_store.write_mapped_tokens(
            keys=keys,
            values=values,
            page_ids=compact_page_ids,
            page_offsets=page_offsets,
        )
        return True
