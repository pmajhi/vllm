# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dynamic-scale selected-layer INT8 pages with common byte-page mirroring."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.worker.experimental.cachegen_int8_adaptive_page_codec import (
    CacheGenInt8AdaptivePageCodec,
    CacheGenInt8AdaptivePageLayout,
)
from vllm.v1.worker.experimental.cachegen_int8_block_page_map import (
    CacheGenInt8BlockPageMap,
)
from vllm.v1.worker.experimental.cachegen_int8_inference_page_store import (
    CacheGenInt8InferencePageStore,
)
from vllm.v1.worker.experimental.cachegen_kv_page_format_registry import (
    CacheGenKVPageFormatRegistry,
)
from vllm.v1.worker.experimental.cachegen_kv_page_geometry import (
    CacheGenKVPageGeometry,
)
from vllm.v1.worker.experimental.cachegen_kv_page_pool import (
    CacheGenKVPageHandle,
    CacheGenKVPagePool,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


@dataclass
class CacheGenInt8DynamicSelectedLayerCache:
    """One selected layer's typed INT8 store and common byte-page mirror."""

    page_map: CacheGenInt8BlockPageMap
    page_store: CacheGenInt8InferencePageStore
    common_page_pool: CacheGenKVPagePool
    common_page_codec: CacheGenInt8AdaptivePageCodec
    common_page_handles: dict[int, CacheGenKVPageHandle]

    @classmethod
    def allocate(
        cls,
        *,
        max_pages: int,
        tokens_per_page: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
    ) -> "CacheGenInt8DynamicSelectedLayerCache":
        geometry = CacheGenKVPageGeometry(
            tokens_per_page=tokens_per_page,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        format_registry = CacheGenKVPageFormatRegistry.from_geometry(geometry)
        common_page_pool = CacheGenKVPagePool(
            num_pages=max_pages,
            page_nbytes=geometry.common_page_nbytes,
            device=device,
            page_formats=format_registry,
        )
        common_page_codec = CacheGenInt8AdaptivePageCodec(
            page_format=format_registry.get(
                CacheGenKVQuantizer.INT8_ADAPTIVE
            ),
            layout=CacheGenInt8AdaptivePageLayout(
                tokens_per_page=tokens_per_page,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
            ),
        )
        return cls(
            page_map=CacheGenInt8BlockPageMap(max_pages=max_pages),
            page_store=CacheGenInt8InferencePageStore.allocate(
                num_pages=max_pages,
                tokens_per_page=tokens_per_page,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                device=device,
            ),
            common_page_pool=common_page_pool,
            common_page_codec=common_page_codec,
            common_page_handles={},
        )

    def reset(self) -> None:
        """Clear typed pages, common byte pages, and compact mappings."""
        self.page_map.reset()
        self.page_store.reset()
        self.common_page_pool.reset()
        self.common_page_handles.clear()

    def map_active_block_table(
        self,
        native_block_table: torch.Tensor,
    ) -> torch.Tensor | None:
        """Map one active native block-table prefix into compact page IDs."""
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

    def _common_page_handle(
        self,
        compact_page_id: int,
    ) -> CacheGenKVPageHandle:
        """Return or allocate the common fixed-byte mirror page."""
        handle = self.common_page_handles.get(compact_page_id)
        if handle is not None:
            return handle

        handle = self.common_page_pool.allocate(
            CacheGenKVQuantizer.INT8_ADAPTIVE
        )
        if handle is None:
            raise RuntimeError(
                "Common fixed-byte INT8 page pool exhausted despite compact "
                "page-map capacity"
            )
        self.common_page_handles[compact_page_id] = handle
        return handle

    def _mirror_active_compact_pages(
        self,
        *,
        compact_page_ids: torch.Tensor,
    ) -> None:
        """Encode every touched compact page, including a partial final page."""
        for compact_page_id in torch.unique(compact_page_ids).tolist():
            compact_page_id = int(compact_page_id)
            valid_tokens = int(
                self.page_store.valid_tokens[compact_page_id].sum().item()
            )
            if valid_tokens == 0:
                continue

            handle = self._common_page_handle(compact_page_id)
            keys, values = self.page_store.read_page(
                page_id=compact_page_id,
                dtype=torch.float32,
            )
            self.common_page_codec.write_page(
                pool=self.common_page_pool,
                handle=handle,
                keys=keys[:valid_tokens],
                values=values[:valid_tokens],
                valid_tokens=valid_tokens,
            )

    def has_common_page_ids(
        self,
        *,
        compact_page_ids: torch.Tensor,
    ) -> bool:
        """Return whether every compact page has a common-slab mirror."""
        if compact_page_ids.ndim != 1:
            raise ValueError("compact_page_ids must be one-dimensional")
        if compact_page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("compact_page_ids must have integer dtype")
        if compact_page_ids.device != self.page_store.device:
            raise ValueError(
                f"compact_page_ids must be on {self.page_store.device}; got "
                f"{compact_page_ids.device}"
            )
        return all(
            int(compact_page_id) in self.common_page_handles
            for compact_page_id in compact_page_ids.tolist()
        )

    def get_common_page_ids(
        self,
        *,
        compact_page_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return common-pool page IDs aligned with compact page IDs.

        Every requested compact page must already have been mirrored. The
        returned tensor is safe to pass to a fused common-byte-slab kernel.
        """
        if compact_page_ids.ndim != 1:
            raise ValueError("compact_page_ids must be one-dimensional")
        if compact_page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("compact_page_ids must have integer dtype")
        if compact_page_ids.device != self.page_store.device:
            raise ValueError(
                f"compact_page_ids must be on {self.page_store.device}; got "
                f"{compact_page_ids.device}"
            )

        common_page_ids: list[int] = []
        for compact_page_id in compact_page_ids.tolist():
            handle = self.common_page_handles.get(int(compact_page_id))
            if handle is None:
                raise ValueError(
                    "Compact page has no common-slab mirror: "
                    f"compact_page_id={compact_page_id}"
                )
            common_page_ids.append(handle.page_id)

        return torch.tensor(
            common_page_ids,
            dtype=compact_page_ids.dtype,
            device=compact_page_ids.device,
        )

    def write_native_mapped_tokens(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        native_page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
    ) -> bool:
        """Map and write arbitrary prefill/decode tokens into INT8 pages.

        ``native_page_ids`` and ``page_offsets`` are one entry per K/V token
        and are derived from the real vLLM slot mapping. Returning ``False``
        means the compact map lacks capacity; callers must retain native
        attention and may skip common-slab validation safely.
        """
        if keys.ndim != 3:
            raise ValueError(
                f"keys must be rank 3, got {tuple(keys.shape)}"
            )
        if values.shape != keys.shape:
            raise ValueError(
                f"values must have shape {tuple(keys.shape)}, "
                f"got {tuple(values.shape)}"
            )
        if native_page_ids.ndim != 1:
            raise ValueError("native_page_ids must be one-dimensional")
        if page_offsets.shape != native_page_ids.shape:
            raise ValueError(
                "page_offsets must have same shape as native_page_ids"
            )
        if native_page_ids.numel() != keys.shape[0]:
            raise ValueError(
                "native_page_ids length must equal number of K/V tokens"
            )
        if native_page_ids.device != self.page_store.device:
            raise ValueError(
                f"native_page_ids must be on {self.page_store.device}; got "
                f"{native_page_ids.device}"
            )
        if page_offsets.device != self.page_store.device:
            raise ValueError(
                f"page_offsets must be on {self.page_store.device}; got "
                f"{page_offsets.device}"
            )

        unique_native_page_ids: list[int] = []
        seen_native_page_ids: set[int] = set()
        for native_page_id in native_page_ids.tolist():
            native_page_id = int(native_page_id)
            if native_page_id not in seen_native_page_ids:
                seen_native_page_ids.add(native_page_id)
                unique_native_page_ids.append(native_page_id)

        compact_ids = self.page_map.map_block_ids(unique_native_page_ids)
        if compact_ids is None:
            return False

        native_to_compact = {
            native_page_id: compact_page_id
            for native_page_id, compact_page_id in zip(
                unique_native_page_ids,
                compact_ids,
            )
        }
        compact_page_ids = torch.tensor(
            [
                native_to_compact[int(native_page_id)]
                for native_page_id in native_page_ids.tolist()
            ],
            dtype=native_page_ids.dtype,
            device=native_page_ids.device,
        )

        self.page_store.write_mapped_tokens(
            keys=keys,
            values=values,
            page_ids=compact_page_ids,
            page_offsets=page_offsets,
        )
        self._mirror_active_compact_pages(
            compact_page_ids=compact_page_ids,
        )
        return True

    def write_current_tokens(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        native_page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
    ) -> bool:
        """Backward-compatible alias for real native-page token writes."""
        return self.write_native_mapped_tokens(
            keys=keys,
            values=values,
            native_page_ids=native_page_ids,
            page_offsets=page_offsets,
        )
