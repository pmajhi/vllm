# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-size heterogeneous page allocation for experimental paged KV."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.worker.experimental.cachegen_kv_page_format import (
    CacheGenKVPageFormat,
)
from vllm.v1.worker.experimental.cachegen_kv_page_format_registry import (
    CacheGenKVPageFormatRegistry,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


@dataclass(frozen=True)
class CacheGenKVPageHandle:
    """A leased fixed-byte page and its immutable quantizer format."""

    page_id: int
    quantizer: CacheGenKVQuantizer
    format: CacheGenKVPageFormat


class CacheGenKVPagePool:
    """GPU-resident fixed-byte pages with explicit heterogeneous formats.

    The byte slab has shape ``[num_pages, page_nbytes]`` and lives on one
    device. Each allocated page receives one immutable format/quantizer ID
    until released. Quantizer-specific encoders write to their declared
    metadata/key/value regions, while fused decode kernels consume those
    regions directly from the slab.
    """

    def __init__(
        self,
        *,
        num_pages: int,
        page_nbytes: int,
        device: torch.device,
        page_formats: dict[CacheGenKVQuantizer, CacheGenKVPageFormat]
        | CacheGenKVPageFormatRegistry,
    ) -> None:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        if page_nbytes <= 0:
            raise ValueError("page_nbytes must be positive")
        if isinstance(page_formats, CacheGenKVPageFormatRegistry):
            page_formats = page_formats.formats()

        if not page_formats:
            raise ValueError("page_formats must not be empty")

        for quantizer, page_format in page_formats.items():
            if page_format.quantizer is not quantizer:
                raise ValueError(
                    "page format quantizer key does not match format value: "
                    f"key={quantizer.value}, "
                    f"format={page_format.quantizer.value}"
                )
            if page_format.page_nbytes != page_nbytes:
                raise ValueError(
                    "all page formats must use the pool fixed page size: "
                    f"expected={page_nbytes}, "
                    f"got={page_format.page_nbytes} for {quantizer.value}"
                )

        self.num_pages = num_pages
        self.page_nbytes = page_nbytes
        self.device = device
        self.page_formats = dict(page_formats)

        self.page_bytes = torch.zeros(
            (num_pages, page_nbytes),
            dtype=torch.uint8,
            device=device,
        )
        self.page_quantizer_ids = torch.full(
            (num_pages,),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        self.page_in_use = torch.zeros(
            (num_pages,),
            dtype=torch.bool,
            device=device,
        )
        self.page_valid_tokens = torch.zeros(
            (num_pages,),
            dtype=torch.int32,
            device=device,
        )
        self._free_page_ids = list(range(num_pages))
        self._handles: dict[int, CacheGenKVPageHandle] = {}

    def allocate(
        self,
        quantizer: CacheGenKVQuantizer,
    ) -> CacheGenKVPageHandle | None:
        """Lease one fixed-byte page for a supported page format."""
        page_format = self.page_formats.get(quantizer)
        if page_format is None:
            raise ValueError(
                "No registered fixed-byte page format for quantizer "
                f"{quantizer.value}"
            )
        if not page_format.implemented:
            raise ValueError(
                "Cannot allocate an unimplemented fixed-byte page format: "
                f"{quantizer.value}"
            )
        if not self._free_page_ids:
            return None

        page_id = self._free_page_ids.pop()
        handle = CacheGenKVPageHandle(
            page_id=page_id,
            quantizer=quantizer,
            format=page_format,
        )
        self._handles[page_id] = handle
        self.page_in_use[page_id] = True
        self.page_quantizer_ids[page_id] = list(
            CacheGenKVQuantizer
        ).index(quantizer)
        return handle

    def get(self, page_id: int) -> CacheGenKVPageHandle | None:
        """Return the active page handle, if the page is leased."""
        return self._handles.get(page_id)

    def release(self, page_id: int) -> CacheGenKVPageHandle | None:
        """Release one page and clear its HBM payload and metadata."""
        handle = self._handles.pop(page_id, None)
        if handle is None:
            return None

        self.page_bytes[page_id].zero_()
        self.page_quantizer_ids[page_id] = -1
        self.page_in_use[page_id] = False
        self.page_valid_tokens[page_id] = 0
        self._free_page_ids.append(page_id)
        return handle

    def reset(self) -> None:
        """Release all pages and clear the fixed-byte HBM slab."""
        self.page_bytes.zero_()
        self.page_quantizer_ids.fill_(-1)
        self.page_in_use.zero_()
        self.page_valid_tokens.zero_()
        self._handles.clear()
        self._free_page_ids = list(range(self.num_pages))

    def set_valid_tokens(
        self,
        *,
        handle: CacheGenKVPageHandle,
        valid_tokens: int,
    ) -> None:
        """Set the number of initialized token slots in one leased page."""
        if not 0 <= valid_tokens <= handle.format.tokens_per_page:
            raise ValueError(
                "valid_tokens must be in "
                f"[0, {handle.format.tokens_per_page}], got {valid_tokens}"
            )
        self.page_bytes_view(handle)
        self.page_valid_tokens[handle.page_id] = valid_tokens

    def get_valid_tokens(
        self,
        handle: CacheGenKVPageHandle,
    ) -> int:
        """Return the number of initialized token slots in one leased page."""
        self.page_bytes_view(handle)
        return int(self.page_valid_tokens[handle.page_id].item())

    def page_bytes_view(
        self,
        handle: CacheGenKVPageHandle,
    ) -> torch.Tensor:
        """Return the owned fixed-byte HBM page for a valid handle."""
        active_handle = self._handles.get(handle.page_id)
        if active_handle != handle:
            raise ValueError(
                f"Page handle {handle.page_id} is not active in this pool"
            )
        return self.page_bytes[handle.page_id]

    @property
    def num_allocated_pages(self) -> int:
        """Return the number of pages currently leased."""
        return len(self._handles)

    @property
    def num_free_pages(self) -> int:
        """Return the number of pages available to lease."""
        return len(self._free_page_ids)

    @property
    def persistent_nbytes(self) -> int:
        """Return persistent HBM allocation including page metadata tensors."""
        return (
            self.page_bytes.numel() * self.page_bytes.element_size()
            + self.page_quantizer_ids.numel()
            * self.page_quantizer_ids.element_size()
            + self.page_in_use.numel() * self.page_in_use.element_size()
            + self.page_valid_tokens.numel()
            * self.page_valid_tokens.element_size()
        )
