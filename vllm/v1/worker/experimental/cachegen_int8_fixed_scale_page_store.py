# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Append-safe fixed-scale INT8 K/V pages for fused CacheGen attention.

Each KV head uses one calibrated symmetric scale for all pages in one store.
Tokens are quantized exactly once during append. This representation contains
only INT8 K/V payloads, FP16 scale metadata, and validity bits; it has no
floating staging cache and no requantization path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
    CacheGenInt8PagedKV,
)

_INT8_QMAX = 127.0


@dataclass
class CacheGenInt8FixedScalePageStore:
    """One layer's append-safe fixed-scale typed INT8 K/V cache."""

    key_pages: torch.Tensor
    value_pages: torch.Tensor
    key_scales: torch.Tensor
    value_scales: torch.Tensor
    valid_tokens: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        num_pages: int,
        tokens_per_page: int,
        num_kv_heads: int,
        head_size: int,
        key_scales: torch.Tensor,
        value_scales: torch.Tensor,
        device: torch.device,
    ) -> "CacheGenInt8FixedScalePageStore":
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if head_size <= 0:
            raise ValueError("head_size must be positive")

        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        expected_scale_shape = (num_kv_heads,)
        for name, scales in (
            ("key_scales", key_scales),
            ("value_scales", value_scales),
        ):
            if tuple(scales.shape) != expected_scale_shape:
                raise ValueError(
                    f"{name} must have shape {expected_scale_shape}; got "
                    f"{tuple(scales.shape)}"
                )
            if not scales.dtype.is_floating_point:
                raise ValueError(f"{name} must have floating-point dtype")
            if scales.device != device:
                raise ValueError(
                    f"{name} must be on {device}; got {scales.device}"
                )
            if torch.any(scales <= 0):
                raise ValueError(f"{name} must be strictly positive")

        page_shape = (
            num_pages,
            tokens_per_page,
            num_kv_heads,
            head_size,
        )
        return cls(
            key_pages=torch.zeros(
                page_shape,
                dtype=torch.int8,
                device=device,
            ),
            value_pages=torch.zeros(
                page_shape,
                dtype=torch.int8,
                device=device,
            ),
            key_scales=key_scales.to(torch.float16).clone(),
            value_scales=value_scales.to(torch.float16).clone(),
            valid_tokens=torch.zeros(
                (num_pages, tokens_per_page),
                dtype=torch.bool,
                device=device,
            ),
        )

    @property
    def num_pages(self) -> int:
        return self.key_pages.shape[0]

    @property
    def tokens_per_page(self) -> int:
        return self.key_pages.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.key_pages.shape[2]

    @property
    def head_size(self) -> int:
        return self.key_pages.shape[3]

    @property
    def device(self) -> torch.device:
        return self.key_pages.device

    @property
    def persistent_nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.key_pages,
                self.value_pages,
                self.key_scales,
                self.value_scales,
                self.valid_tokens,
            )
        )

    def as_paged_kv(self) -> CacheGenInt8PagedKV:
        """Expand fixed head scales to the fused kernel's per-page contract."""
        return CacheGenInt8PagedKV(
            key_pages=self.key_pages,
            value_pages=self.value_pages,
            key_scales=self.key_scales.float().expand(
                self.num_pages,
                self.num_kv_heads,
            ),
            value_scales=self.value_scales.float().expand(
                self.num_pages,
                self.num_kv_heads,
            ),
        )

    def _validate_page_ids(self, page_ids: torch.Tensor) -> None:
        if page_ids.ndim != 1:
            raise ValueError(
                f"page_ids must be rank 1; got {tuple(page_ids.shape)}"
            )
        if page_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("page_ids must have integer dtype")
        if page_ids.device != self.device:
            raise ValueError(
                f"page_ids must be on {self.device}; got {page_ids.device}"
            )
        if torch.any(page_ids < 0) or torch.any(page_ids >= self.num_pages):
            raise ValueError(
                f"page_ids must lie in [0, {self.num_pages})"
            )

    def write_mapped_tokens(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
    ) -> None:
        """Quantize each arriving token once with fixed per-KV-head scales."""
        expected_shape = (self.num_kv_heads, self.head_size)
        if keys.ndim != 3 or tuple(keys.shape[1:]) != expected_shape:
            raise ValueError(
                "keys must have shape "
                f"[num_tokens, {expected_shape[0]}, {expected_shape[1]}]; "
                f"got {tuple(keys.shape)}"
            )
        if values.shape != keys.shape:
            raise ValueError(
                f"values must have shape {tuple(keys.shape)}; got "
                f"{tuple(values.shape)}"
            )
        if not keys.dtype.is_floating_point:
            raise ValueError("keys must have floating-point dtype")
        if not values.dtype.is_floating_point:
            raise ValueError("values must have floating-point dtype")
        if keys.device != self.device or values.device != self.device:
            raise ValueError(
                f"keys and values must be on {self.device}; got "
                f"{keys.device} and {values.device}"
            )

        self._validate_page_ids(page_ids)
        if page_offsets.ndim != 1 or page_offsets.shape != page_ids.shape:
            raise ValueError(
                "page_offsets must have the same rank-1 shape as page_ids"
            )
        if page_offsets.dtype not in (torch.int32, torch.int64):
            raise ValueError("page_offsets must have integer dtype")
        if page_offsets.device != self.device:
            raise ValueError(
                f"page_offsets must be on {self.device}; got "
                f"{page_offsets.device}"
            )
        if page_ids.numel() != keys.shape[0]:
            raise ValueError(
                f"page_ids must have shape [{keys.shape[0]}]; got "
                f"{tuple(page_ids.shape)}"
            )
        if (
            torch.any(page_offsets < 0)
            or torch.any(page_offsets >= self.tokens_per_page)
        ):
            raise ValueError(
                f"page_offsets must lie in [0, {self.tokens_per_page})"
            )

        key_scales = self.key_scales.float()
        value_scales = self.value_scales.float()
        key_quantized = torch.clamp(
            torch.round(keys.float() / key_scales[None, :, None]),
            min=-_INT8_QMAX,
            max=_INT8_QMAX,
        ).to(torch.int8)
        value_quantized = torch.clamp(
            torch.round(values.float() / value_scales[None, :, None]),
            min=-_INT8_QMAX,
            max=_INT8_QMAX,
        ).to(torch.int8)

        self.key_pages[page_ids, page_offsets] = key_quantized
        self.value_pages[page_ids, page_offsets] = value_quantized
        self.valid_tokens[page_ids, page_offsets] = True
