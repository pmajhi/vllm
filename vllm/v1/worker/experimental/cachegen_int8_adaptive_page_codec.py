# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adaptive INT8 codec for fixed-byte heterogeneous CacheGen KV pages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.worker.experimental.cachegen_kv_page_format import (
    CacheGenKVPageFormat,
)
from vllm.v1.worker.experimental.cachegen_kv_page_pool import (
    CacheGenKVPageHandle,
    CacheGenKVPagePool,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


@dataclass(frozen=True)
class CacheGenInt8AdaptivePageLayout:
    """Concrete tensor geometry carried by one fixed-byte INT8 page."""

    tokens_per_page: int
    num_kv_heads: int
    head_size: int

    @property
    def values_per_page(self) -> int:
        """Return scalar K/V values represented by one token page."""
        return self.tokens_per_page * self.num_kv_heads * self.head_size

    @property
    def scale_count(self) -> int:
        """Return one dynamic scale per K/V head."""
        return self.num_kv_heads


class CacheGenInt8AdaptivePageCodec:
    """Read/write adaptive INT8 K/V tensors through a fixed-byte page ABI.

    Metadata layout, all in little-endian FP32:
      [key scales: num_kv_heads]
      [value scales: num_kv_heads]

    Payload layout:
      [INT8 key values]
      [INT8 value values]

    The codec is intentionally CPU/GPU tensor-native and does not create
    dequantized K/V page copies. A later fused Triton path will consume the
    same scale and payload regions directly from the common byte-page slab.
    """

    def __init__(
        self,
        *,
        page_format: CacheGenKVPageFormat,
        layout: CacheGenInt8AdaptivePageLayout,
    ) -> None:
        if page_format.quantizer is not CacheGenKVQuantizer.INT8_ADAPTIVE:
            raise ValueError(
                "Adaptive INT8 codec requires int8_adaptive page format"
            )
        if not page_format.implemented:
            raise ValueError(
                "Adaptive INT8 codec requires an implemented page format"
            )

        self.page_format = page_format
        self.layout = layout
        self.key_scales_by_page: torch.Tensor | None = None
        self.value_scales_by_page: torch.Tensor | None = None

        required_metadata_nbytes = 2 * layout.scale_count * 4
        if page_format.metadata_nbytes < required_metadata_nbytes:
            raise ValueError(
                "INT8 page metadata region is too small for K/V FP32 scales: "
                f"required={required_metadata_nbytes}, "
                f"available={page_format.metadata_nbytes}"
            )

        required_payload_nbytes = 2 * layout.values_per_page
        available_payload_nbytes = (
            page_format.key_payload_nbytes
            + page_format.value_payload_nbytes
        )
        if available_payload_nbytes < required_payload_nbytes:
            raise ValueError(
                "INT8 page payload region is too small for K/V values: "
                f"required={required_payload_nbytes}, "
                f"available={available_payload_nbytes}"
            )

        if page_format.key_payload_nbytes < layout.values_per_page:
            raise ValueError("INT8 key payload region is too small")
        if page_format.value_payload_nbytes < layout.values_per_page:
            raise ValueError("INT8 value payload region is too small")

    def _require_active_handle(
        self,
        *,
        pool: CacheGenKVPagePool,
        handle: CacheGenKVPageHandle,
    ) -> torch.Tensor:
        if handle.quantizer is not CacheGenKVQuantizer.INT8_ADAPTIVE:
            raise ValueError("Adaptive INT8 codec received wrong page handle")
        return pool.page_bytes_view(handle)

    def write_page(
        self,
        *,
        pool: CacheGenKVPagePool,
        handle: CacheGenKVPageHandle,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_tokens: int | None = None,
    ) -> None:
        """Dynamically quantize one full or prefix K/V token page into bytes."""
        if valid_tokens is None:
            valid_tokens = self.layout.tokens_per_page
        if not 0 < valid_tokens <= self.layout.tokens_per_page:
            raise ValueError(
                "valid_tokens must be in "
                f"[1, {self.layout.tokens_per_page}], got {valid_tokens}"
            )

        expected_shape = (
            valid_tokens,
            self.layout.num_kv_heads,
            self.layout.head_size,
        )
        if keys.shape != expected_shape:
            raise ValueError(
                f"keys must have shape {expected_shape}, got {tuple(keys.shape)}"
            )
        if values.shape != expected_shape:
            raise ValueError(
                f"values must have shape {expected_shape}, "
                f"got {tuple(values.shape)}"
            )
        pool_device = pool.page_bytes.device
        if keys.device != pool_device or values.device != pool_device:
            raise ValueError("keys and values must be on the pool device")
        if not keys.dtype.is_floating_point or not values.dtype.is_floating_point:
            raise ValueError("keys and values must be floating point")

        page = self._require_active_handle(pool=pool, handle=handle)
        page.zero_()

        if self.key_scales_by_page is None:
            self.key_scales_by_page = torch.zeros(
                (pool.num_pages, self.layout.scale_count),
                dtype=torch.float32,
                device=pool.page_bytes.device,
            )
            self.value_scales_by_page = torch.zeros_like(
                self.key_scales_by_page
            )

        assert self.value_scales_by_page is not None

        pool.set_valid_tokens(handle=handle, valid_tokens=valid_tokens)

        key_scales = (
            keys.float().abs().amax(dim=(0, 2)).clamp_min(1e-8) / 127.0
        )
        value_scales = (
            values.float().abs().amax(dim=(0, 2)).clamp_min(1e-8) / 127.0
        )

        self.key_scales_by_page[handle.page_id].copy_(key_scales)
        self.value_scales_by_page[handle.page_id].copy_(value_scales)

        key_quantized = torch.round(
            keys.float() / key_scales[None, :, None]
        ).clamp(-127, 127).to(torch.int8)
        value_quantized = torch.round(
            values.float() / value_scales[None, :, None]
        ).clamp(-127, 127).to(torch.int8)

        scale_nbytes = self.layout.scale_count * 4
        metadata = page[
            self.page_format.metadata_offset:
            self.page_format.metadata_offset + 2 * scale_nbytes
        ]
        metadata.view(torch.float32)[:self.layout.scale_count].copy_(
            key_scales
        )
        metadata.view(torch.float32)[
            self.layout.scale_count:2 * self.layout.scale_count
        ].copy_(value_scales)

        key_payload = page[
            self.page_format.key_payload_offset:
            self.page_format.key_payload_offset + self.layout.values_per_page
        ].view(torch.int8)
        key_payload[:key_quantized.numel()].copy_(key_quantized.reshape(-1))

        value_payload = page[
            self.page_format.value_payload_offset:
            self.page_format.value_payload_offset + self.layout.values_per_page
        ].view(torch.int8)
        value_payload[:value_quantized.numel()].copy_(
            value_quantized.reshape(-1)
        )

    def read_page(
        self,
        *,
        pool: CacheGenKVPagePool,
        handle: CacheGenKVPageHandle,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize one complete K/V page for reference validation only."""
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")

        page = self._require_active_handle(pool=pool, handle=handle)
        scale_nbytes = self.layout.scale_count * 4
        metadata = page[
            self.page_format.metadata_offset:
            self.page_format.metadata_offset + 2 * scale_nbytes
        ]

        key_scales = metadata.view(torch.float32)[
            :self.layout.scale_count
        ]
        value_scales = metadata.view(torch.float32)[
            self.layout.scale_count:2 * self.layout.scale_count
        ]

        key_payload = page[
            self.page_format.key_payload_offset:
            self.page_format.key_payload_offset + self.layout.values_per_page
        ].view(torch.int8)
        value_payload = page[
            self.page_format.value_payload_offset:
            self.page_format.value_payload_offset + self.layout.values_per_page
        ].view(torch.int8)

        shape = (
            self.layout.tokens_per_page,
            self.layout.num_kv_heads,
            self.layout.head_size,
        )
        keys = (
            key_payload.reshape(shape).float()
            * key_scales[None, :, None]
        ).to(dtype)
        values = (
            value_payload.reshape(shape).float()
            * value_scales[None, :, None]
        ).to(dtype)
        return keys, values
