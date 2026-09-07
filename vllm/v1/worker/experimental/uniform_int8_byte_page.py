# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uniform INT8 K/V storage encoded inside fixed-byte physical pages."""

from dataclasses import dataclass

import torch

from vllm.v1.worker.experimental.int8_kv import (
    dequantize_symmetric_int8,
    quantize_symmetric_int8,
)


PAGE_HEADER_BYTES = 64


@dataclass(frozen=True)
class UniformInt8BytePageLayout:
    """Byte offsets for Uniform INT8 K/V storage in one physical page."""

    page_bytes: int
    num_kv_heads: int
    head_size: int
    tokens_per_page: int
    key_payload_offset: int
    value_payload_offset: int
    key_scales_offset: int
    value_scales_offset: int
    used_bytes: int

    @property
    def tail_bytes(self) -> int:
        return self.page_bytes - self.used_bytes

    @property
    def payload_bytes_per_token(self) -> int:
        return self.num_kv_heads * self.head_size

    @property
    def scale_bytes_per_token(self) -> int:
        return self.num_kv_heads * torch.empty(
            (), dtype=torch.float32
        ).element_size()


def make_uniform_int8_byte_page_layout(
    *,
    page_bytes: int,
    num_kv_heads: int,
    head_size: int,
) -> UniformInt8BytePageLayout:
    """Build a fixed-byte page layout for per-token/per-head INT8 K/V."""
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_size <= 0:
        raise ValueError("head_size must be positive")

    int8_bytes = torch.empty((), dtype=torch.int8).element_size()
    fp32_bytes = torch.empty((), dtype=torch.float32).element_size()

    payload_bytes_per_token = num_kv_heads * head_size * int8_bytes
    scale_bytes_per_token = num_kv_heads * fp32_bytes
    bytes_per_token = 2 * (payload_bytes_per_token + scale_bytes_per_token)

    usable_bytes = page_bytes - PAGE_HEADER_BYTES
    tokens_per_page = usable_bytes // bytes_per_token
    if tokens_per_page <= 0:
        raise ValueError(
            "page_bytes has insufficient usable bytes for one INT8 K/V token"
        )

    key_payload_offset = PAGE_HEADER_BYTES
    value_payload_offset = (
        key_payload_offset + tokens_per_page * payload_bytes_per_token
    )
    key_scales_offset = (
        value_payload_offset + tokens_per_page * payload_bytes_per_token
    )
    value_scales_offset = (
        key_scales_offset + tokens_per_page * scale_bytes_per_token
    )
    used_bytes = (
        value_scales_offset + tokens_per_page * scale_bytes_per_token
    )

    return UniformInt8BytePageLayout(
        page_bytes=page_bytes,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        tokens_per_page=tokens_per_page,
        key_payload_offset=key_payload_offset,
        value_payload_offset=value_payload_offset,
        key_scales_offset=key_scales_offset,
        value_scales_offset=value_scales_offset,
        used_bytes=used_bytes,
    )


class UniformInt8BytePagePool:
    """Reference fixed-byte page pool with Uniform INT8 K/V entries.

    This is intentionally a tensor/view reference implementation. It proves
    the on-page layout and page-ID/offset semantics before fused kernels are
    introduced. It is not the final vLLM attention backend.
    """

    def __init__(
        self,
        *,
        num_pages: int,
        page_bytes: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
    ) -> None:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")

        self.layout = make_uniform_int8_byte_page_layout(
            page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        self.num_pages = num_pages
        self.device = device
        self.pages = torch.zeros(
            (num_pages, page_bytes),
            dtype=torch.uint8,
            device=device,
        )

    def _validate_address(self, page_id: int, page_offset: int) -> None:
        if not 0 <= page_id < self.num_pages:
            raise IndexError(
                f"page_id must be in [0, {self.num_pages}), got {page_id}"
            )
        if not 0 <= page_offset < self.layout.tokens_per_page:
            raise IndexError(
                "page_offset must be in "
                f"[0, {self.layout.tokens_per_page}), got {page_offset}"
            )

    def _payload_view(self, page_id: int, offset: int) -> torch.Tensor:
        byte_count = self.layout.payload_bytes_per_token
        start = offset * byte_count
        return self.pages[
            page_id,
            start:start + byte_count,
        ]

    def _scale_view(self, page_id: int, offset: int) -> torch.Tensor:
        byte_count = self.layout.scale_bytes_per_token
        start = offset * byte_count
        return self.pages[
            page_id,
            start:start + byte_count,
        ]

    def _typed_view(
        self,
        *,
        page_id: int,
        base_offset: int,
        page_offset: int,
        byte_count: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        start = base_offset + page_offset * byte_count
        end = start + byte_count
        return self.pages[page_id, start:end].view(dtype).view(shape)

    def write(
        self,
        *,
        page_id: int,
        page_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Quantize and write one K/V token into one physical byte page."""
        self._validate_address(page_id, page_offset)

        expected_shape = (self.layout.num_kv_heads, self.layout.head_size)
        if tuple(key.shape) != expected_shape:
            raise ValueError(
                f"key must have shape {expected_shape}, got {tuple(key.shape)}"
            )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"value must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )
        if key.device != self.device or value.device != self.device:
            raise ValueError("key and value must be on the page-pool device")

        quantized_key, key_scale = quantize_symmetric_int8(key)
        quantized_value, value_scale = quantize_symmetric_int8(value)

        payload_bytes = self.layout.payload_bytes_per_token
        scale_bytes = self.layout.scale_bytes_per_token
        payload_shape = expected_shape
        scale_shape = (self.layout.num_kv_heads, 1)

        self._typed_view(
            page_id=page_id,
            base_offset=self.layout.key_payload_offset,
            page_offset=page_offset,
            byte_count=payload_bytes,
            dtype=torch.int8,
            shape=payload_shape,
        ).copy_(quantized_key)

        self._typed_view(
            page_id=page_id,
            base_offset=self.layout.value_payload_offset,
            page_offset=page_offset,
            byte_count=payload_bytes,
            dtype=torch.int8,
            shape=payload_shape,
        ).copy_(quantized_value)

        self._typed_view(
            page_id=page_id,
            base_offset=self.layout.key_scales_offset,
            page_offset=page_offset,
            byte_count=scale_bytes,
            dtype=torch.float32,
            shape=scale_shape,
        ).copy_(key_scale)

        self._typed_view(
            page_id=page_id,
            base_offset=self.layout.value_scales_offset,
            page_offset=page_offset,
            byte_count=scale_bytes,
            dtype=torch.float32,
            shape=scale_shape,
        ).copy_(value_scale)

    def read(
        self,
        *,
        page_id: int,
        page_offset: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read and dequantize one K/V token from one physical byte page."""
        self._validate_address(page_id, page_offset)

        payload_bytes = self.layout.payload_bytes_per_token
        scale_bytes = self.layout.scale_bytes_per_token
        payload_shape = (
            self.layout.num_kv_heads,
            self.layout.head_size,
        )
        scale_shape = (self.layout.num_kv_heads, 1)

        quantized_key = self._typed_view(
            page_id=page_id,
            base_offset=self.layout.key_payload_offset,
            page_offset=page_offset,
            byte_count=payload_bytes,
            dtype=torch.int8,
            shape=payload_shape,
        )
        quantized_value = self._typed_view(
            page_id=page_id,
            base_offset=self.layout.value_payload_offset,
            page_offset=page_offset,
            byte_count=payload_bytes,
            dtype=torch.int8,
            shape=payload_shape,
        )
        key_scale = self._typed_view(
            page_id=page_id,
            base_offset=self.layout.key_scales_offset,
            page_offset=page_offset,
            byte_count=scale_bytes,
            dtype=torch.float32,
            shape=scale_shape,
        )
        value_scale = self._typed_view(
            page_id=page_id,
            base_offset=self.layout.value_scales_offset,
            page_offset=page_offset,
            byte_count=scale_bytes,
            dtype=torch.float32,
            shape=scale_shape,
        )

        return (
            dequantize_symmetric_int8(quantized_key, key_scale, dtype),
            dequantize_symmetric_int8(quantized_value, value_scale, dtype),
        )

    def write_batch(
        self,
        *,
        physical_page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Quantize and write packed K/V tokens to fixed-byte pages."""
        if physical_page_ids.ndim != 1:
            raise ValueError("physical_page_ids must be one-dimensional")
        if page_offsets.shape != physical_page_ids.shape:
            raise ValueError(
                "page_offsets must have the same shape as physical_page_ids"
            )

        num_tokens = physical_page_ids.numel()
        expected_shape = (
            num_tokens,
            self.layout.num_kv_heads,
            self.layout.head_size,
        )
        if tuple(keys.shape) != expected_shape:
            raise ValueError(
                f"keys must have shape {expected_shape}, got {tuple(keys.shape)}"
            )
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"values must have shape {expected_shape}, "
                f"got {tuple(values.shape)}"
            )
        if physical_page_ids.device != self.device:
            raise ValueError(
                "physical_page_ids must be on the page-pool device"
            )
        if page_offsets.device != self.device:
            raise ValueError("page_offsets must be on the page-pool device")
        if keys.device != self.device or values.device != self.device:
            raise ValueError("keys and values must be on the page-pool device")

        for token_index in range(num_tokens):
            self.write(
                page_id=int(physical_page_ids[token_index]),
                page_offset=int(page_offsets[token_index]),
                key=keys[token_index],
                value=values[token_index],
            )

    def read_batch(
        self,
        *,
        physical_page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read packed K/V tokens from fixed-byte pages in input order."""
        if physical_page_ids.ndim != 1:
            raise ValueError("physical_page_ids must be one-dimensional")
        if page_offsets.shape != physical_page_ids.shape:
            raise ValueError(
                "page_offsets must have the same shape as physical_page_ids"
            )
        if physical_page_ids.device != self.device:
            raise ValueError(
                "physical_page_ids must be on the page-pool device"
            )
        if page_offsets.device != self.device:
            raise ValueError("page_offsets must be on the page-pool device")

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        for token_index in range(physical_page_ids.numel()):
            key, value = self.read(
                page_id=int(physical_page_ids[token_index]),
                page_offset=int(page_offsets[token_index]),
                dtype=dtype,
            )
            keys.append(key)
            values.append(value)

        expected_shape = (
            0,
            self.layout.num_kv_heads,
            self.layout.head_size,
        )
        if not keys:
            empty = torch.empty(
                expected_shape,
                dtype=dtype,
                device=self.device,
            )
            return empty, empty.clone()

        return torch.stack(keys), torch.stack(values)
