# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference fixed-byte CacheGen-style INT8 KV-page adapter.

This implements CacheGen-style vector quantization only: unsigned bins plus
per-token/per-head FP16 maximum magnitudes. It deliberately excludes CacheGen
delta transforms and arithmetic coding so pages retain direct token access for
paged attention experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.worker.experimental.cachegen_int8_page_geometry import (
    CacheGenInt8PageGeometry,
    make_cachegen_int8_page_geometry,
)


@dataclass(frozen=True)
class CacheGenInt8BytePageLayout:
    """Byte offsets for a direct-access CacheGen-style INT8 page."""

    geometry: CacheGenInt8PageGeometry
    key_payload_offset: int
    value_payload_offset: int
    key_max_offset: int
    value_max_offset: int
    used_bytes: int

    @classmethod
    def from_geometry(
        cls,
        geometry: CacheGenInt8PageGeometry,
    ) -> "CacheGenInt8BytePageLayout":
        key_payload_offset = 0
        value_payload_offset = key_payload_offset + geometry.key_payload_bytes
        key_max_offset = value_payload_offset + geometry.value_payload_bytes
        value_max_offset = key_max_offset + geometry.key_max_bytes
        used_bytes = value_max_offset + geometry.value_max_bytes

        if used_bytes > geometry.page_bytes:
            raise ValueError("CacheGen-style layout exceeds physical page bytes")

        return cls(
            geometry=geometry,
            key_payload_offset=key_payload_offset,
            value_payload_offset=value_payload_offset,
            key_max_offset=key_max_offset,
            value_max_offset=value_max_offset,
            used_bytes=used_bytes,
        )

    @property
    def page_bytes(self) -> int:
        return self.geometry.page_bytes

    @property
    def tokens_per_page(self) -> int:
        return self.geometry.tokens_per_page

    @property
    def num_kv_heads(self) -> int:
        return self.geometry.num_kv_heads

    @property
    def head_size(self) -> int:
        return self.geometry.head_size

    @property
    def slack_bytes(self) -> int:
        return self.page_bytes - self.used_bytes

    @property
    def vector_bytes(self) -> int:
        return self.head_size

    @property
    def max_bytes_per_token(self) -> int:
        return self.num_kv_heads * 2

    def key_vector_offset(self, page_offset: int, kv_head: int) -> int:
        self._validate_page_offset(page_offset)
        self._validate_kv_head(kv_head)
        vector_index = page_offset * self.num_kv_heads + kv_head
        return self.key_payload_offset + vector_index * self.vector_bytes

    def value_vector_offset(self, page_offset: int, kv_head: int) -> int:
        self._validate_page_offset(page_offset)
        self._validate_kv_head(kv_head)
        vector_index = page_offset * self.num_kv_heads + kv_head
        return self.value_payload_offset + vector_index * self.vector_bytes

    def key_max_offset_for(self, page_offset: int, kv_head: int) -> int:
        self._validate_page_offset(page_offset)
        self._validate_kv_head(kv_head)
        max_index = page_offset * self.num_kv_heads + kv_head
        return self.key_max_offset + max_index * 2

    def value_max_offset_for(self, page_offset: int, kv_head: int) -> int:
        self._validate_page_offset(page_offset)
        self._validate_kv_head(kv_head)
        max_index = page_offset * self.num_kv_heads + kv_head
        return self.value_max_offset + max_index * 2

    def _validate_page_offset(self, page_offset: int) -> None:
        if not 0 <= page_offset < self.tokens_per_page:
            raise ValueError(
                f"page_offset must be in [0, {self.tokens_per_page}), "
                f"got {page_offset}"
            )

    def _validate_kv_head(self, kv_head: int) -> None:
        if not 0 <= kv_head < self.num_kv_heads:
            raise ValueError(
                f"kv_head must be in [0, {self.num_kv_heads}), got {kv_head}"
            )


class CacheGenInt8FixedBytePageAdapter:
    """Direct-access reference adapter over uint8 fixed-byte page storage."""

    _MAX_CODE = 127
    _ZERO_CODE = 127

    def __init__(
        self,
        *,
        page_pool: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        if page_pool.ndim != 2:
            raise ValueError("page_pool must have shape [num_pages, page_bytes]")
        if page_pool.dtype is not torch.uint8:
            raise ValueError("page_pool must have dtype torch.uint8")
        if page_pool.shape[0] <= 0:
            raise ValueError("page_pool must contain at least one page")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if head_size <= 0:
            raise ValueError("head_size must be positive")

        self.page_pool = page_pool
        geometry = make_cachegen_int8_page_geometry(
            page_bytes=page_pool.shape[1],
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        self.layout = CacheGenInt8BytePageLayout.from_geometry(geometry)

    @property
    def num_pages(self) -> int:
        return self.page_pool.shape[0]

    @property
    def page_bytes(self) -> int:
        return self.page_pool.shape[1]

    @property
    def tokens_per_page(self) -> int:
        return self.layout.tokens_per_page

    def write_token(
        self,
        *,
        page_id: int,
        page_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Quantize and store one token's K/V vectors in one fixed byte page."""
        self._validate_page_id(page_id)
        self._validate_token_shape(key, name="key")
        self._validate_token_shape(value, name="value")

        page = self.page_pool[page_id]
        for kv_head in range(self.layout.num_kv_heads):
            self._write_vector(
                page=page,
                vector=key[kv_head],
                payload_offset=self.layout.key_vector_offset(
                    page_offset,
                    kv_head,
                ),
                max_offset=self.layout.key_max_offset_for(
                    page_offset,
                    kv_head,
                ),
            )
            self._write_vector(
                page=page,
                vector=value[kv_head],
                payload_offset=self.layout.value_vector_offset(
                    page_offset,
                    kv_head,
                ),
                max_offset=self.layout.value_max_offset_for(
                    page_offset,
                    kv_head,
                ),
            )

    def read_token(
        self,
        *,
        page_id: int,
        page_offset: int,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read and dequantize one token's K/V vectors from one byte page."""
        self._validate_page_id(page_id)
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")

        page = self.page_pool[page_id]
        key = torch.empty(
            (self.layout.num_kv_heads, self.layout.head_size),
            device=page.device,
            dtype=dtype,
        )
        value = torch.empty_like(key)
        for kv_head in range(self.layout.num_kv_heads):
            key[kv_head] = self._read_vector(
                page=page,
                payload_offset=self.layout.key_vector_offset(
                    page_offset,
                    kv_head,
                ),
                max_offset=self.layout.key_max_offset_for(
                    page_offset,
                    kv_head,
                ),
                dtype=dtype,
            )
            value[kv_head] = self._read_vector(
                page=page,
                payload_offset=self.layout.value_vector_offset(
                    page_offset,
                    kv_head,
                ),
                max_offset=self.layout.value_max_offset_for(
                    page_offset,
                    kv_head,
                ),
                dtype=dtype,
            )
        return key, value

    def _write_vector(
        self,
        *,
        page: torch.Tensor,
        vector: torch.Tensor,
        payload_offset: int,
        max_offset: int,
    ) -> None:
        vector_fp32 = vector.to(torch.float32)
        max_abs = vector_fp32.abs().amax()
        max_tensor = max_abs.to(torch.float16).reshape(1)

        if max_abs == 0:
            quantized = torch.full(
                (self.layout.head_size,),
                self._ZERO_CODE,
                device=page.device,
                dtype=torch.uint8,
            )
        else:
            quantized = torch.round(
                vector_fp32 * (self._MAX_CODE / max_abs)
            ).clamp(
                min=-self._MAX_CODE,
                max=self._MAX_CODE,
            ).to(torch.int16)
            quantized = (quantized + self._ZERO_CODE).to(torch.uint8)

        page[payload_offset:payload_offset + self.layout.vector_bytes] = (
            quantized
        )
        page[max_offset:max_offset + 2] = (
            max_tensor.view(torch.uint8)
        )

    def _read_vector(
        self,
        *,
        page: torch.Tensor,
        payload_offset: int,
        max_offset: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        quantized = page[
            payload_offset:payload_offset + self.layout.vector_bytes
        ].to(torch.int16)
        max_tensor = page[max_offset:max_offset + 2].view(torch.float16)
        max_abs = max_tensor.reshape(1).to(torch.float32)[0]

        signed = quantized - self._ZERO_CODE
        return (signed.to(torch.float32) * (max_abs / self._MAX_CODE)).to(
            dtype
        )

    def _validate_page_id(self, page_id: int) -> None:
        if not 0 <= page_id < self.num_pages:
            raise ValueError(
                f"page_id must be in [0, {self.num_pages}), got {page_id}"
            )

    def _validate_token_shape(self, tensor: torch.Tensor, *, name: str) -> None:
        expected = (self.layout.num_kv_heads, self.layout.head_size)
        if tensor.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
            )
        if not tensor.dtype.is_floating_point:
            raise ValueError(f"{name} must have floating-point dtype")
        if tensor.device != self.page_pool.device:
            raise ValueError(
                f"{name} must be on {self.page_pool.device}, "
                f"got {tensor.device}"
            )


def write_cachegen_int8_mapped_tokens(
    *,
    adapter: CacheGenInt8FixedBytePageAdapter,
    keys: torch.Tensor,
    values: torch.Tensor,
    quantized_page_ids: torch.Tensor,
    quantized_page_offsets: torch.Tensor,
) -> None:
    """Reference-write scheduled K/V tokens using VLLM page mapping metadata.

    The mapping tensors contain one physical page ID and codec-local page
    offset per scheduled token. This intentionally uses a Python loop as a
    correctness reference before a fused GPU write kernel replaces it.
    """
    expected_token_shape = (
        adapter.layout.num_kv_heads,
        adapter.layout.head_size,
    )
    if keys.ndim != 3 or tuple(keys.shape[1:]) != expected_token_shape:
        raise ValueError(
            "keys must have shape "
            f"[num_tokens, {expected_token_shape[0]}, "
            f"{expected_token_shape[1]}], got {tuple(keys.shape)}"
        )
    if values.shape != keys.shape:
        raise ValueError(
            f"values must have shape {tuple(keys.shape)}, "
            f"got {tuple(values.shape)}"
        )
    if keys.device != adapter.page_pool.device:
        raise ValueError(
            f"keys must be on {adapter.page_pool.device}, got {keys.device}"
        )
    if values.device != adapter.page_pool.device:
        raise ValueError(
            f"values must be on {adapter.page_pool.device}, got {values.device}"
        )
    if not keys.dtype.is_floating_point:
        raise ValueError("keys must have floating-point dtype")
    if not values.dtype.is_floating_point:
        raise ValueError("values must have floating-point dtype")

    num_tokens = keys.shape[0]
    for name, mapping in (
        ("quantized_page_ids", quantized_page_ids),
        ("quantized_page_offsets", quantized_page_offsets),
    ):
        if mapping.ndim != 1 or mapping.shape[0] != num_tokens:
            raise ValueError(
                f"{name} must have shape [{num_tokens}], "
                f"got {tuple(mapping.shape)}"
            )
        if mapping.device != adapter.page_pool.device:
            raise ValueError(
                f"{name} must be on {adapter.page_pool.device}, "
                f"got {mapping.device}"
            )
        if mapping.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must have integer dtype")

    for token_index in range(num_tokens):
        adapter.write_token(
            page_id=int(quantized_page_ids[token_index]),
            page_offset=int(quantized_page_offsets[token_index]),
            key=keys[token_index],
            value=values[token_index],
        )
