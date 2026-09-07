# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compact typed INT8 K/V pages for experimental fused CacheGen attention.

The persistent representation contains only packed INT8 K/V pages, per-page
FP16 scales, FP16 running maxima, and token-validity bits. It intentionally
does not retain floating K/V staging pages.

When an arriving token increases a page/head maximum, existing quantized
entries in that page are rescaled and requantized from the previous INT8
representation. This is an eager correctness/reference writer; a later fused
append kernel can replace it without changing the read-side page layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
    CacheGenInt8PagedKV,
)

_INT8_QMAX = 127.0


@dataclass
class CacheGenInt8InferencePageStore:
    """One layer's compact typed paged INT8 K/V inference cache."""

    key_pages: torch.Tensor
    value_pages: torch.Tensor
    key_scales: torch.Tensor
    value_scales: torch.Tensor
    key_abs_max: torch.Tensor
    value_abs_max: torch.Tensor
    valid_tokens: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        num_pages: int,
        tokens_per_page: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
    ) -> "CacheGenInt8InferencePageStore":
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if head_size <= 0:
            raise ValueError("head_size must be positive")

        page_shape = (
            num_pages,
            tokens_per_page,
            num_kv_heads,
            head_size,
        )
        metadata_shape = (num_pages, num_kv_heads)
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
            key_scales=torch.zeros(
                metadata_shape,
                dtype=torch.float16,
                device=device,
            ),
            value_scales=torch.zeros(
                metadata_shape,
                dtype=torch.float16,
                device=device,
            ),
            key_abs_max=torch.zeros(
                metadata_shape,
                dtype=torch.float16,
                device=device,
            ),
            value_abs_max=torch.zeros(
                metadata_shape,
                dtype=torch.float16,
                device=device,
            ),
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

    def reset(self) -> None:
        """Clear every stored compact INT8 page and its metadata in place."""
        self.key_pages.zero_()
        self.value_pages.zero_()
        self.key_scales.zero_()
        self.value_scales.zero_()
        self.valid_tokens.zero_()

    @property
    def persistent_nbytes(self) -> int:
        """Bytes retained by the compact adaptive INT8 page store."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.key_pages,
                self.value_pages,
                self.key_scales,
                self.value_scales,
                self.key_abs_max,
                self.value_abs_max,
                self.valid_tokens,
            )
        )

    def as_paged_kv(self) -> CacheGenInt8PagedKV:
        """Return packed pages and float32 scales for fused decode attention."""
        return CacheGenInt8PagedKV(
            key_pages=self.key_pages,
            value_pages=self.value_pages,
            key_scales=self.key_scales.float(),
            value_scales=self.value_scales.float(),
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

    @staticmethod
    def _quantize_with_scales(
        values: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        safe_scales = torch.where(
            scales > 0,
            scales,
            torch.ones_like(scales),
        )
        return torch.clamp(
            torch.round(values.float() / safe_scales[None, :, None]),
            min=-_INT8_QMAX,
            max=_INT8_QMAX,
        ).to(torch.int8)

    @staticmethod
    def _dequantize_with_scales(
        values: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        return values.float() * scales.float()[None, :, None]

    def _requantize_page(
        self,
        *,
        page_id: int,
        new_key_abs_max: torch.Tensor,
        new_value_abs_max: torch.Tensor,
    ) -> None:
        """Rescale existing valid tokens and install new page/head scales."""
        old_key_scales = self.key_scales[page_id].float()
        old_value_scales = self.value_scales[page_id].float()
        new_key_scales = new_key_abs_max.float() / _INT8_QMAX
        new_value_scales = new_value_abs_max.float() / _INT8_QMAX

        valid_mask = self.valid_tokens[page_id]
        if bool(valid_mask.any()):
            decoded_keys = self._dequantize_with_scales(
                self.key_pages[page_id],
                old_key_scales,
            )
            decoded_values = self._dequantize_with_scales(
                self.value_pages[page_id],
                old_value_scales,
            )
            self.key_pages[page_id] = self._quantize_with_scales(
                decoded_keys,
                new_key_scales,
            )
            self.value_pages[page_id] = self._quantize_with_scales(
                decoded_values,
                new_value_scales,
            )
            self.key_pages[page_id, ~valid_mask] = 0
            self.value_pages[page_id, ~valid_mask] = 0

        self.key_abs_max[page_id] = new_key_abs_max.to(torch.float16)
        self.value_abs_max[page_id] = new_value_abs_max.to(torch.float16)
        self.key_scales[page_id] = new_key_scales.to(torch.float16)
        self.value_scales[page_id] = new_value_scales.to(torch.float16)

    def read_page(
        self,
        *,
        page_id: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference-dequantize one complete typed INT8 K/V page.

        This is intentionally a validation/mirroring helper. Fused attention
        must consume quantized page storage directly rather than materialize
        this decoded page in HBM.
        """
        if not 0 <= page_id < self.num_pages:
            raise ValueError(
                f"page_id must be in [0, {self.num_pages}), got {page_id}"
            )
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")

        keys = (
            self.key_pages[page_id].float()
            * self.key_scales[page_id][None, :, None]
        ).to(dtype)
        values = (
            self.value_pages[page_id].float()
            * self.value_scales[page_id][None, :, None]
        ).to(dtype)
        return keys, values

    def write_mapped_tokens(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
    ) -> None:
        """Write arbitrary token K/V entries into compact pages.

        Existing page entries are requantized when an arriving token grows a
        page/head scale. The method expects one physical page and offset per
        token, and supports repeated page IDs in the same write.
        """
        expected_token_shape = (self.num_kv_heads, self.head_size)
        if keys.ndim != 3 or tuple(keys.shape[1:]) != expected_token_shape:
            raise ValueError(
                "keys must have shape "
                f"[num_tokens, {expected_token_shape[0]}, "
                f"{expected_token_shape[1]}]; got {tuple(keys.shape)}"
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

        for page_id_tensor in torch.unique(page_ids):
            page_id = int(page_id_tensor.item())
            token_mask = page_ids == page_id
            page_offsets_i = page_offsets[token_mask]
            keys_i = keys[token_mask]
            values_i = values[token_mask]

            incoming_key_abs_max = keys_i.float().abs().amax(dim=(0, 2))
            incoming_value_abs_max = values_i.float().abs().amax(dim=(0, 2))
            new_key_abs_max = torch.maximum(
                self.key_abs_max[page_id].float(),
                incoming_key_abs_max,
            )
            new_value_abs_max = torch.maximum(
                self.value_abs_max[page_id].float(),
                incoming_value_abs_max,
            )

            scale_changed = (
                not torch.equal(
                    new_key_abs_max,
                    self.key_abs_max[page_id].float(),
                )
                or not torch.equal(
                    new_value_abs_max,
                    self.value_abs_max[page_id].float(),
                )
            )
            if scale_changed:
                self._requantize_page(
                    page_id=page_id,
                    new_key_abs_max=new_key_abs_max,
                    new_value_abs_max=new_value_abs_max,
                )

            key_scales = self.key_scales[page_id].float()
            value_scales = self.value_scales[page_id].float()
            self.key_pages[page_id, page_offsets_i] = (
                self._quantize_with_scales(keys_i, key_scales)
            )
            self.value_pages[page_id, page_offsets_i] = (
                self._quantize_with_scales(values_i, value_scales)
            )
            self.valid_tokens[page_id, page_offsets_i] = True
