import math

import torch
import torch.nn.functional as F

from vllm.v1.worker.experimental.int8_kv import (
    dequantize_symmetric_int8,
    quantize_symmetric_int8,
)

class Int8PagedKVCache:
    """Reference INT8 paged KV cache addressed by page ID and page offset.

    This is a correctness-only cache. It is not connected to vLLM's production
    KV-cache allocator or attention backends.
    """

    def __init__(
        self,
        num_physical_pages: int,
        tokens_per_page: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
    ) -> None:
        if num_physical_pages <= 0:
            raise ValueError("num_physical_pages must be positive")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if head_size <= 0:
            raise ValueError("head_size must be positive")

        self.num_physical_pages = num_physical_pages
        self.tokens_per_page = tokens_per_page
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

        kv_shape = (
            num_physical_pages,
            tokens_per_page,
            num_kv_heads,
            head_size,
        )
        scale_shape = (
            num_physical_pages,
            tokens_per_page,
            num_kv_heads,
            1,
        )

        self.keys = torch.zeros(kv_shape, dtype=torch.int8, device=device)
        self.values = torch.zeros(kv_shape, dtype=torch.int8, device=device)
        self.key_scales = torch.ones(
            scale_shape,
            dtype=torch.float32,
            device=device,
        )
        self.value_scales = torch.ones(
            scale_shape,
            dtype=torch.float32,
            device=device,
        )

    def write(
        self,
        physical_page_id: int,
        page_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._validate_page_id(physical_page_id)
        self._validate_page_offset(page_offset)

        expected_shape = (self.num_kv_heads, self.head_size)
        if tuple(key.shape) != expected_shape:
            raise ValueError(
                f"key must have shape {expected_shape}, got {tuple(key.shape)}"
            )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"value must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )

        quantized_key, key_scale = quantize_symmetric_int8(key)
        quantized_value, value_scale = quantize_symmetric_int8(value)

        self.keys[physical_page_id, page_offset] = quantized_key
        self.values[physical_page_id, page_offset] = quantized_value
        self.key_scales[physical_page_id, page_offset] = key_scale
        self.value_scales[physical_page_id, page_offset] = value_scale

    def write_batch(
        self,
        physical_page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write packed K/V tokens using physical-page IDs and offsets."""
        if physical_page_ids.ndim != 1:
            raise ValueError("physical_page_ids must be one-dimensional")
        if page_offsets.shape != physical_page_ids.shape:
            raise ValueError(
                "page_offsets must have the same shape as physical_page_ids"
            )

        num_tokens = physical_page_ids.numel()
        expected_shape = (self.num_kv_heads, self.head_size)

        if tuple(keys.shape) != (num_tokens, *expected_shape):
            raise ValueError(
                f"keys must have shape {(num_tokens, *expected_shape)}, "
                f"got {tuple(keys.shape)}"
            )
        if tuple(values.shape) != (num_tokens, *expected_shape):
            raise ValueError(
                f"values must have shape {(num_tokens, *expected_shape)}, "
                f"got {tuple(values.shape)}"
            )
        if keys.device != self.keys.device:
            raise ValueError("keys must be on the cache device")
        if values.device != self.values.device:
            raise ValueError("values must be on the cache device")
        if physical_page_ids.device != self.keys.device:
            raise ValueError("physical_page_ids must be on the cache device")
        if page_offsets.device != self.keys.device:
            raise ValueError("page_offsets must be on the cache device")

        page_ids = physical_page_ids.to(torch.long)
        offsets = page_offsets.to(torch.long)
        self._validate_batch_addresses(page_ids, offsets)

        quantized_keys, key_scales = quantize_symmetric_int8(keys)
        quantized_values, value_scales = quantize_symmetric_int8(values)

        self.keys[page_ids, offsets] = quantized_keys
        self.values[page_ids, offsets] = quantized_values
        self.key_scales[page_ids, offsets] = key_scales
        self.value_scales[page_ids, offsets] = value_scales

    def read(
        self,
        physical_page_id: int,
        page_offset: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_page_id(physical_page_id)
        self._validate_page_offset(page_offset)

        key = dequantize_symmetric_int8(
            self.keys[physical_page_id, page_offset],
            self.key_scales[physical_page_id, page_offset],
            dtype,
        )
        value = dequantize_symmetric_int8(
            self.values[physical_page_id, page_offset],
            self.value_scales[physical_page_id, page_offset],
            dtype,
        )
        return key, value

    def read_batch(
        self,
        physical_page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read packed K/V tokens using physical-page IDs and offsets."""
        if physical_page_ids.ndim != 1:
            raise ValueError("physical_page_ids must be one-dimensional")
        if page_offsets.shape != physical_page_ids.shape:
            raise ValueError(
                "page_offsets must have the same shape as physical_page_ids"
            )
        if physical_page_ids.device != self.keys.device:
            raise ValueError("physical_page_ids must be on the cache device")
        if page_offsets.device != self.keys.device:
            raise ValueError("page_offsets must be on the cache device")

        page_ids = physical_page_ids.to(torch.long)
        offsets = page_offsets.to(torch.long)
        self._validate_batch_addresses(page_ids, offsets)

        keys = dequantize_symmetric_int8(
            self.keys[page_ids, offsets],
            self.key_scales[page_ids, offsets],
            dtype,
        )
        values = dequantize_symmetric_int8(
            self.values[page_ids, offsets],
            self.value_scales[page_ids, offsets],
            dtype,
        )
        return keys, values

    def _validate_batch_addresses(
        self,
        physical_page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
    ) -> None:
        if torch.any(physical_page_ids < 0):
            raise IndexError("physical_page_ids must be nonnegative")
        if torch.any(physical_page_ids >= self.num_physical_pages):
            raise IndexError("physical_page_ids exceed cache capacity")
        if torch.any(page_offsets < 0):
            raise IndexError("page_offsets must be nonnegative")
        if torch.any(page_offsets >= self.tokens_per_page):
            raise IndexError("page_offsets exceed page capacity")

        slots = physical_page_ids * self.tokens_per_page + page_offsets
        if torch.unique(slots).numel() != slots.numel():
            raise ValueError("batch contains duplicate physical page addresses")

    def _validate_page_offset(self, page_offset: int) -> None:
        if not 0 <= page_offset < self.tokens_per_page:
            raise IndexError(
                f"page_offset must be in [0, {self.tokens_per_page}), "
                f"got {page_offset}"
            )

    def _validate_page_id(self, physical_page_id: int) -> None:
        if not 0 <= physical_page_id < self.num_physical_pages:
            raise IndexError(
                f"physical_page_id must be in "
                f"[0, {self.num_physical_pages}), got {physical_page_id}"
            )

    def attention(
        self,
        query: torch.Tensor,
        physical_page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Reference attention over K/V gathered from quantized pages."""
        if query.ndim != 2:
            raise ValueError("query must have shape [num_kv_heads, head_size]")

        expected_shape = (self.num_kv_heads, self.head_size)
        if tuple(query.shape) != expected_shape:
            raise ValueError(
                f"query must have shape {expected_shape}, "
                f"got {tuple(query.shape)}"
            )

        keys, values = self.read_batch(
            physical_page_ids,
            page_offsets,
            query.dtype,
        )

        scale = 1.0 / math.sqrt(query.shape[-1])
        scores = torch.einsum("hd,thd->ht", query, keys) * scale
        weights = F.softmax(scores, dim=-1)
        return torch.einsum("ht,thd->hd", weights, values)