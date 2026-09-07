import torch

from vllm.v1.worker.experimental.int8_kv import Int8KVPage
import math

import torch.nn.functional as F

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

        self.num_physical_pages = num_physical_pages
        self.pages = [
            Int8KVPage(
                tokens_per_page=tokens_per_page,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                device=device,
            )
            for _ in range(num_physical_pages)
        ]

    def write(
        self,
        physical_page_id: int,
        page_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._validate_page_id(physical_page_id)
        self.pages[physical_page_id].write(page_offset, key, value)

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
        expected_shape = self.pages[0].keys.shape[1:]

        if keys.shape != (num_tokens, *expected_shape):
            raise ValueError(
                f"keys must have shape {(num_tokens, *expected_shape)}, "
                f"got {tuple(keys.shape)}"
            )
        if values.shape != (num_tokens, *expected_shape):
            raise ValueError(
                f"values must have shape {(num_tokens, *expected_shape)}, "
                f"got {tuple(values.shape)}"
            )

        for token_index in range(num_tokens):
            self.write(
                physical_page_id=int(physical_page_ids[token_index].item()),
                page_offset=int(page_offsets[token_index].item()),
                key=keys[token_index],
                value=values[token_index],
            )

    def read(
        self,
        physical_page_id: int,
        page_offset: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_page_id(physical_page_id)
        return self.pages[physical_page_id].read(page_offset, dtype)

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

        keys = []
        values = []

        for token_index in range(physical_page_ids.numel()):
            key, value = self.read(
                physical_page_id=int(physical_page_ids[token_index].item()),
                page_offset=int(page_offsets[token_index].item()),
                dtype=dtype,
            )
            keys.append(key)
            values.append(value)

        return torch.stack(keys), torch.stack(values)

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

        expected_shape = self.pages[0].keys.shape[1:]
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