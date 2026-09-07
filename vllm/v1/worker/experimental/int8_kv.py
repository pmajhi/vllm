import torch


def quantize_symmetric_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-last-dimension symmetric INT8 reference quantizer."""
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-8) / 127.0
    q = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_symmetric_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct floating-point values from INT8 values and scales."""
    return (q.float() * scale).to(dtype)


class Int8KVPage:
    """Reference INT8 storage for one physical KV page.

    Storage layout:
    - keys and values: [tokens_per_page, num_kv_heads, head_size], int8
    - key/value scales: [tokens_per_page, num_kv_heads, 1], float32

    This is a correctness reference only. It is not connected to vLLM's
    production cache allocator or attention backends.
    """

    def __init__(
        self,
        tokens_per_page: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
    ) -> None:
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if head_size <= 0:
            raise ValueError("head_size must be positive")

        shape = (tokens_per_page, num_kv_heads, head_size)
        scale_shape = (tokens_per_page, num_kv_heads, 1)

        self.tokens_per_page = tokens_per_page
        self.keys = torch.zeros(shape, dtype=torch.int8, device=device)
        self.values = torch.zeros(shape, dtype=torch.int8, device=device)
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
        page_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if not 0 <= page_offset < self.tokens_per_page:
            raise IndexError(
                f"page_offset must be in [0, {self.tokens_per_page}), "
                f"got {page_offset}"
            )

        expected_shape = self.keys.shape[1:]
        if key.shape != expected_shape:
            raise ValueError(
                f"key must have shape {expected_shape}, got {tuple(key.shape)}"
            )
        if value.shape != expected_shape:
            raise ValueError(
                f"value must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )

        quantized_key, key_scale = quantize_symmetric_int8(key)
        quantized_value, value_scale = quantize_symmetric_int8(value)

        self.keys[page_offset] = quantized_key
        self.values[page_offset] = quantized_value
        self.key_scales[page_offset] = key_scale
        self.value_scales[page_offset] = value_scale

    def read(
        self,
        page_offset: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= page_offset < self.tokens_per_page:
            raise IndexError(
                f"page_offset must be in [0, {self.tokens_per_page}), "
                f"got {page_offset}"
            )

        key = dequantize_symmetric_int8(
            self.keys[page_offset],
            self.key_scales[page_offset],
            dtype,
        )
        value = dequantize_symmetric_int8(
            self.values[page_offset],
            self.value_scales[page_offset],
            dtype,
        )
        return key, value