# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference INT2 affine quantization primitives for experimental KV caches."""

import torch

_INT2_LEVELS = 4
_INT2_MAX = _INT2_LEVELS - 1


def pack_int2(values: torch.Tensor) -> torch.Tensor:
    """Pack unsigned INT2 values along the last dimension into uint8 bytes."""
    if values.dtype != torch.uint8:
        raise ValueError("values must have dtype torch.uint8")
    if values.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be divisible by 4")
    if torch.any(values > _INT2_MAX):
        raise ValueError("values must be in the range [0, 3]")

    reshaped = values.reshape(*values.shape[:-1], -1, 4)
    return (
        reshaped[..., 0]
        | (reshaped[..., 1] << 2)
        | (reshaped[..., 2] << 4)
        | (reshaped[..., 3] << 6)
    ).contiguous()


def unpack_int2(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes into unsigned INT2 values along the last dimension."""
    if packed.dtype != torch.uint8:
        raise ValueError("packed must have dtype torch.uint8")

    return torch.stack(
        (
            packed & 0x03,
            (packed >> 2) & 0x03,
            (packed >> 4) & 0x03,
            (packed >> 6) & 0x03,
        ),
        dim=-1,
    ).reshape(*packed.shape[:-1], packed.shape[-1] * 4).contiguous()


def quantize_affine_int2(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize each final-axis vector to unsigned INT2 using affine scaling."""
    if not values.is_floating_point():
        raise ValueError("values must have a floating-point dtype")
    if values.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be divisible by 4")

    minimum = values.amin(dim=-1, keepdim=True)
    maximum = values.amax(dim=-1, keepdim=True)
    scale = (maximum - minimum) / _INT2_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    quantized = torch.round((values - minimum) / scale)
    quantized = torch.clamp(quantized, 0, _INT2_MAX).to(torch.uint8)

    return pack_int2(quantized), scale, minimum


def dequantize_affine_int2(
    packed: torch.Tensor,
    scale: torch.Tensor,
    minimum: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore affine-quantized INT2 values to the requested floating dtype."""
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")

    quantized = unpack_int2(packed).to(torch.float32)
    restored = quantized * scale.to(torch.float32) + minimum.to(torch.float32)
    return restored.to(dtype)
