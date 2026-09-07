from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class CacheGenKVQuantizerKind(str, Enum):
    BF16 = "bf16"
    FP8 = "fp8"
    ADAPTIVE_INT8 = "adaptive_int8"
    KIVI_INT2 = "kivi_int2"


@dataclass(frozen=True)
class CacheGenKVQuantizerConfig:
    kind: CacheGenKVQuantizerKind
    page_size: int = 16
    kivi_group_size: int = 32
    kivi_residual_length: int = 128


@dataclass
class AffineQuantizedTensor:
    values: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    original_shape: tuple[int, ...]


def _require_floating(x: torch.Tensor) -> None:
    if not x.is_floating_point():
        raise TypeError(f"expected floating-point input, got {x.dtype}")


def _affine_quantize_last_dim(
    x: torch.Tensor,
    bits: int,
) -> AffineQuantizedTensor:
    _require_floating(x)
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")

    qmax = (1 << bits) - 1
    x32 = x.float()
    x_min = x32.amin(dim=-1, keepdim=True)
    x_max = x32.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp_min(1e-8) / qmax
    zero = torch.round((-x_min) / scale).clamp(0, qmax)
    values = torch.round(x32 / scale + zero).clamp(0, qmax).to(torch.uint8)
    return AffineQuantizedTensor(
        values=values,
        scale=scale,
        zero=zero,
        original_shape=tuple(x.shape),
    )


def _affine_dequantize_last_dim(
    quantized: AffineQuantizedTensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (
        (quantized.values.float() - quantized.zero) * quantized.scale
    ).to(dtype)


class CacheGenKVQuantizer:
    config: CacheGenKVQuantizerConfig

    def quantize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[object, object]:
        raise NotImplementedError

    def dequantize(
        self,
        keys: object,
        values: object,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BF16KVQuantizer(CacheGenKVQuantizer):
    def __init__(self, config: CacheGenKVQuantizerConfig) -> None:
        self.config = config

    def quantize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return keys.to(torch.bfloat16), values.to(torch.bfloat16)

    def dequantize(
        self,
        keys: object,
        values: object,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(keys, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        return keys.to(dtype), values.to(dtype)


class FP8ReferenceKVQuantizer(CacheGenKVQuantizer):
    def __init__(self, config: CacheGenKVQuantizerConfig) -> None:
        self.config = config
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is unavailable")

    def quantize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_floating(keys)
        _require_floating(values)
        return (
            keys.to(torch.float8_e4m3fn),
            values.to(torch.float8_e4m3fn),
        )

    def dequantize(
        self,
        keys: object,
        values: object,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(keys, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        return keys.to(dtype), values.to(dtype)


class AdaptiveInt8ReferenceKVQuantizer(CacheGenKVQuantizer):
    def __init__(self, config: CacheGenKVQuantizerConfig) -> None:
        self.config = config

    def quantize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[AffineQuantizedTensor, AffineQuantizedTensor]:
        return (
            _affine_quantize_last_dim(keys, bits=8),
            _affine_quantize_last_dim(values, bits=8),
        )

    def dequantize(
        self,
        keys: object,
        values: object,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(keys, AffineQuantizedTensor)
        assert isinstance(values, AffineQuantizedTensor)
        return (
            _affine_dequantize_last_dim(keys, dtype),
            _affine_dequantize_last_dim(values, dtype),
        )


@dataclass
class KIVIReferenceCache:
    quantized_prefix: AffineQuantizedTensor
    residual: torch.Tensor
    prefix_tokens: int


class KIVIReferenceKVQuantizer(CacheGenKVQuantizer):
    """Reference KIVI-style mixed cache.

    This module models the essential KIVI storage contract:
    keys use channel groups, values use token groups, and the newest
    residual_length tokens remain in full precision. It is a reference
    codec, not the final packed 2-bit Triton decode format.
    """

    def __init__(self, config: CacheGenKVQuantizerConfig) -> None:
        self.config = config
        if config.kivi_group_size <= 0:
            raise ValueError("kivi_group_size must be positive")
        if config.kivi_residual_length < 0:
            raise ValueError("kivi_residual_length must be nonnegative")

    def _split_prefix_residual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual_length = min(self.config.kivi_residual_length, x.shape[0])
        prefix_tokens = x.shape[0] - residual_length
        return x[:prefix_tokens], x[prefix_tokens:]

    def _quantize_keys(self, keys: torch.Tensor) -> KIVIReferenceCache:
        prefix, residual = self._split_prefix_residual(keys)
        if prefix.numel() == 0:
            empty = AffineQuantizedTensor(
                values=torch.empty_like(prefix, dtype=torch.uint8),
                scale=torch.empty(
                    (*prefix.shape[:-1], 1),
                    dtype=torch.float32,
                    device=prefix.device,
                ),
                zero=torch.empty(
                    (*prefix.shape[:-1], 1),
                    dtype=torch.float32,
                    device=prefix.device,
                ),
                original_shape=tuple(prefix.shape),
            )
            return KIVIReferenceCache(empty, residual, 0)

        group = self.config.kivi_group_size
        head_size = prefix.shape[-1]
        padded = (group - head_size % group) % group
        grouped = torch.nn.functional.pad(prefix, (0, padded))
        grouped = grouped.view(*prefix.shape[:-1], -1, group)
        q = _affine_quantize_last_dim(grouped, bits=2)
        return KIVIReferenceCache(q, residual, prefix.shape[0])

    def _dequantize_keys(
        self,
        cache: KIVIReferenceCache,
        dtype: torch.dtype,
        head_size: int,
    ) -> torch.Tensor:
        if cache.prefix_tokens == 0:
            return cache.residual.to(dtype)
        prefix = _affine_dequantize_last_dim(
            cache.quantized_prefix,
            dtype,
        ).reshape(cache.prefix_tokens, *cache.residual.shape[1:-1], -1)
        prefix = prefix[..., :head_size]
        return torch.cat((prefix, cache.residual.to(dtype)), dim=0)

    def _quantize_values(self, values: torch.Tensor) -> KIVIReferenceCache:
        prefix, residual = self._split_prefix_residual(values)
        if prefix.numel() == 0:
            empty = AffineQuantizedTensor(
                values=torch.empty_like(prefix, dtype=torch.uint8),
                scale=torch.empty(
                    (*prefix.shape[:-1], 1),
                    dtype=torch.float32,
                    device=prefix.device,
                ),
                zero=torch.empty(
                    (*prefix.shape[:-1], 1),
                    dtype=torch.float32,
                    device=prefix.device,
                ),
                original_shape=tuple(prefix.shape),
            )
            return KIVIReferenceCache(empty, residual, 0)
        return KIVIReferenceCache(
            _affine_quantize_last_dim(prefix, bits=2),
            residual,
            prefix.shape[0],
        )

    def _dequantize_values(
        self,
        cache: KIVIReferenceCache,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if cache.prefix_tokens == 0:
            return cache.residual.to(dtype)
        prefix = _affine_dequantize_last_dim(cache.quantized_prefix, dtype)
        return torch.cat((prefix, cache.residual.to(dtype)), dim=0)

    def quantize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[KIVIReferenceCache, KIVIReferenceCache]:
        _require_floating(keys)
        _require_floating(values)
        if keys.shape != values.shape:
            raise ValueError("keys and values must have matching shapes")
        if keys.ndim < 2:
            raise ValueError("expected [tokens, ..., head_size]")
        return self._quantize_keys(keys), self._quantize_values(values)

    def dequantize(
        self,
        keys: object,
        values: object,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(keys, KIVIReferenceCache)
        assert isinstance(values, KIVIReferenceCache)
        head_size = keys.residual.shape[-1]
        return (
            self._dequantize_keys(keys, dtype, head_size),
            self._dequantize_values(values, dtype),
        )


def build_cachegen_kv_quantizer(
    config: CacheGenKVQuantizerConfig,
) -> CacheGenKVQuantizer:
    if config.kind is CacheGenKVQuantizerKind.BF16:
        return BF16KVQuantizer(config)
    if config.kind is CacheGenKVQuantizerKind.FP8:
        return FP8ReferenceKVQuantizer(config)
    if config.kind is CacheGenKVQuantizerKind.ADAPTIVE_INT8:
        return AdaptiveInt8ReferenceKVQuantizer(config)
    if config.kind is CacheGenKVQuantizerKind.KIVI_INT2:
        return KIVIReferenceKVQuantizer(config)
    raise ValueError(f"unsupported quantizer kind: {config.kind}")
