import pytest
import torch

from vllm.v1.worker.experimental.cachegen_kv_quantizers import (
    CacheGenKVQuantizerConfig,
    CacheGenKVQuantizerKind,
    KIVIReferenceCache,
    build_cachegen_kv_quantizer,
)


def _sample_kv(tokens: int = 192, heads: int = 2, head_size: int = 64):
    torch.manual_seed(0)
    keys = torch.randn(tokens, heads, head_size, dtype=torch.bfloat16)
    values = torch.randn(tokens, heads, head_size, dtype=torch.bfloat16)
    return keys, values


@pytest.mark.parametrize(
    "kind",
    [
        CacheGenKVQuantizerKind.BF16,
        CacheGenKVQuantizerKind.FP8,
        CacheGenKVQuantizerKind.ADAPTIVE_INT8,
        CacheGenKVQuantizerKind.KIVI_INT2,
    ],
)
def test_quantizer_round_trip_shape_and_finiteness(kind):
    keys, values = _sample_kv()
    quantizer = build_cachegen_kv_quantizer(
        CacheGenKVQuantizerConfig(
            kind=kind,
            kivi_group_size=32,
            kivi_residual_length=128,
        )
    )

    q_keys, q_values = quantizer.quantize(keys, values)
    d_keys, d_values = quantizer.dequantize(
        q_keys,
        q_values,
        dtype=torch.float32,
    )

    assert d_keys.shape == keys.shape
    assert d_values.shape == values.shape
    assert torch.isfinite(d_keys).all()
    assert torch.isfinite(d_values).all()


def test_kivi_preserves_full_precision_residual():
    keys, values = _sample_kv(tokens=192)
    residual_length = 128
    quantizer = build_cachegen_kv_quantizer(
        CacheGenKVQuantizerConfig(
            kind=CacheGenKVQuantizerKind.KIVI_INT2,
            kivi_group_size=32,
            kivi_residual_length=residual_length,
        )
    )

    q_keys, q_values = quantizer.quantize(keys, values)

    assert isinstance(q_keys, KIVIReferenceCache)
    assert isinstance(q_values, KIVIReferenceCache)
    assert q_keys.prefix_tokens == 64
    assert q_values.prefix_tokens == 64

    d_keys, d_values = quantizer.dequantize(
        q_keys,
        q_values,
        dtype=torch.bfloat16,
    )

    torch.testing.assert_close(
        d_keys[-residual_length:],
        keys[-residual_length:],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        d_values[-residual_length:],
        values[-residual_length:],
        rtol=0,
        atol=0,
    )


def test_kivi_quantizes_keys_by_channel_groups_and_values_per_token():
    keys, values = _sample_kv(tokens=192, heads=2, head_size=64)
    quantizer = build_cachegen_kv_quantizer(
        CacheGenKVQuantizerConfig(
            kind=CacheGenKVQuantizerKind.KIVI_INT2,
            kivi_group_size=32,
            kivi_residual_length=128,
        )
    )

    q_keys, q_values = quantizer.quantize(keys, values)

    assert q_keys.quantized_prefix.values.shape[-1] == 32
    assert q_values.quantized_prefix.values.shape == values[:64].shape
    assert q_keys.quantized_prefix.values.dtype == torch.uint8
    assert q_values.quantized_prefix.values.dtype == torch.uint8


def test_adaptive_int8_error_is_bounded_for_reference_input():
    keys, values = _sample_kv()
    quantizer = build_cachegen_kv_quantizer(
        CacheGenKVQuantizerConfig(
            kind=CacheGenKVQuantizerKind.ADAPTIVE_INT8,
        )
    )
    q_keys, q_values = quantizer.quantize(keys, values)
    d_keys, d_values = quantizer.dequantize(
        q_keys,
        q_values,
        dtype=torch.float32,
    )

    assert (d_keys - keys.float()).abs().max().item() < 0.05
    assert (d_values - values.float()).abs().max().item() < 0.05
