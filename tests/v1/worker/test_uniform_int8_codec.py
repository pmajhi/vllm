import pytest
import torch

from vllm.v1.quantized_kv_codec import QuantizedKVCodecRegistry
from vllm.v1.worker.experimental.uniform_int8_codec import (
    UniformInt8KVCodec,
)

def test_uniform_int8_codec_calculates_page_geometry() -> None:
    codec = UniformInt8KVCodec()

    geometry = codec.page_geometry(
        physical_page_bytes=4096,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
    )

    # Per token:
    # K + V values: 2 * 2 heads * 8 int8 values = 32 bytes.
    # K + V scales: 2 * 2 heads * 4 float32 bytes = 16 bytes.
    # Total: 48 bytes/token. 4096 // 48 = 85 tokens, 16 bytes padding.
    assert geometry.tokens_per_page == 85
    assert geometry.payload_bytes == 2720
    assert geometry.metadata_bytes == 1360
    assert geometry.internal_padding_bytes == 16


def test_uniform_int8_codec_round_trip() -> None:
    torch.manual_seed(0)
    codec = UniformInt8KVCodec()

    key = torch.randn(2, 8, dtype=torch.float32)
    value = torch.randn(2, 8, dtype=torch.float32)

    encoded = codec.encode(key, value)
    restored_key, restored_value = codec.decode(encoded, torch.float32)

    torch.testing.assert_close(restored_key, key, rtol=0.02, atol=0.02)
    torch.testing.assert_close(restored_value, value, rtol=0.02, atol=0.02)


@pytest.mark.parametrize(
    ("physical_page_bytes", "num_kv_heads", "head_size", "error_message"),
    [
        (0, 1, 1, "physical_page_bytes"),
        (64, 0, 1, "num_kv_heads"),
        (64, 1, 0, "head_size"),
        (1, 1, 1, "insufficient bytes"),
    ],
)
def test_uniform_int8_codec_rejects_invalid_geometry(
    physical_page_bytes: int,
    num_kv_heads: int,
    head_size: int,
    error_message: str,
) -> None:
    codec = UniformInt8KVCodec()

    with pytest.raises(ValueError, match=error_message):
        codec.page_geometry(
            physical_page_bytes=physical_page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.float16,
        )

def test_quantized_kv_codec_registry_returns_registered_codec() -> None:
    codec = UniformInt8KVCodec()
    registry = QuantizedKVCodecRegistry({codec.codec_id: codec})

    assert registry.get(1) is codec


def test_quantized_kv_codec_registry_rejects_unknown_codec() -> None:
    registry = QuantizedKVCodecRegistry({})

    with pytest.raises(ValueError, match="Unsupported quantizer_id: 99"):
        registry.get(99)

def test_uniform_int8_codec_describes_its_storage() -> None:
    codec = UniformInt8KVCodec()

    assert codec.storage_dtype is torch.int8
    assert codec.metadata_shape == (1,)