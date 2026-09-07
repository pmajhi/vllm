import torch

from vllm.v1.quantized_kv_codec import (
    QuantizedKVCodec,
    QuantizedKVPageGeometry,
)
from vllm.v1.worker.experimental.int8_kv import (
    dequantize_symmetric_int8,
    quantize_symmetric_int8,
)


class UniformInt8KVCodec(QuantizedKVCodec):
    """Symmetric INT8 reference codec with per-token/per-head scales."""

    codec_id = 1
    name = "uniform_int8"

    def key_metadata_shape(
        self,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_kv_heads, 1)

    def value_metadata_shape(
        self,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_kv_heads, 1)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.int8

    def page_geometry(
        self,
        *,
        physical_page_bytes: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> QuantizedKVPageGeometry:
        if physical_page_bytes <= 0:
            raise ValueError("physical_page_bytes must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if head_size <= 0:
            raise ValueError("head_size must be positive")

        page_header_bytes = 0
        scale_bytes = 4
        payload_bytes_per_token = 2 * num_kv_heads * head_size
        metadata_bytes_per_token = 2 * num_kv_heads * scale_bytes
        bytes_per_token = payload_bytes_per_token + metadata_bytes_per_token

        tokens_per_page = physical_page_bytes // bytes_per_token
        if tokens_per_page <= 0:
            raise ValueError(
                "physical page has insufficient bytes for one INT8 K/V token"
            )

        payload_bytes = tokens_per_page * payload_bytes_per_token
        metadata_bytes = tokens_per_page * metadata_bytes_per_token
        internal_padding_bytes = (
            physical_page_bytes
            - page_header_bytes
            - payload_bytes
            - metadata_bytes
        )

        return QuantizedKVPageGeometry(
            physical_page_bytes=physical_page_bytes,
            page_header_bytes=page_header_bytes,
            payload_bytes=payload_bytes,
            metadata_bytes=metadata_bytes,
            tokens_per_page=tokens_per_page,
            internal_padding_bytes=internal_padding_bytes,
        )

    def encode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        quantized_key, key_scale = quantize_symmetric_int8(key)
        quantized_value, value_scale = quantize_symmetric_int8(value)
        return quantized_key, quantized_value, key_scale, value_scale

    def decode(
        self,
        encoded: tuple[torch.Tensor, ...],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_key, quantized_value, key_scale, value_scale = encoded
        key = dequantize_symmetric_int8(quantized_key, key_scale, dtype)
        value = dequantize_symmetric_int8(
            quantized_value,
            value_scale,
            dtype,
        )
        return key, value