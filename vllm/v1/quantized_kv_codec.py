from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedKVPageGeometry:
    """Physical-byte geometry for one codec's KV cache page."""

    physical_page_bytes: int
    page_header_bytes: int
    payload_bytes: int
    metadata_bytes: int
    tokens_per_page: int
    internal_padding_bytes: int


class QuantizedKVCodec(ABC):
    """Codec-specific storage and encode/decode contract for paged KV cache."""

    codec_id: int
    name: str

    @abstractmethod
    def key_metadata_shape(
        self,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Metadata shape for one stored key token."""
        raise NotImplementedError

    @abstractmethod
    def value_metadata_shape(
        self,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Metadata shape for one stored value token."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def storage_dtype(self) -> torch.dtype:
        """Dtype used for the encoded K and V payload."""
        raise NotImplementedError

    @abstractmethod
    def page_geometry(
        self,
        *,
        physical_page_bytes: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> QuantizedKVPageGeometry:
        """Return page capacity and byte partitioning for this codec."""

    @abstractmethod
    def encode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Encode one K/V token into payload tensors and codec metadata."""

    @abstractmethod
    def decode(
        self,
        encoded: tuple[torch.Tensor, ...],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one K/V token for attention."""

@dataclass(frozen=True)
class QuantizedKVCodecRegistry:
    """Immutable mapping from quantizer IDs to KV-cache codecs."""

    codecs: dict[int, QuantizedKVCodec]

    def get(self, codec_id: int) -> QuantizedKVCodec:
        try:
            return self.codecs[codec_id]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported quantizer_id: {codec_id}. "
                f"Supported IDs: {sorted(self.codecs)}"
            ) from exc