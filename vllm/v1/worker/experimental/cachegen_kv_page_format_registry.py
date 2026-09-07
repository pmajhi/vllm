# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common fixed-byte page formats for heterogeneous experimental KV storage."""

from __future__ import annotations

from vllm.v1.worker.experimental.cachegen_kv_page_format import (
    CacheGenKVPageFormat,
)
from vllm.v1.worker.experimental.cachegen_kv_page_geometry import (
    CacheGenKVPageGeometry,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)

DEFAULT_CACHEGEN_KV_PAGE_NBYTES = 4096
DEFAULT_CACHEGEN_KV_TOKENS_PER_PAGE = 16


class CacheGenKVPageFormatRegistry:
    """Registry of same-sized page formats and quantizer ownership.

    ``INT8_ADAPTIVE`` is the only currently implemented CacheGen page format.
    The others reserve stable identifiers for future codecs. Native vLLM
    storage, including vLLM's own FP8 cache mode, remains outside this slab.

    Metadata is bounded to one quarter of a page and rounded down to a
    four-byte boundary. At the default 4096-byte page geometry this yields a
    256-byte INT8 metadata region and leaves 1920 bytes each for K and V.
    """

    def __init__(
        self,
        *,
        page_nbytes: int = DEFAULT_CACHEGEN_KV_PAGE_NBYTES,
        tokens_per_page: int = DEFAULT_CACHEGEN_KV_TOKENS_PER_PAGE,
    ) -> None:
        if page_nbytes <= 0:
            raise ValueError("page_nbytes must be positive")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if page_nbytes < 16:
            raise ValueError("page_nbytes must be at least 16")

        self.page_nbytes = page_nbytes
        self.tokens_per_page = tokens_per_page
        self._formats = self._make_formats()

    def _make_formats(self) -> dict[CacheGenKVQuantizer, CacheGenKVPageFormat]:
        """Build page descriptors sharing one physical HBM page size."""
        metadata_nbytes = min(256, self.page_nbytes // 4)
        metadata_nbytes -= metadata_nbytes % 4

        payload_nbytes = self.page_nbytes - metadata_nbytes
        key_payload_nbytes = payload_nbytes // 2
        value_payload_nbytes = payload_nbytes - key_payload_nbytes

        return {
            CacheGenKVQuantizer.NATIVE: CacheGenKVPageFormat(
                quantizer=CacheGenKVQuantizer.NATIVE,
                page_nbytes=self.page_nbytes,
                tokens_per_page=self.tokens_per_page,
                metadata_nbytes=0,
                key_payload_nbytes=0,
                value_payload_nbytes=0,
                supports_fused_decode=False,
                implemented=False,
            ),
            CacheGenKVQuantizer.FP8: CacheGenKVPageFormat(
                quantizer=CacheGenKVQuantizer.FP8,
                page_nbytes=self.page_nbytes,
                tokens_per_page=self.tokens_per_page,
                metadata_nbytes=0,
                key_payload_nbytes=0,
                value_payload_nbytes=0,
                supports_fused_decode=False,
                implemented=False,
            ),
            CacheGenKVQuantizer.INT8_ADAPTIVE: CacheGenKVPageFormat(
                quantizer=CacheGenKVQuantizer.INT8_ADAPTIVE,
                page_nbytes=self.page_nbytes,
                tokens_per_page=self.tokens_per_page,
                metadata_nbytes=metadata_nbytes,
                key_payload_nbytes=key_payload_nbytes,
                value_payload_nbytes=value_payload_nbytes,
                supports_fused_decode=True,
                implemented=True,
            ),
            CacheGenKVQuantizer.KIVI: CacheGenKVPageFormat(
                quantizer=CacheGenKVQuantizer.KIVI,
                page_nbytes=self.page_nbytes,
                tokens_per_page=self.tokens_per_page,
                metadata_nbytes=0,
                key_payload_nbytes=0,
                value_payload_nbytes=0,
                supports_fused_decode=False,
                implemented=False,
            ),
            CacheGenKVQuantizer.TURBOQUANT: CacheGenKVPageFormat(
                quantizer=CacheGenKVQuantizer.TURBOQUANT,
                page_nbytes=self.page_nbytes,
                tokens_per_page=self.tokens_per_page,
                metadata_nbytes=0,
                key_payload_nbytes=0,
                value_payload_nbytes=0,
                supports_fused_decode=False,
                implemented=False,
            ),
            CacheGenKVQuantizer.INT4_GROUPWISE: CacheGenKVPageFormat(
                quantizer=CacheGenKVQuantizer.INT4_GROUPWISE,
                page_nbytes=self.page_nbytes,
                tokens_per_page=self.tokens_per_page,
                metadata_nbytes=0,
                key_payload_nbytes=0,
                value_payload_nbytes=0,
                supports_fused_decode=False,
                implemented=False,
            ),
        }

    @classmethod
    def from_geometry(
        cls,
        geometry: CacheGenKVPageGeometry,
    ) -> "CacheGenKVPageFormatRegistry":
        """Build a registry whose common page size fits adaptive INT8 K/V."""
        return cls(
            page_nbytes=geometry.common_page_nbytes,
            tokens_per_page=geometry.tokens_per_page,
        )

    @classmethod
    def from_geometry(
        cls,
        geometry: CacheGenKVPageGeometry,
    ) -> "CacheGenKVPageFormatRegistry":
        """Build a registry whose page size fits the given INT8 geometry."""
        return cls(
            page_nbytes=geometry.common_page_nbytes,
            tokens_per_page=geometry.tokens_per_page,
        )

    def get(
        self,
        quantizer: CacheGenKVQuantizer,
    ) -> CacheGenKVPageFormat:
        """Return one quantizer's fixed-byte page descriptor."""
        return self._formats[quantizer]

    def formats(self) -> dict[CacheGenKVQuantizer, CacheGenKVPageFormat]:
        """Return a shallow copy of all registered page descriptors."""
        return dict(self._formats)

    def implemented_formats(self) -> dict[
        CacheGenKVQuantizer,
        CacheGenKVPageFormat,
    ]:
        """Return only page formats with implemented CacheGen payloads."""
        return {
            quantizer: page_format
            for quantizer, page_format in self._formats.items()
            if page_format.implemented
        }
