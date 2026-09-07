# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-byte page ABI for heterogeneous experimental KV quantizers."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


@dataclass(frozen=True)
class CacheGenKVPageFormat:
    """One quantizer's fixed-byte K/V page ABI.

    A physical page has the same total byte budget for every format. Individual
    quantizers choose how to partition that budget between quantized K/V
    payload and per-page metadata. The fused attention kernel dispatches from
    ``quantizer`` and consumes payload/metadata directly from GPU memory.
    """

    quantizer: CacheGenKVQuantizer
    page_nbytes: int
    tokens_per_page: int
    metadata_nbytes: int
    key_payload_nbytes: int
    value_payload_nbytes: int
    supports_fused_decode: bool
    implemented: bool

    def __post_init__(self) -> None:
        if self.page_nbytes <= 0:
            raise ValueError("page_nbytes must be positive")
        if self.tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if self.metadata_nbytes < 0:
            raise ValueError("metadata_nbytes must be non-negative")
        if self.key_payload_nbytes < 0:
            raise ValueError("key_payload_nbytes must be non-negative")
        if self.value_payload_nbytes < 0:
            raise ValueError("value_payload_nbytes must be non-negative")

        allocated = (
            self.metadata_nbytes
            + self.key_payload_nbytes
            + self.value_payload_nbytes
        )
        if allocated > self.page_nbytes:
            raise ValueError(
                "page payload and metadata exceed fixed page budget: "
                f"allocated={allocated}, page_nbytes={self.page_nbytes}"
            )

        if self.supports_fused_decode and not self.implemented:
            raise ValueError(
                "An unimplemented page format cannot support fused decode"
            )

    @property
    def unused_nbytes(self) -> int:
        """Return reserved bytes not currently consumed by the page ABI."""
        return self.page_nbytes - (
            self.metadata_nbytes
            + self.key_payload_nbytes
            + self.value_payload_nbytes
        )

    @property
    def metadata_offset(self) -> int:
        """Return the byte offset of the metadata region."""
        return 0

    @property
    def key_payload_offset(self) -> int:
        """Return the byte offset of quantized key payload."""
        return self.metadata_nbytes

    @property
    def value_payload_offset(self) -> int:
        """Return the byte offset of quantized value payload."""
        return self.metadata_nbytes + self.key_payload_nbytes
