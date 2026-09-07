# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Registry and capabilities for experimental and native KV quantizers."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


@dataclass(frozen=True)
class CacheGenKVQuantizerCapabilities:
    """Capabilities and ownership of one KV quantizer selection."""

    supports_assignment: bool
    uses_cachegen_page_store: bool
    supports_fused_decode: bool
    supports_authoritative_decode: bool


class CacheGenKVQuantizerRegistry:
    """Static capability registry for native and experimental KV quantizers.

    ``native`` and ``fp8`` are vLLM-owned storage/attention selections.
    ``int8_adaptive`` is the currently implemented CacheGen page-store and
    fused-decode route. ``int4_groupwise`` remains reserved until implemented.
    """

    def __init__(self) -> None:
        self._capabilities = {
            CacheGenKVQuantizer.NATIVE: CacheGenKVQuantizerCapabilities(
                supports_assignment=True,
                uses_cachegen_page_store=False,
                supports_fused_decode=False,
                supports_authoritative_decode=False,
            ),
            CacheGenKVQuantizer.INT8_ADAPTIVE: (
                CacheGenKVQuantizerCapabilities(
                    supports_assignment=True,
                    uses_cachegen_page_store=True,
                    supports_fused_decode=True,
                    supports_authoritative_decode=True,
                )
            ),
            CacheGenKVQuantizer.FP8: CacheGenKVQuantizerCapabilities(
                supports_assignment=True,
                uses_cachegen_page_store=False,
                supports_fused_decode=False,
                supports_authoritative_decode=False,
            ),
            CacheGenKVQuantizer.KIVI: CacheGenKVQuantizerCapabilities(
                supports_assignment=False,
                uses_cachegen_page_store=False,
                supports_fused_decode=False,
                supports_authoritative_decode=False,
            ),
            CacheGenKVQuantizer.TURBOQUANT: (
                CacheGenKVQuantizerCapabilities(
                    supports_assignment=False,
                    uses_cachegen_page_store=False,
                    supports_fused_decode=False,
                    supports_authoritative_decode=False,
                )
            ),
            CacheGenKVQuantizer.INT4_GROUPWISE: (
                CacheGenKVQuantizerCapabilities(
                    supports_assignment=False,
                    uses_cachegen_page_store=False,
                    supports_fused_decode=False,
                    supports_authoritative_decode=False,
                )
            ),
        }

    def capabilities(
        self,
        quantizer: CacheGenKVQuantizer,
    ) -> CacheGenKVQuantizerCapabilities:
        """Return immutable capability metadata for one quantizer."""
        return self._capabilities[quantizer]

    def supports_assignment(
        self,
        quantizer: CacheGenKVQuantizer,
    ) -> bool:
        """Return whether one request may select this quantizer."""
        return self.capabilities(quantizer).supports_assignment

    def uses_cachegen_page_store(
        self,
        quantizer: CacheGenKVQuantizer,
    ) -> bool:
        """Return whether selection requires experimental CacheGen pages."""
        return self.capabilities(quantizer).uses_cachegen_page_store

    def supports_authoritative_decode(
        self,
        quantizer: CacheGenKVQuantizer,
    ) -> bool:
        """Return whether a fused CacheGen output may be authoritative."""
        return self.capabilities(quantizer).supports_authoritative_decode
