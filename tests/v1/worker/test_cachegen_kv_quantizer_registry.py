# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.experimental.cachegen_kv_quantizer_registry import (
    CacheGenKVQuantizerRegistry,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


def test_int8_adaptive_is_the_only_authoritative_quantizer() -> None:
    registry = CacheGenKVQuantizerRegistry()

    assert registry.supports_authoritative_decode(
        CacheGenKVQuantizer.INT8_ADAPTIVE
    )

    for quantizer in (
        CacheGenKVQuantizer.NATIVE,
        CacheGenKVQuantizer.FP8,
        CacheGenKVQuantizer.KIVI,
        CacheGenKVQuantizer.TURBOQUANT,
        CacheGenKVQuantizer.INT4_GROUPWISE,
    ):
        assert not registry.supports_authoritative_decode(quantizer)


def test_int8_adaptive_uses_complete_cachegen_capabilities() -> None:
    capabilities = CacheGenKVQuantizerRegistry().capabilities(
        CacheGenKVQuantizer.INT8_ADAPTIVE
    )

    assert capabilities.supports_assignment
    assert capabilities.uses_cachegen_page_store
    assert capabilities.supports_fused_decode
    assert capabilities.supports_authoritative_decode


def test_native_and_fp8_are_assignable_without_cachegen_pages() -> None:
    registry = CacheGenKVQuantizerRegistry()

    for quantizer in (
        CacheGenKVQuantizer.NATIVE,
        CacheGenKVQuantizer.FP8,
    ):
        capabilities = registry.capabilities(quantizer)
        assert capabilities.supports_assignment
        assert not capabilities.uses_cachegen_page_store
        assert not capabilities.supports_fused_decode
        assert not capabilities.supports_authoritative_decode


def test_future_quantizers_are_reserved_until_implemented() -> None:
    registry = CacheGenKVQuantizerRegistry()

    for quantizer in (
        CacheGenKVQuantizer.KIVI,
        CacheGenKVQuantizer.TURBOQUANT,
        CacheGenKVQuantizer.INT4_GROUPWISE,
    ):
        capabilities = registry.capabilities(quantizer)
        assert not capabilities.supports_assignment
        assert not capabilities.uses_cachegen_page_store
        assert not capabilities.supports_fused_decode
        assert not capabilities.supports_authoritative_decode
