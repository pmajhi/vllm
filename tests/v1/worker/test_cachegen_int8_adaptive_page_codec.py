# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_adaptive_page_codec import (
    CacheGenInt8AdaptivePageCodec,
    CacheGenInt8AdaptivePageLayout,
)
from vllm.v1.worker.experimental.cachegen_kv_page_format_registry import (
    CacheGenKVPageFormatRegistry,
)
from vllm.v1.worker.experimental.cachegen_kv_page_pool import (
    CacheGenKVPagePool,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


def make_codec_and_pool() -> tuple[
    CacheGenInt8AdaptivePageCodec,
    CacheGenKVPagePool,
]:
    layout = CacheGenInt8AdaptivePageLayout(
        tokens_per_page=4,
        num_kv_heads=2,
        head_size=8,
    )
    registry = CacheGenKVPageFormatRegistry(
        page_nbytes=256,
        tokens_per_page=layout.tokens_per_page,
    )
    pool = CacheGenKVPagePool(
        num_pages=2,
        page_nbytes=256,
        device=torch.device("cpu"),
        page_formats=registry,
    )
    codec = CacheGenInt8AdaptivePageCodec(
        page_format=registry.get(CacheGenKVQuantizer.INT8_ADAPTIVE),
        layout=layout,
    )
    return codec, pool


def test_round_trip_recovers_int8_quantized_page_values() -> None:
    codec, pool = make_codec_and_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None

    keys = torch.linspace(-1.0, 1.0, steps=64).reshape(4, 2, 8)
    values = torch.linspace(1.0, -1.0, steps=64).reshape(4, 2, 8)

    codec.write_page(
        pool=pool,
        handle=handle,
        keys=keys,
        values=values,
    )
    decoded_keys, decoded_values = codec.read_page(
        pool=pool,
        handle=handle,
        dtype=torch.float32,
    )

    key_error = (decoded_keys - keys).abs().max().item()
    value_error = (decoded_values - values).abs().max().item()

    assert key_error <= 1.0 / 127.0
    assert value_error <= 1.0 / 127.0


def test_write_clears_unused_page_bytes() -> None:
    codec, pool = make_codec_and_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None

    pool.page_bytes_view(handle).fill_(255)
    keys = torch.zeros((4, 2, 8))
    values = torch.zeros_like(keys)

    codec.write_page(
        pool=pool,
        handle=handle,
        keys=keys,
        values=values,
    )

    page = pool.page_bytes_view(handle)
    used_nbytes = (
        codec.page_format.metadata_nbytes
        + codec.layout.values_per_page
        + codec.layout.values_per_page
    )
    assert not torch.any(page[used_nbytes:])


def test_rejects_wrong_token_shape() -> None:
    codec, pool = make_codec_and_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None

    with pytest.raises(ValueError, match="keys must have shape"):
        codec.write_page(
            pool=pool,
            handle=handle,
            keys=torch.zeros((3, 2, 8)),
            values=torch.zeros((4, 2, 8)),
        )


def test_rejects_nonfloating_input() -> None:
    codec, pool = make_codec_and_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None

    keys = torch.zeros((4, 2, 8), dtype=torch.int8)
    values = torch.zeros((4, 2, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="floating point"):
        codec.write_page(
            pool=pool,
            handle=handle,
            keys=keys,
            values=values,
        )


def test_rejects_released_handle() -> None:
    codec, pool = make_codec_and_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None
    pool.release(handle.page_id)

    with pytest.raises(ValueError, match="not active"):
        codec.read_page(
            pool=pool,
            handle=handle,
            dtype=torch.float32,
        )


def test_requires_sufficient_page_payload() -> None:
    layout = CacheGenInt8AdaptivePageLayout(
        tokens_per_page=4,
        num_kv_heads=2,
        head_size=8,
    )
    registry = CacheGenKVPageFormatRegistry(
        page_nbytes=128,
        tokens_per_page=layout.tokens_per_page,
    )

    with pytest.raises(ValueError, match="payload region is too small"):
        CacheGenInt8AdaptivePageCodec(
            page_format=registry.get(CacheGenKVQuantizer.INT8_ADAPTIVE),
            layout=layout,
        )


def test_prefix_page_tracks_valid_tokens_and_zeroes_unused_payload() -> None:
    codec, pool = make_codec_and_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None

    keys = torch.ones((2, 2, 8))
    values = torch.full_like(keys, -0.5)

    codec.write_page(
        pool=pool,
        handle=handle,
        keys=keys,
        values=values,
        valid_tokens=2,
    )

    assert pool.get_valid_tokens(handle) == 2

    decoded_keys, decoded_values = codec.read_page(
        pool=pool,
        handle=handle,
        dtype=torch.float32,
    )
    assert torch.max(torch.abs(decoded_keys[:2] - keys)) <= 1.0 / 127.0
    assert torch.max(torch.abs(decoded_values[:2] - values)) <= 1.0 / 127.0
    assert not torch.any(decoded_keys[2:])
    assert not torch.any(decoded_values[2:])


def test_prefix_page_rejects_invalid_valid_token_count() -> None:
    codec, pool = make_codec_and_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None
    keys = torch.ones((1, 2, 8))

    with pytest.raises(ValueError, match="valid_tokens must be"):
        codec.write_page(
            pool=pool,
            handle=handle,
            keys=keys,
            values=keys,
            valid_tokens=0,
        )
