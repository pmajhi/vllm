# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_adaptive_page_codec import (
    CacheGenInt8AdaptivePageCodec,
    CacheGenInt8AdaptivePageLayout,
)
from vllm.v1.worker.experimental.cachegen_int8_common_page_attention import (
    cachegen_int8_common_page_decode_attention,
)
from vllm.v1.worker.experimental.cachegen_int8_common_page_decode_triton import (
    cachegen_int8_common_page_decode_attention_triton,
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


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Common-page Triton decode requires CUDA.",
)


@pytest.mark.parametrize("sequence_length", [1, 4, 6, 8])
def test_common_page_triton_matches_byte_page_reference(
    sequence_length: int,
) -> None:
    device = torch.device("cuda")
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
        device=device,
        page_formats=registry,
    )
    codec = CacheGenInt8AdaptivePageCodec(
        page_format=registry.get(CacheGenKVQuantizer.INT8_ADAPTIVE),
        layout=layout,
    )

    handles = []
    for page_index in range(2):
        handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
        assert handle is not None
        keys = torch.linspace(
            -1.0 + page_index,
            1.0 + page_index,
            steps=64,
            device=device,
        ).reshape(4, 2, 8)
        values = torch.linspace(
            1.0 - page_index,
            -1.0 - page_index,
            steps=64,
            device=device,
        ).reshape(4, 2, 8)
        codec.write_page(
            pool=pool,
            handle=handle,
            keys=keys,
            values=values,
        )
        handles.append(handle)

    query = torch.linspace(
        -0.5,
        0.5,
        steps=32,
        device=device,
        dtype=torch.float32,
    ).reshape(4, 8)

    expected = cachegen_int8_common_page_decode_attention(
        query=query,
        pool=pool,
        codec=codec,
        page_handles=handles,
        sequence_length=sequence_length,
    )
    page_ids = torch.tensor(
        [handle.page_id for handle in handles],
        dtype=torch.int32,
        device=device,
    )
    page_format = registry.get(CacheGenKVQuantizer.INT8_ADAPTIVE)
    actual = cachegen_int8_common_page_decode_attention_triton(
        query=query,
        page_bytes=pool.page_bytes,
        key_scales=codec.key_scales_by_page,
        value_scales=codec.value_scales_by_page,
        page_ids=page_ids,
        sequence_length=sequence_length,
        num_kv_heads=layout.num_kv_heads,
        tokens_per_page=layout.tokens_per_page,
        head_size=layout.head_size,
        metadata_nbytes=page_format.metadata_nbytes,
        key_payload_offset=page_format.key_payload_offset,
        value_payload_offset=page_format.value_payload_offset,
    )

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
