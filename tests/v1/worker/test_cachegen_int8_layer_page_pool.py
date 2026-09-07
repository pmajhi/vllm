# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_layer_page_pool import (
    CacheGenInt8LayerPagePool,
)


LAYERS = (
    "model.decoder.layers.0.self_attn.attn",
    "model.decoder.layers.1.self_attn.attn",
)


def make_pool() -> CacheGenInt8LayerPagePool:
    return CacheGenInt8LayerPagePool.allocate(
        layer_names=list(LAYERS),
        pages_per_layer=3,
        page_bytes=64,
        device=torch.device("cpu"),
    )


def test_layer_views_are_disjoint() -> None:
    pool = make_pool()

    first = pool.layer_page_pool(LAYERS[0])
    second = pool.layer_page_pool(LAYERS[1])

    assert pool.num_layers == 2
    assert pool.total_pages == 6
    assert pool.page_bytes == 64
    assert first.shape == (3, 64)
    assert second.shape == (3, 64)

    first[0, 0] = 17
    second[0, 0] = 29

    assert first[0, 0].item() == 17
    assert second[0, 0].item() == 29
    assert pool.page_pool[0, 0].item() == 17
    assert pool.page_pool[3, 0].item() == 29


def test_layer_index_and_unknown_layer() -> None:
    pool = make_pool()

    assert pool.layer_index(LAYERS[0]) == 0
    assert pool.layer_index(LAYERS[1]) == 1

    with pytest.raises(ValueError, match="Unknown CacheGen shadow layer"):
        pool.layer_page_pool("missing.layer")


@pytest.mark.parametrize(
    "layer_names,pages_per_layer,page_bytes,error",
    [
        ([], 1, 64, "layer_names"),
        ([LAYERS[0], LAYERS[0]], 1, 64, "unique"),
        ([LAYERS[0]], 0, 64, "pages_per_layer"),
        ([LAYERS[0]], 1, 0, "page_bytes"),
    ],
)
def test_invalid_allocation_arguments(
    layer_names: list[str],
    pages_per_layer: int,
    page_bytes: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        CacheGenInt8LayerPagePool.allocate(
            layer_names=layer_names,
            pages_per_layer=pages_per_layer,
            page_bytes=page_bytes,
            device=torch.device("cpu"),
        )
