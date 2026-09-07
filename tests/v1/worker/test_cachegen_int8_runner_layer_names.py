# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def make_group(
    layer_names: list[str],
    *,
    is_attention: bool = True,
) -> SimpleNamespace:
    if is_attention:
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            use_mla=False,
        )
    else:
        spec = object()
    return SimpleNamespace(
        layer_names=layer_names,
        kv_cache_spec=spec,
    )


def test_collects_decoder_attention_layers_in_group_order() -> None:
    config = SimpleNamespace(
        kv_cache_groups=[
            make_group(["layers.0.attn", "layers.1.attn"]),
            make_group(["layers.2.attn"]),
        ]
    )

    assert GPUModelRunner._get_cachegen_int8_decoder_layer_names(
        config
    ) == [
        "layers.0.attn",
        "layers.1.attn",
        "layers.2.attn",
    ]


def test_skips_non_attention_groups() -> None:
    config = SimpleNamespace(
        kv_cache_groups=[
            make_group(["layers.0.attn"]),
            make_group(["mamba.0"], is_attention=False),
        ]
    )

    assert GPUModelRunner._get_cachegen_int8_decoder_layer_names(
        config
    ) == ["layers.0.attn"]


def test_rejects_duplicate_attention_layer_name() -> None:
    config = SimpleNamespace(
        kv_cache_groups=[
            make_group(["layers.0.attn"]),
            make_group(["layers.0.attn"]),
        ]
    )

    with pytest.raises(ValueError, match="unique decoder"):
        GPUModelRunner._get_cachegen_int8_decoder_layer_names(config)
