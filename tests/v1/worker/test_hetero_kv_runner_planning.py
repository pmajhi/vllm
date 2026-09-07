# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def make_input_batch() -> InputBatch:
    return InputBatch(
        max_num_reqs=4,
        max_model_len=256,
        max_num_batched_tokens=32,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=128,
        block_sizes=[16],
    )


def make_attention_config(
    geometries: list[tuple[int, int]],
) -> KVCacheConfig:
    groups = [
        SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(
                block_size=16,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.bfloat16,
                use_mla=False,
            )
        )
        for num_kv_heads, head_size in geometries
    ]
    return SimpleNamespace(kv_cache_groups=groups)


def make_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = make_input_batch()
    return runner


def test_runner_does_not_install_planning_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING",
        raising=False,
    )
    runner = make_runner()

    runner._configure_hetero_kv_page_planning(
        make_attention_config([(8, 128)]),
    )

    assert runner.input_batch._hetero_kv_tokens_per_page_by_codec is None


def test_runner_installs_model_geometry_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    runner = make_runner()

    runner._configure_hetero_kv_page_planning(
        make_attention_config([(8, 128)]),
    )

    assert runner.input_batch._hetero_kv_tokens_per_page_by_codec is not None
    assert runner.input_batch._tokens_per_page_for_request(
        quantizer_id=2,
        physical_page_bytes=16 * 4096,
    ) > 0


def test_runner_rejects_models_without_attention_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    runner = make_runner()

    with pytest.raises(ValueError, match="at least one decoder AttentionSpec"):
        runner._configure_hetero_kv_page_planning(
            SimpleNamespace(kv_cache_groups=[]),
        )


def test_runner_rejects_mismatched_attention_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    runner = make_runner()

    with pytest.raises(ValueError, match="share num_kv_heads and head_size"):
        runner._configure_hetero_kv_page_planning(
            make_attention_config([(8, 128), (16, 128)]),
        )
