# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.quantized_kv_layout import (
    get_quantized_kv_page_layout,
    tokens_per_page_for_quantizer,
)
from vllm.v1.worker.gpu_input_batch import InputBatch


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


def baseline_physical_page_bytes(input_batch: InputBatch) -> int:
    baseline_layout = get_quantized_kv_page_layout(0)
    return input_batch.block_table[0].block_size * baseline_layout.bytes_per_token


def test_nonbaseline_request_requires_installed_model_geometry() -> None:
    input_batch = make_input_batch()

    with pytest.raises(
        RuntimeError,
        match="has not been configured with model attention geometry",
    ):
        input_batch._tokens_per_page_for_request(
            quantizer_id=2,
            physical_page_bytes=baseline_physical_page_bytes(input_batch),
        )


def test_installed_registry_capacities_use_configurable_page_size() -> None:
    input_batch = make_input_batch()
    input_batch.configure_hetero_kv_page_planning(
        num_kv_heads=8,
        head_size=128,
        page_bytes=128 * 1024,
    )

    capacities = {
        codec_id: input_batch._tokens_per_page_for_request(
            quantizer_id=codec_id,
            physical_page_bytes=baseline_physical_page_bytes(input_batch),
        )
        for codec_id in (2, 3, 6)
    }

    assert all(capacity > 0 for capacity in capacities.values())
    assert len(set(capacities.values())) == len(capacities)


def test_baseline_capacity_remains_legacy_policy() -> None:
    input_batch = make_input_batch()
    physical_page_bytes = baseline_physical_page_bytes(input_batch)

    assert input_batch._tokens_per_page_for_request(
        quantizer_id=0,
        physical_page_bytes=physical_page_bytes,
    ) == tokens_per_page_for_quantizer(
        0,
        physical_page_bytes,
    )


def test_unknown_codec_is_rejected_after_configuration() -> None:
    input_batch = make_input_batch()
    input_batch.configure_hetero_kv_page_planning(
        num_kv_heads=8,
        head_size=128,
        page_bytes=128 * 1024,
    )

    with pytest.raises(ValueError, match="Unsupported heterogeneous KV codec ID"):
        input_batch._tokens_per_page_for_request(
            quantizer_id=99,
            physical_page_bytes=baseline_physical_page_bytes(input_batch),
        )


def test_page_size_environment_setting_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_BYTES",
        str(128 * 1024),
    )
    input_batch = make_input_batch()
    input_batch.configure_hetero_kv_page_planning(
        num_kv_heads=8,
        head_size=128,
    )

    capacity_from_environment = input_batch._tokens_per_page_for_request(
        quantizer_id=2,
        physical_page_bytes=baseline_physical_page_bytes(input_batch),
    )

    input_batch_explicit = make_input_batch()
    input_batch_explicit.configure_hetero_kv_page_planning(
        num_kv_heads=8,
        head_size=128,
        page_bytes=128 * 1024,
    )
    capacity_explicit = input_batch_explicit._tokens_per_page_for_request(
        quantizer_id=2,
        physical_page_bytes=baseline_physical_page_bytes(input_batch_explicit),
    )

    assert capacity_from_environment == capacity_explicit


def test_too_small_page_size_rejects_full_codec_catalog() -> None:
    input_batch = make_input_batch()

    with pytest.raises(
        ValueError,
        match="insufficient usable bytes for one complete KIVI key group",
    ):
        input_batch.configure_hetero_kv_page_planning(
            num_kv_heads=8,
            head_size=128,
            page_bytes=64 * 1024,
        )
