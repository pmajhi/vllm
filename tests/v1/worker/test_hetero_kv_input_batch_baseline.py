# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    HeteroKVCodecId,
)
from vllm.v1.quantized_kv_layout import (
    get_quantized_kv_page_layout,
)


BLOCK_SIZE = 16
PAGE_BYTES = 128 * 1024


def make_batch() -> InputBatch:
    return InputBatch(
        max_num_reqs=4,
        max_model_len=256,
        max_num_batched_tokens=32,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=32000,
        block_sizes=[BLOCK_SIZE],
    )


def test_baseline_tokens_per_page_needs_no_hetero_planning() -> None:
    batch = make_batch()

    actual = batch._tokens_per_page_for_request(
        quantizer_id=int(HeteroKVCodecId.BASELINE),
        physical_page_bytes=PAGE_BYTES,
    )

    baseline_layout = get_quantized_kv_page_layout(
        int(HeteroKVCodecId.BASELINE)
    )
    assert actual == PAGE_BYTES // baseline_layout.bytes_per_token


def test_nonbaseline_tokens_per_page_requires_hetero_planning() -> None:
    batch = make_batch()
    nonbaseline_codec = next(
        codec
        for codec in HeteroKVCodecId
        if codec is not HeteroKVCodecId.BASELINE
    )

    with pytest.raises(RuntimeError, match="has not been configured"):
        batch._tokens_per_page_for_request(
            quantizer_id=int(nonbaseline_codec),
            physical_page_bytes=PAGE_BYTES,
        )


def test_nonbaseline_tokens_per_page_uses_configured_geometry() -> None:
    batch = make_batch()
    batch.configure_hetero_kv_page_planning(
        num_kv_heads=2,
        head_size=64,
        page_bytes=PAGE_BYTES,
    )
    nonbaseline_codec = next(
        codec
        for codec in HeteroKVCodecId
        if codec is not HeteroKVCodecId.BASELINE
    )

    actual = batch._tokens_per_page_for_request(
        quantizer_id=int(nonbaseline_codec),
        physical_page_bytes=PAGE_BYTES,
    )

    assert actual > 0
