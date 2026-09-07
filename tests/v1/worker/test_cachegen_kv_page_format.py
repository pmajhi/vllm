# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_kv_page_format import (
    CacheGenKVPageFormat,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


def make_format(**overrides: int | bool | CacheGenKVQuantizer) -> CacheGenKVPageFormat:
    values: dict[str, int | bool | CacheGenKVQuantizer] = {
        "quantizer": CacheGenKVQuantizer.INT8_ADAPTIVE,
        "page_nbytes": 4096,
        "tokens_per_page": 16,
        "metadata_nbytes": 256,
        "key_payload_nbytes": 1920,
        "value_payload_nbytes": 1920,
        "supports_fused_decode": True,
        "implemented": True,
    }
    values.update(overrides)
    return CacheGenKVPageFormat(**values)  # type: ignore[arg-type]


def test_fixed_byte_page_offsets_are_contiguous() -> None:
    page_format = make_format()

    assert page_format.metadata_offset == 0
    assert page_format.key_payload_offset == 256
    assert page_format.value_payload_offset == 2176
    assert page_format.unused_nbytes == 0


def test_page_format_can_reserve_unused_bytes() -> None:
    page_format = make_format(value_payload_nbytes=1800)

    assert page_format.unused_nbytes == 120


@pytest.mark.parametrize(
    "field",
    [
        "page_nbytes",
        "tokens_per_page",
    ],
)
def test_positive_layout_fields_are_required(field: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        make_format(**{field: 0})


@pytest.mark.parametrize(
    "field",
    [
        "metadata_nbytes",
        "key_payload_nbytes",
        "value_payload_nbytes",
    ],
)
def test_payload_fields_must_be_nonnegative(field: str) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        make_format(**{field: -1})


def test_payload_cannot_exceed_fixed_page_budget() -> None:
    with pytest.raises(ValueError, match="exceed fixed page budget"):
        make_format(key_payload_nbytes=2000, value_payload_nbytes=2000)


def test_unimplemented_format_cannot_claim_fused_decode() -> None:
    with pytest.raises(ValueError, match="unimplemented"):
        make_format(
            quantizer=CacheGenKVQuantizer.KIVI,
            implemented=False,
        )
