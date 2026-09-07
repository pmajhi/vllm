# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference adapter from V1 quantized-page metadata to Uniform INT8 pages."""

import torch

from vllm.v1.worker.experimental.uniform_int8_byte_page import (
    UniformInt8BytePagePool,
)


def write_vllm_quantized_pages(
    *,
    pool: UniformInt8BytePagePool,
    quantized_page_ids: torch.Tensor,
    quantized_page_offsets: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Write K/V tokens using the page IDs and offsets emitted by V1."""
    if torch.any(quantized_page_offsets < 0):
        raise ValueError("quantized_page_offsets must be nonnegative")
    if torch.any(quantized_page_offsets >= pool.layout.tokens_per_page):
        raise ValueError(
            "quantized_page_offsets exceed the Uniform INT8 page capacity"
        )

    pool.write_batch(
        physical_page_ids=quantized_page_ids,
        page_offsets=quantized_page_offsets,
        keys=keys,
        values=values,
    )


def read_vllm_quantized_pages(
    *,
    pool: UniformInt8BytePagePool,
    quantized_page_ids: torch.Tensor,
    quantized_page_offsets: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read K/V tokens in the same packed order emitted by V1."""
    if torch.any(quantized_page_offsets < 0):
        raise ValueError("quantized_page_offsets must be nonnegative")
    if torch.any(quantized_page_offsets >= pool.layout.tokens_per_page):
        raise ValueError(
            "quantized_page_offsets exceed the Uniform INT8 page capacity"
        )

    return pool.read_batch(
        physical_page_ids=quantized_page_ids,
        page_offsets=quantized_page_offsets,
        dtype=dtype,
    )
