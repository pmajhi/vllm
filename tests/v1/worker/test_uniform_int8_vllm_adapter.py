# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.experimental.uniform_int8_byte_page import (
    UniformInt8BytePagePool,
)
from vllm.v1.worker.experimental.uniform_int8_vllm_adapter import (
    read_vllm_quantized_pages,
    write_vllm_quantized_pages,
)


def test_adapter_uses_vllm_heterogeneous_page_mapping() -> None:
    torch.manual_seed(2)

    block_table = BlockTable(
        block_size=16,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
    )

    # Both requests use the physical Uniform INT8 capacity. Different
    # capacities are valid only when the page codec/layout differs.
    block_table.add_row([0, 1, 2, 3], row_idx=0)
    block_table.add_row([4, 5, 6, 7], row_idx=1)

    request_indices = np.array([1, 0, 1, 0], dtype=np.int32)
    positions = np.array([83, 84, 84, 15], dtype=np.int32)
    tokens_per_page_by_request = np.array([84, 84], dtype=np.int32)

    page_ids_np, page_offsets_np = block_table.compute_quantized_mapping(
        request_indices,
        positions,
        tokens_per_page_by_request,
    )

    assert page_ids_np.tolist() == [4, 1, 5, 0]
    assert page_offsets_np.tolist() == [83, 0, 0, 15]

    page_ids = torch.from_numpy(page_ids_np)
    page_offsets = torch.from_numpy(page_offsets_np)

    pool = UniformInt8BytePagePool(
        num_pages=8,
        page_bytes=4096,
        num_kv_heads=2,
        head_size=8,
        device=torch.device("cpu"),
    )
    assert pool.layout.tokens_per_page >= 32

    keys = torch.randn(4, 2, 8)
    values = torch.randn(4, 2, 8)

    write_vllm_quantized_pages(
        pool=pool,
        quantized_page_ids=page_ids,
        quantized_page_offsets=page_offsets,
        keys=keys,
        values=values,
    )
    decoded_keys, decoded_values = read_vllm_quantized_pages(
        pool=pool,
        quantized_page_ids=page_ids,
        quantized_page_offsets=page_offsets,
        dtype=torch.float32,
    )

    assert torch.allclose(decoded_keys, keys, atol=0.03, rtol=0.03)
    assert torch.allclose(decoded_values, values, atol=0.03, rtol=0.03)



def test_adapter_rejects_offsets_outside_physical_page_capacity() -> None:
    pool = UniformInt8BytePagePool(
        num_pages=1,
        page_bytes=4096,
        num_kv_heads=2,
        head_size=8,
        device=torch.device("cpu"),
    )
    assert pool.layout.tokens_per_page == 84

    page_ids = torch.tensor([0], dtype=torch.int32)
    invalid_offsets = torch.tensor([84], dtype=torch.int32)
    keys = torch.zeros(1, 2, 8)
    values = torch.zeros(1, 2, 8)

    import pytest

    with pytest.raises(ValueError, match="exceed the Uniform INT8 page capacity"):
        write_vllm_quantized_pages(
            pool=pool,
            quantized_page_ids=page_ids,
            quantized_page_offsets=invalid_offsets,
            keys=keys,
            values=values,
        )

    with pytest.raises(ValueError, match="exceed the Uniform INT8 page capacity"):
        read_vllm_quantized_pages(
            pool=pool,
            quantized_page_ids=page_ids,
            quantized_page_offsets=invalid_offsets,
            dtype=torch.float32,
        )
