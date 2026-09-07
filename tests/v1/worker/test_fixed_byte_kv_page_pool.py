# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_fixed_byte_page_pool_has_common_page_extent() -> None:
    pool = GPUModelRunner.allocate_fixed_byte_kv_page_pool(
        num_pages=4,
        page_bytes=64 * 1024,
        device=torch.device("cpu"),
    )

    assert pool.shape == (4, 64 * 1024)
    assert pool.dtype is torch.uint8
    assert pool.device.type == "cpu"
    assert pool.numel() == 4 * 64 * 1024
    assert pool.element_size() == 1
    assert pool.nbytes == 4 * 64 * 1024
    assert torch.count_nonzero(pool) == 0


def test_fixed_byte_page_pool_pages_are_independent() -> None:
    pool = GPUModelRunner.allocate_fixed_byte_kv_page_pool(
        num_pages=3,
        page_bytes=32,
        device=torch.device("cpu"),
    )

    pool[1, 7] = 255

    assert pool[0, 7].item() == 0
    assert pool[1, 7].item() == 255
    assert pool[2, 7].item() == 0


@pytest.mark.parametrize(
    ("num_pages", "page_bytes", "message"),
    [
        (0, 64, "num_pages must be positive"),
        (-1, 64, "num_pages must be positive"),
        (1, 0, "page_bytes must be positive"),
        (1, -1, "page_bytes must be positive"),
    ],
)
def test_fixed_byte_page_pool_rejects_invalid_sizes(
    num_pages: int,
    page_bytes: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GPUModelRunner.allocate_fixed_byte_kv_page_pool(
            num_pages=num_pages,
            page_bytes=page_bytes,
            device=torch.device("cpu"),
        )
