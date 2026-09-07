import torch

from vllm.v1.worker.experimental.int8_kv import Int8KVPage
import math

import torch.nn.functional as F


def test_int8_kv_page_round_trip() -> None:
    torch.manual_seed(0)

    page = Int8KVPage(
        tokens_per_page=4,
        num_kv_heads=2,
        head_size=8,
        device=torch.device("cpu"),
    )

    key = torch.randn(2, 8, dtype=torch.float32)
    value = torch.randn(2, 8, dtype=torch.float32)

    page.write(page_offset=2, key=key, value=value)
    restored_key, restored_value = page.read(
        page_offset=2,
        dtype=torch.float32,
    )

    torch.testing.assert_close(restored_key, key, rtol=0.02, atol=0.02)
    torch.testing.assert_close(restored_value, value, rtol=0.02, atol=0.02)

import math

import torch.nn.functional as F

from vllm.v1.worker.experimental.int8_kv import (
    dequantize_symmetric_int8,
    quantize_symmetric_int8,
)


def test_int8_kv_attention_is_close_to_fp32_reference() -> None:
    torch.manual_seed(0)

    num_kv_heads = 2
    head_size = 16
    num_tokens = 7
    scale = 1.0 / math.sqrt(head_size)

    query = torch.randn(num_kv_heads, head_size, dtype=torch.float32)
    keys = torch.randn(
        num_tokens,
        num_kv_heads,
        head_size,
        dtype=torch.float32,
    )
    values = torch.randn(
        num_tokens,
        num_kv_heads,
        head_size,
        dtype=torch.float32,
    )

    reference_scores = torch.einsum("hd,thd->ht", query, keys) * scale
    reference_weights = F.softmax(reference_scores, dim=-1)
    reference_output = torch.einsum(
        "ht,thd->hd",
        reference_weights,
        values,
    )

    quantized_keys, key_scales = quantize_symmetric_int8(keys)
    quantized_values, value_scales = quantize_symmetric_int8(values)

    restored_keys = dequantize_symmetric_int8(
        quantized_keys,
        key_scales,
        torch.float32,
    )
    restored_values = dequantize_symmetric_int8(
        quantized_values,
        value_scales,
        torch.float32,
    )

    int8_scores = torch.einsum("hd,thd->ht", query, restored_keys) * scale
    int8_weights = F.softmax(int8_scores, dim=-1)
    int8_output = torch.einsum(
        "ht,thd->hd",
        int8_weights,
        restored_values,
    )

    torch.testing.assert_close(
        int8_output,
        reference_output,
        rtol=0.05,
        atol=0.05,
    )

def test_int8_kv_page_keeps_tokens_at_distinct_offsets() -> None:
    page = Int8KVPage(
        tokens_per_page=2,
        num_kv_heads=1,
        head_size=4,
        device=torch.device("cpu"),
    )

    key_at_zero = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0]],
        dtype=torch.float32,
    )
    value_at_zero = torch.tensor(
        [[4.0, 3.0, -2.0, -1.0]],
        dtype=torch.float32,
    )
    key_at_one = torch.tensor(
        [[-5.0, 6.0, -7.0, 8.0]],
        dtype=torch.float32,
    )
    value_at_one = torch.tensor(
        [[8.0, -7.0, 6.0, -5.0]],
        dtype=torch.float32,
    )

    page.write(0, key_at_zero, value_at_zero)
    page.write(1, key_at_one, value_at_one)

    restored_key_zero, restored_value_zero = page.read(0, torch.float32)
    restored_key_one, restored_value_one = page.read(1, torch.float32)

    torch.testing.assert_close(
        restored_key_zero,
        key_at_zero,
        rtol=0.02,
        atol=0.02,
    )
    torch.testing.assert_close(
        restored_value_zero,
        value_at_zero,
        rtol=0.02,
        atol=0.02,
    )
    torch.testing.assert_close(
        restored_key_one,
        key_at_one,
        rtol=0.02,
        atol=0.02,
    )
    torch.testing.assert_close(
        restored_value_one,
        value_at_one,
        rtol=0.02,
        atol=0.02,
    )