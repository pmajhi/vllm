import pytest
import torch

from vllm.v1.worker.experimental.int8_paged_kv_cache import Int8PagedKVCache


def test_int8_paged_kv_cache_writes_and_reads_multiple_pages() -> None:
    cache = Int8PagedKVCache(
        num_physical_pages=4,
        tokens_per_page=2,
        num_kv_heads=1,
        head_size=4,
        device=torch.device("cpu"),
    )

    key_a = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    value_a = torch.tensor([[4.0, 3.0, -2.0, -1.0]])
    key_b = torch.tensor([[-5.0, 6.0, -7.0, 8.0]])
    value_b = torch.tensor([[8.0, -7.0, 6.0, -5.0]])

    cache.write(physical_page_id=3, page_offset=1, key=key_a, value=value_a)
    cache.write(physical_page_id=1, page_offset=0, key=key_b, value=value_b)

    restored_key_a, restored_value_a = cache.read(3, 1, torch.float32)
    restored_key_b, restored_value_b = cache.read(1, 0, torch.float32)

    torch.testing.assert_close(restored_key_a, key_a, rtol=0.02, atol=0.02)
    torch.testing.assert_close(
        restored_value_a,
        value_a,
        rtol=0.02,
        atol=0.02,
    )
    torch.testing.assert_close(restored_key_b, key_b, rtol=0.02, atol=0.02)
    torch.testing.assert_close(
        restored_value_b,
        value_b,
        rtol=0.02,
        atol=0.02,
    )


def test_int8_paged_kv_cache_rejects_unknown_page() -> None:
    cache = Int8PagedKVCache(
        num_physical_pages=2,
        tokens_per_page=2,
        num_kv_heads=1,
        head_size=4,
        device=torch.device("cpu"),
    )
    key = torch.zeros(1, 4)
    value = torch.zeros(1, 4)

    with pytest.raises(IndexError, match="physical_page_id"):
        cache.write(physical_page_id=2, page_offset=0, key=key, value=value)

def test_int8_paged_kv_cache_write_batch_uses_page_ids_and_offsets() -> None:
    cache = Int8PagedKVCache(
        num_physical_pages=4,
        tokens_per_page=2,
        num_kv_heads=1,
        head_size=4,
        device=torch.device("cpu"),
    )

    page_ids = torch.tensor([3, 1, 3], dtype=torch.int32)
    page_offsets = torch.tensor([1, 0, 0], dtype=torch.int32)

    keys = torch.tensor(
        [
            [[1.0, -2.0, 3.0, -4.0]],
            [[-5.0, 6.0, -7.0, 8.0]],
            [[2.0, 4.0, -6.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [[4.0, 3.0, -2.0, -1.0]],
            [[8.0, -7.0, 6.0, -5.0]],
            [[-1.0, -3.0, 5.0, 7.0]],
        ],
        dtype=torch.float32,
    )

    cache.write_batch(page_ids, page_offsets, keys, values)

    for token_index in range(3):
        key, value = cache.read(
            int(page_ids[token_index]),
            int(page_offsets[token_index]),
            torch.float32,
        )
        torch.testing.assert_close(
            key,
            keys[token_index],
            rtol=0.02,
            atol=0.02,
        )
        torch.testing.assert_close(
            value,
            values[token_index],
            rtol=0.02,
            atol=0.02,
        )

def test_int8_paged_kv_cache_read_batch_preserves_packed_order() -> None:
    cache = Int8PagedKVCache(
        num_physical_pages=4,
        tokens_per_page=2,
        num_kv_heads=1,
        head_size=4,
        device=torch.device("cpu"),
    )

    page_ids = torch.tensor([3, 1, 3], dtype=torch.int32)
    page_offsets = torch.tensor([1, 0, 0], dtype=torch.int32)

    keys = torch.tensor(
        [
            [[1.0, -2.0, 3.0, -4.0]],
            [[-5.0, 6.0, -7.0, 8.0]],
            [[2.0, 4.0, -6.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [[4.0, 3.0, -2.0, -1.0]],
            [[8.0, -7.0, 6.0, -5.0]],
            [[-1.0, -3.0, 5.0, 7.0]],
        ],
        dtype=torch.float32,
    )

    cache.write_batch(page_ids, page_offsets, keys, values)
    restored_keys, restored_values = cache.read_batch(
        page_ids,
        page_offsets,
        torch.float32,
    )

    torch.testing.assert_close(restored_keys, keys, rtol=0.02, atol=0.02)
    torch.testing.assert_close(restored_values, values, rtol=0.02, atol=0.02)

def test_int8_paged_kv_cache_attention_is_close_to_fp32_reference() -> None:
    torch.manual_seed(0)

    cache = Int8PagedKVCache(
        num_physical_pages=4,
        tokens_per_page=2,
        num_kv_heads=2,
        head_size=8,
        device=torch.device("cpu"),
    )

    page_ids = torch.tensor([3, 1, 3, 0], dtype=torch.int32)
    page_offsets = torch.tensor([1, 0, 0, 1], dtype=torch.int32)

    query = torch.randn(2, 8, dtype=torch.float32)
    keys = torch.randn(4, 2, 8, dtype=torch.float32)
    values = torch.randn(4, 2, 8, dtype=torch.float32)

    cache.write_batch(page_ids, page_offsets, keys, values)
    int8_output = cache.attention(query, page_ids, page_offsets)

    scale = 1.0 / (query.shape[-1] ** 0.5)
    reference_scores = torch.einsum("hd,thd->ht", query, keys) * scale
    reference_weights = torch.softmax(reference_scores, dim=-1)
    reference_output = torch.einsum(
        "ht,thd->hd",
        reference_weights,
        values,
    )

    torch.testing.assert_close(
        int8_output,
        reference_output,
        rtol=0.05,
        atol=0.05,
    )

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for this test",
)
def test_int8_paged_kv_cache_attention_runs_on_cuda() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")

    cache = Int8PagedKVCache(
        num_physical_pages=4,
        tokens_per_page=2,
        num_kv_heads=2,
        head_size=8,
        device=device,
    )

    page_ids = torch.tensor(
        [3, 1, 3, 0],
        dtype=torch.int32,
        device=device,
    )
    page_offsets = torch.tensor(
        [1, 0, 0, 1],
        dtype=torch.int32,
        device=device,
    )
    query = torch.randn(2, 8, dtype=torch.float32, device=device)
    keys = torch.randn(4, 2, 8, dtype=torch.float32, device=device)
    values = torch.randn(4, 2, 8, dtype=torch.float32, device=device)

    cache.write_batch(page_ids, page_offsets, keys, values)
    int8_output = cache.attention(query, page_ids, page_offsets)

    scale = 1.0 / (query.shape[-1] ** 0.5)
    reference_scores = torch.einsum("hd,thd->ht", query, keys) * scale
    reference_weights = torch.softmax(reference_scores, dim=-1)
    reference_output = torch.einsum(
        "ht,thd->hd",
        reference_weights,
        values,
    )

    torch.testing.assert_close(
        int8_output,
        reference_output,
        rtol=0.05,
        atol=0.05,
    )
    assert int8_output.device.type == "cuda"