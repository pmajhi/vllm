import pytest

from vllm.v1.worker.experimental.cachegen_fixed_byte_pages import (
    CacheGenFixedBytePagePool,
    CacheGenIntegrityMode,
    CacheGenQuantizer,
)


@pytest.fixture
def pool():
    return CacheGenFixedBytePagePool(
        physical_page_bytes=65536,
        page_count=8,
        num_kv_heads=8,
        head_size=128,
        integrity_mode=CacheGenIntegrityMode.CRC32,
    )


def test_one_fixed_physical_page_size(pool):
    assert {layout.physical_page_bytes for layout in pool.layouts.values()} == {
        65536
    }


def test_lower_precision_fits_more_tokens(pool):
    assert (
        pool.layouts[CacheGenQuantizer.BF16].tokens_per_page
        < pool.layouts[CacheGenQuantizer.INT8].tokens_per_page
    )
    assert (
        pool.layouts[CacheGenQuantizer.INT8].tokens_per_page
        < pool.layouts[CacheGenQuantizer.INT4].tokens_per_page
    )


def test_one_free_page_can_be_reused_by_different_quantizer(pool):
    bf16 = pool.allocate(
        sequence_id=1,
        quantizer=CacheGenQuantizer.BF16,
        token_count=1,
    )
    assert bf16 is not None
    page_id = bf16.page_ids[0]

    pool.free(1)

    int4 = pool.allocate(
        sequence_id=2,
        quantizer=CacheGenQuantizer.INT4,
        token_count=1,
    )
    assert int4 is not None
    assert int4.page_ids[0] == page_id


def test_mixed_quantizers_share_one_free_list(pool):
    requests = [
        (1, CacheGenQuantizer.BF16, 17),
        (2, CacheGenQuantizer.FP8, 33),
        (3, CacheGenQuantizer.INT8, 33),
        (4, CacheGenQuantizer.INT4, 65),
    ]
    for sequence_id, quantizer, tokens in requests:
        assert pool.allocate(
            sequence_id=sequence_id,
            quantizer=quantizer,
            token_count=tokens,
        ) is not None

    stats = pool.stats()
    assert stats.active_sequences == 4
    assert stats.active_pages > 0
    assert stats.pages_by_quantizer["bf16"] > 0
    assert stats.pages_by_quantizer["fp8_e4m3"] > 0
    assert stats.pages_by_quantizer["int8"] > 0
    assert stats.pages_by_quantizer["int4"] > 0


def test_crc_detects_payload_corruption(pool):
    allocation = pool.allocate(
        sequence_id=1,
        quantizer=CacheGenQuantizer.INT8,
        token_count=1,
    )
    assert allocation is not None
    page_id = allocation.page_ids[0]

    pool.write_payload(page_id, b"cachegen payload")
    assert pool.verify_page(page_id)

    pool.inject_bit_flip(
        page_id=page_id,
        byte_offset_within_payload=0,
        bit=3,
    )
    assert not pool.verify_page(page_id)


def test_crc_reserves_only_tail_bytes(pool):
    no_crc = CacheGenFixedBytePagePool(
        physical_page_bytes=65536,
        page_count=1,
        num_kv_heads=8,
        head_size=128,
        integrity_mode=CacheGenIntegrityMode.OFF,
    )
    crc = CacheGenFixedBytePagePool(
        physical_page_bytes=65536,
        page_count=1,
        num_kv_heads=8,
        head_size=128,
        integrity_mode=CacheGenIntegrityMode.CRC32,
    )

    assert no_crc.layouts[CacheGenQuantizer.INT8].physical_page_bytes == (
        crc.layouts[CacheGenQuantizer.INT8].physical_page_bytes
    )
    assert crc.layouts[CacheGenQuantizer.INT8].integrity_bytes == 4
    assert crc.layouts[CacheGenQuantizer.INT8].tokens_per_page <= (
        no_crc.layouts[CacheGenQuantizer.INT8].tokens_per_page
    )
