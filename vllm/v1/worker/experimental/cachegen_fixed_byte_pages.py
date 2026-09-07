from __future__ import annotations

import binascii
from dataclasses import dataclass
from enum import Enum
from math import floor


class CacheGenQuantizer(str, Enum):
    BF16 = "bf16"
    FP8 = "fp8_e4m3"
    INT8 = "int8"
    INT4 = "int4"


class CacheGenIntegrityMode(str, Enum):
    OFF = "off"
    CRC32 = "crc32"


@dataclass(frozen=True)
class CacheGenPageFormat:
    quantizer: CacheGenQuantizer
    bits_per_element: int
    quantizer_metadata_bytes: int


@dataclass(frozen=True)
class CacheGenPageLayout:
    quantizer: CacheGenQuantizer
    physical_page_bytes: int
    header_bytes: int
    integrity_bytes: int
    quantizer_metadata_bytes: int
    payload_bytes: int
    bytes_per_token: int
    tokens_per_page: int
    tail_slack_bytes: int


@dataclass
class CacheGenPage:
    page_id: int
    data: bytearray | None = None
    quantizer: CacheGenQuantizer | None = None
    valid_tokens: int = 0
    sequence_id: int | None = None
    checksum: int | None = None


@dataclass(frozen=True)
class CacheGenPageAllocation:
    sequence_id: int
    quantizer: CacheGenQuantizer
    page_ids: tuple[int, ...]
    token_count: int
    allocated_token_capacity: int
    requested_payload_bytes: int
    allocated_page_bytes: int
    internal_fragmentation_bytes: int


@dataclass(frozen=True)
class CacheGenPoolStats:
    physical_page_bytes: int
    total_pages: int
    free_pages: int
    active_pages: int
    active_sequences: int
    allocation_failures: int
    allocated_page_bytes: int
    requested_payload_bytes: int
    internal_fragmentation_bytes: int
    tail_slack_bytes: int
    checksum_bytes_reserved: int
    pages_by_quantizer: dict[str, int]

    @property
    def internal_fragmentation_ratio(self) -> float:
        if not self.allocated_page_bytes:
            return 0.0
        return self.internal_fragmentation_bytes / self.allocated_page_bytes


DEFAULT_FORMATS = {
    CacheGenQuantizer.BF16: CacheGenPageFormat(
        quantizer=CacheGenQuantizer.BF16,
        bits_per_element=16,
        quantizer_metadata_bytes=0,
    ),
    CacheGenQuantizer.FP8: CacheGenPageFormat(
        quantizer=CacheGenQuantizer.FP8,
        bits_per_element=8,
        quantizer_metadata_bytes=32,
    ),
    CacheGenQuantizer.INT8: CacheGenPageFormat(
        quantizer=CacheGenQuantizer.INT8,
        bits_per_element=8,
        quantizer_metadata_bytes=32,
    ),
    CacheGenQuantizer.INT4: CacheGenPageFormat(
        quantizer=CacheGenQuantizer.INT4,
        bits_per_element=4,
        quantizer_metadata_bytes=48,
    ),
}


class CacheGenFixedBytePagePool:
    """One physical free list for BF16, FP8, INT8, and INT4 pages.

    Pages are fixed in physical bytes. Quantizer selection changes token
    capacity, payload format, and metadata, but not allocation unit size.
    """

    def __init__(
        self,
        *,
        physical_page_bytes: int,
        page_count: int,
        num_kv_heads: int,
        head_size: int,
        header_bytes: int = 32,
        integrity_mode: CacheGenIntegrityMode = CacheGenIntegrityMode.OFF,
        formats: dict[CacheGenQuantizer, CacheGenPageFormat] | None = None,
    ) -> None:
        if physical_page_bytes <= 0:
            raise ValueError("physical_page_bytes must be positive")
        if page_count <= 0:
            raise ValueError("page_count must be positive")
        if num_kv_heads <= 0 or head_size <= 0:
            raise ValueError("num_kv_heads and head_size must be positive")
        if header_bytes < 0:
            raise ValueError("header_bytes must be nonnegative")

        self.physical_page_bytes = physical_page_bytes
        self.page_count = page_count
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.header_bytes = header_bytes
        self.integrity_mode = integrity_mode
        self.integrity_bytes = (
            4 if integrity_mode is CacheGenIntegrityMode.CRC32 else 0
        )
        self.formats = formats or DEFAULT_FORMATS

        self.layouts = {
            quantizer: self._make_layout(page_format)
            for quantizer, page_format in self.formats.items()
        }
        self.pages = [
            CacheGenPage(page_id=page_id)
            for page_id in range(page_count)
        ]
        self.free_page_ids = list(range(page_count))
        self.active_allocations: dict[int, CacheGenPageAllocation] = {}
        self.allocation_failures = 0

    def _make_layout(
        self,
        page_format: CacheGenPageFormat,
    ) -> CacheGenPageLayout:
        bytes_per_token_numerator = (
            2
            * self.num_kv_heads
            * self.head_size
            * page_format.bits_per_element
        )
        if bytes_per_token_numerator % 8:
            raise ValueError("bits-per-token must be byte aligned")

        bytes_per_token = bytes_per_token_numerator // 8
        payload_bytes = (
            self.physical_page_bytes
            - self.header_bytes
            - self.integrity_bytes
            - page_format.quantizer_metadata_bytes
        )
        tokens_per_page = floor(payload_bytes / bytes_per_token)
        if tokens_per_page <= 0:
            raise ValueError(
                f"page too small for {page_format.quantizer.value}"
            )

        used_payload_bytes = tokens_per_page * bytes_per_token
        return CacheGenPageLayout(
            quantizer=page_format.quantizer,
            physical_page_bytes=self.physical_page_bytes,
            header_bytes=self.header_bytes,
            integrity_bytes=self.integrity_bytes,
            quantizer_metadata_bytes=page_format.quantizer_metadata_bytes,
            payload_bytes=used_payload_bytes,
            bytes_per_token=bytes_per_token,
            tokens_per_page=tokens_per_page,
            tail_slack_bytes=payload_bytes - used_payload_bytes,
        )

    def allocate(
        self,
        *,
        sequence_id: int,
        quantizer: CacheGenQuantizer,
        token_count: int,
    ) -> CacheGenPageAllocation | None:
        if sequence_id in self.active_allocations:
            raise ValueError(f"sequence {sequence_id} is already active")
        if token_count <= 0:
            raise ValueError("token_count must be positive")

        layout = self.layouts[quantizer]
        pages_needed = (
            token_count + layout.tokens_per_page - 1
        ) // layout.tokens_per_page
        if len(self.free_page_ids) < pages_needed:
            self.allocation_failures += 1
            return None

        page_ids = tuple(
            self.free_page_ids.pop() for _ in range(pages_needed)
        )
        remaining = token_count
        for page_id in page_ids:
            page = self.pages[page_id]
            page.quantizer = quantizer
            page.sequence_id = sequence_id
            page.valid_tokens = min(remaining, layout.tokens_per_page)
            page.checksum = None
            remaining -= page.valid_tokens

        requested_payload_bytes = token_count * layout.bytes_per_token
        allocated_page_bytes = pages_needed * self.physical_page_bytes
        internal_fragmentation_bytes = (
            allocated_page_bytes
            - pages_needed
            * (
                self.header_bytes
                + self.integrity_bytes
                + layout.quantizer_metadata_bytes
            )
            - requested_payload_bytes
        )

        allocation = CacheGenPageAllocation(
            sequence_id=sequence_id,
            quantizer=quantizer,
            page_ids=page_ids,
            token_count=token_count,
            allocated_token_capacity=pages_needed * layout.tokens_per_page,
            requested_payload_bytes=requested_payload_bytes,
            allocated_page_bytes=allocated_page_bytes,
            internal_fragmentation_bytes=internal_fragmentation_bytes,
        )
        self.active_allocations[sequence_id] = allocation
        return allocation

    def free(self, sequence_id: int) -> None:
        allocation = self.active_allocations.pop(sequence_id)
        for page_id in allocation.page_ids:
            page = self.pages[page_id]
            page.data = None
            page.quantizer = None
            page.valid_tokens = 0
            page.sequence_id = None
            page.checksum = None
            self.free_page_ids.append(page_id)

    def payload_slice(self, page_id: int) -> slice:
        page = self.pages[page_id]
        if page.quantizer is None:
            raise ValueError("page is free")
        layout = self.layouts[page.quantizer]
        start = (
            self.header_bytes + layout.quantizer_metadata_bytes
        )
        return slice(start, start + layout.payload_bytes)

    def write_payload(self, page_id: int, payload: bytes) -> None:
        page = self.pages[page_id]
        if page.data is None:
            page.data = bytearray(self.physical_page_bytes)
        payload_slice = self.payload_slice(page_id)
        if len(payload) > payload_slice.stop - payload_slice.start:
            raise ValueError("payload does not fit page")
        page.data[payload_slice.start:payload_slice.start + len(payload)] = payload
        if self.integrity_mode is CacheGenIntegrityMode.CRC32:
            page.checksum = binascii.crc32(
                page.data[payload_slice]
            ) & 0xFFFFFFFF
            checksum_start = self.physical_page_bytes - 4
            page.data[checksum_start:] = page.checksum.to_bytes(
                4,
                byteorder="little",
                signed=False,
            )

    def verify_page(self, page_id: int) -> bool:
        page = self.pages[page_id]
        if page.quantizer is None:
            raise ValueError("page is free")
        if self.integrity_mode is CacheGenIntegrityMode.OFF:
            return True
        assert page.data is not None
        assert page.checksum is not None
        payload_slice = self.payload_slice(page_id)
        observed = binascii.crc32(page.data[payload_slice]) & 0xFFFFFFFF
        stored = int.from_bytes(
            page.data[self.physical_page_bytes - 4:],
            byteorder="little",
            signed=False,
        )
        return observed == page.checksum == stored

    def inject_bit_flip(
        self,
        *,
        page_id: int,
        byte_offset_within_payload: int,
        bit: int = 0,
    ) -> None:
        if not 0 <= bit < 8:
            raise ValueError("bit must be in [0, 7]")
        page = self.pages[page_id]
        if page.data is None:
            raise ValueError("page has no written payload")
        payload_slice = self.payload_slice(page_id)
        absolute_offset = payload_slice.start + byte_offset_within_payload
        if absolute_offset >= payload_slice.stop:
            raise ValueError("payload offset is outside the page")
        self.pages[page_id].data[absolute_offset] ^= 1 << bit

    def stats(self) -> CacheGenPoolStats:
        active_pages = [
            page for page in self.pages if page.quantizer is not None
        ]
        active_allocations = list(self.active_allocations.values())
        pages_by_quantizer = {
            quantizer.value: sum(
                1
                for page in active_pages
                if page.quantizer is quantizer
            )
            for quantizer in CacheGenQuantizer
        }
        return CacheGenPoolStats(
            physical_page_bytes=self.physical_page_bytes,
            total_pages=self.page_count,
            free_pages=len(self.free_page_ids),
            active_pages=len(active_pages),
            active_sequences=len(active_allocations),
            allocation_failures=self.allocation_failures,
            allocated_page_bytes=sum(
                allocation.allocated_page_bytes
                for allocation in active_allocations
            ),
            requested_payload_bytes=sum(
                allocation.requested_payload_bytes
                for allocation in active_allocations
            ),
            internal_fragmentation_bytes=sum(
                allocation.internal_fragmentation_bytes
                for allocation in active_allocations
            ),
            tail_slack_bytes=sum(
                self.layouts[page.quantizer].tail_slack_bytes
                for page in active_pages
                if page.quantizer is not None
            ),
            checksum_bytes_reserved=(
                len(active_pages) * self.integrity_bytes
            ),
            pages_by_quantizer=pages_by_quantizer,
        )
