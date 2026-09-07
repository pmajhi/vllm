# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Codec-agnostic allocator for experimental heterogeneous fixed-byte KV pages."""

from dataclasses import dataclass, field


@dataclass
class HeteroFixedBytePagePool:
    """A deterministic shared page-ID allocator.

    The pool deliberately owns only page identity and accounting. Physical
    byte storage and codec-specific page interpretation remain separate.
    Freed IDs are reused in ascending order to make allocator behavior
    reproducible in tests and fragmentation experiments.
    """

    num_pages: int
    page_bytes: int
    _free_page_ids: list[int] = field(init=False, repr=False)
    _allocated_page_ids: set[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_pages <= 0:
            raise ValueError("num_pages must be positive")
        if self.page_bytes <= 0:
            raise ValueError("page_bytes must be positive")

        self._free_page_ids = list(range(self.num_pages))
        self._allocated_page_ids = set()

    @property
    def num_free_pages(self) -> int:
        return len(self._free_page_ids)

    @property
    def num_allocated_pages(self) -> int:
        return len(self._allocated_page_ids)

    @property
    def total_bytes(self) -> int:
        return self.num_pages * self.page_bytes

    @property
    def allocated_bytes(self) -> int:
        return self.num_allocated_pages * self.page_bytes

    @property
    def free_bytes(self) -> int:
        return self.num_free_pages * self.page_bytes

    def allocate(self) -> int:
        """Allocate one shared physical page ID."""
        if not self._free_page_ids:
            raise MemoryError(
                "Heterogeneous fixed-byte KV page pool is out of pages"
            )

        page_id = self._free_page_ids.pop(0)
        self._allocated_page_ids.add(page_id)
        return page_id

    def allocate_many(self, num_pages: int) -> list[int]:
        """Allocate ``num_pages`` IDs atomically with respect to exhaustion."""
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        if num_pages > self.num_free_pages:
            raise MemoryError(
                "Heterogeneous fixed-byte KV page pool is out of pages"
            )

        page_ids = self._free_page_ids[:num_pages]
        del self._free_page_ids[:num_pages]
        self._allocated_page_ids.update(page_ids)
        return page_ids

    def free(self, page_id: int) -> None:
        """Return one allocated physical page ID to the shared pool."""
        self._validate_page_id(page_id)
        if page_id not in self._allocated_page_ids:
            raise ValueError(
                f"Page ID {page_id} is not currently allocated"
            )

        self._allocated_page_ids.remove(page_id)
        self._free_page_ids.append(page_id)
        self._free_page_ids.sort()

    def free_many(self, page_ids: list[int]) -> None:
        """Return multiple allocated page IDs atomically."""
        if not page_ids:
            raise ValueError("page_ids must not be empty")
        if len(set(page_ids)) != len(page_ids):
            raise ValueError("page_ids must not contain duplicates")

        for page_id in page_ids:
            self._validate_page_id(page_id)
            if page_id not in self._allocated_page_ids:
                raise ValueError(
                    f"Page ID {page_id} is not currently allocated"
                )

        self._allocated_page_ids.difference_update(page_ids)
        self._free_page_ids.extend(page_ids)
        self._free_page_ids.sort()

    def is_allocated(self, page_id: int) -> bool:
        """Return whether a valid page ID is currently allocated."""
        self._validate_page_id(page_id)
        return page_id in self._allocated_page_ids

    def _validate_page_id(self, page_id: int) -> None:
        if not 0 <= page_id < self.num_pages:
            raise ValueError(
                f"Page ID {page_id} is outside [0, {self.num_pages})"
            )
