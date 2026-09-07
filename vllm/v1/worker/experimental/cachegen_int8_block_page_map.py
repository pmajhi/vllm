# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded mapping from native KV block IDs to compact INT8 page IDs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CacheGenInt8BlockPageMap:
    """Map live native physical KV block IDs into a bounded compact page pool.

    The map preserves a stable mapping while a native block remains active.
    It never evicts or silently reuses a compact page. If capacity is exhausted,
    `map_block_ids` returns None and the caller must use its native-attention
    fallback for that execution step.

    Native block lifecycle release/eviction support will be added before
    enabling long-running all-layer inference. The initial selected-layer route
    is deliberately bounded and fallback-safe.
    """

    max_pages: int
    _native_to_compact: dict[int, int] = field(default_factory=dict)
    _compact_to_native: list[int | None] = field(init=False)

    def __post_init__(self) -> None:
        if self.max_pages <= 0:
            raise ValueError(f"max_pages must be positive; got {self.max_pages}")
        self._compact_to_native = [None] * self.max_pages

    def reset(self) -> None:
        """Forget all compact mappings without changing configured capacity."""
        self._native_to_compact.clear()
        self._next_compact_page_id = 0

    @property
    def num_mapped_pages(self) -> int:
        return len(self._native_to_compact)

    @property
    def remaining_pages(self) -> int:
        return self.max_pages - self.num_mapped_pages

    def compact_page_id(self, native_block_id: int) -> int | None:
        """Return the compact page ID for a native block, if mapped."""
        return self._native_to_compact.get(native_block_id)

    def map_block_ids(
        self,
        native_block_ids: list[int],
    ) -> list[int] | None:
        """Map an active native block-table prefix into compact page IDs.

        The operation is atomic: if there are not enough free compact pages for
        all previously unseen IDs, it returns None without changing state.
        """
        if any(block_id < 0 for block_id in native_block_ids):
            raise ValueError("native_block_ids must be nonnegative")

        unseen_ids: list[int] = []
        seen_unmapped: set[int] = set()
        for block_id in native_block_ids:
            if (
                block_id not in self._native_to_compact
                and block_id not in seen_unmapped
            ):
                unseen_ids.append(block_id)
                seen_unmapped.add(block_id)

        if len(unseen_ids) > self.remaining_pages:
            return None

        for block_id in unseen_ids:
            compact_page_id = self._compact_to_native.index(None)
            self._native_to_compact[block_id] = compact_page_id
            self._compact_to_native[compact_page_id] = block_id

        return [
            self._native_to_compact[block_id]
            for block_id in native_block_ids
        ]

    def clear(self) -> None:
        """Forget all mappings and make every compact page available."""
        self._native_to_compact.clear()
        self._compact_to_native = [None] * self.max_pages
