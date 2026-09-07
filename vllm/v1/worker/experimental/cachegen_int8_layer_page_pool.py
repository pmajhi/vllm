# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-isolated fixed-byte pools for CacheGen-style KV shadow storage."""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class CacheGenInt8LayerPagePool:
    """Layer-namespaced views over one contiguous fixed-byte page pool."""

    page_pool: torch.Tensor
    layer_names: tuple[str, ...]
    pages_per_layer: int
    layer_indices: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_indices",
            {layer_name: index for index, layer_name in enumerate(self.layer_names)},
        )

    @classmethod
    def allocate(
        cls,
        *,
        layer_names: list[str],
        pages_per_layer: int,
        page_bytes: int,
        device: torch.device,
    ) -> "CacheGenInt8LayerPagePool":
        if not layer_names:
            raise ValueError("layer_names must not be empty")
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("layer_names must be unique")
        if pages_per_layer <= 0:
            raise ValueError("pages_per_layer must be positive")
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")

        page_pool = torch.zeros(
            (len(layer_names) * pages_per_layer, page_bytes),
            dtype=torch.uint8,
            device=device,
        )
        return cls(
            page_pool=page_pool,
            layer_names=tuple(layer_names),
            pages_per_layer=pages_per_layer,
        )

    @property
    def num_layers(self) -> int:
        return len(self.layer_names)

    @property
    def page_bytes(self) -> int:
        return self.page_pool.shape[1]

    @property
    def total_pages(self) -> int:
        return self.page_pool.shape[0]

    def layer_index(self, layer_name: str) -> int:
        try:
            return self.layer_indices[layer_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown CacheGen shadow layer {layer_name!r}"
            ) from exc

    def layer_page_bounds(self, layer_name: str) -> tuple[int, int]:
        start = self.layer_index(layer_name) * self.pages_per_layer
        return start, start + self.pages_per_layer

    def layer_page_pool(self, layer_name: str) -> torch.Tensor:
        start, end = self.layer_page_bounds(layer_name)
        return self.page_pool[start:end]
