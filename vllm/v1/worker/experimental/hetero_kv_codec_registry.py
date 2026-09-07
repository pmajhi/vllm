# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planning registry for experimental heterogeneous fixed-byte KV pages."""

from dataclasses import dataclass
from enum import IntEnum

from vllm.v1.worker.experimental.cachegen_int8_page_geometry import (
    make_cachegen_int8_page_geometry,
)
from vllm.v1.worker.experimental.kivi_page_geometry import (
    kivi_residual_bytes,
    make_kivi_page_geometry,
)
from vllm.v1.worker.experimental.turboquant_page_geometry import (
    make_turboquant_page_geometry,
    turboquant_ring_buffer_bytes,
)
from vllm.v1.worker.experimental.uniform_int8_codec import UniformInt8KVCodec


DEFAULT_HETERO_KV_PAGE_BYTES = 128 * 1024


class HeteroKVCodecId(IntEnum):
    """Stable IDs transported with experimental V1 requests."""

    BASELINE = 0
    UNIFORM_INT8 = 1
    CACHEGEN_INT8 = 2
    KIVI_K2_V2 = 3
    KIVI_K4_V4 = 4
    KIVI_K8_V8 = 5
    TURBOQUANT_K3_V2 = 6
    TURBOQUANT_K3_V4 = 7


@dataclass(frozen=True)
class HeteroKVCodecPlan:
    """Resolved storage plan for one codec and one attention-layer geometry."""

    codec_id: HeteroKVCodecId
    name: str
    page_bytes: int
    tokens_per_page: int
    external_bytes_per_sequence_per_layer: int
    key_bits: int | None = None
    value_bits: int | None = None
    key_group_size: int | None = None
    value_group_size: int | None = None
    residual_tokens: int = 0
    ring_capacity: int = 0

    def __post_init__(self) -> None:
        if self.page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        if self.tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if self.external_bytes_per_sequence_per_layer < 0:
            raise ValueError(
                "external_bytes_per_sequence_per_layer must be nonnegative"
            )


def supported_hetero_kv_codec_ids() -> tuple[HeteroKVCodecId, ...]:
    return tuple(HeteroKVCodecId)


def resolve_hetero_kv_codec_plan(
    *,
    codec_id: int | HeteroKVCodecId,
    page_bytes: int = DEFAULT_HETERO_KV_PAGE_BYTES,
    num_kv_heads: int,
    head_size: int,
) -> HeteroKVCodecPlan:
    """Resolve one codec into page capacity and external-state accounting."""
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_size <= 0:
        raise ValueError("head_size must be positive")

    try:
        resolved_id = HeteroKVCodecId(codec_id)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported hetero KV codec ID: {codec_id}. "
            f"Supported IDs: {[int(item) for item in HeteroKVCodecId]}"
        ) from exc

    if resolved_id is HeteroKVCodecId.BASELINE:
        # Baseline pages are only planning placeholders. Standard VLLM remains
        # responsible for their actual allocation and attention execution.
        bytes_per_token = 2 * num_kv_heads * head_size * 2
        tokens_per_page = page_bytes // bytes_per_token
        if tokens_per_page <= 0:
            raise ValueError(
                "page_bytes has insufficient usable bytes for one baseline "
                "BF16 K/V token"
            )
        return HeteroKVCodecPlan(
            codec_id=resolved_id,
            name="baseline_bf16",
            page_bytes=page_bytes,
            tokens_per_page=tokens_per_page,
            external_bytes_per_sequence_per_layer=0,
        )

    if resolved_id is HeteroKVCodecId.UNIFORM_INT8:
        geometry = UniformInt8KVCodec().page_geometry(
            physical_page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=None,  # The reference geometry does not consume dtype.
        )
        return HeteroKVCodecPlan(
            codec_id=resolved_id,
            name="uniform_int8",
            page_bytes=page_bytes,
            tokens_per_page=geometry.tokens_per_page,
            external_bytes_per_sequence_per_layer=0,
            key_bits=8,
            value_bits=8,
        )

    if resolved_id is HeteroKVCodecId.CACHEGEN_INT8:
        geometry = make_cachegen_int8_page_geometry(
            page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        return HeteroKVCodecPlan(
            codec_id=resolved_id,
            name=geometry.name,
            page_bytes=page_bytes,
            tokens_per_page=geometry.tokens_per_page,
            external_bytes_per_sequence_per_layer=0,
            key_bits=8,
            value_bits=8,
        )

    if resolved_id in (
        HeteroKVCodecId.KIVI_K2_V2,
        HeteroKVCodecId.KIVI_K4_V4,
        HeteroKVCodecId.KIVI_K8_V8,
    ):
        bit_width = {
            HeteroKVCodecId.KIVI_K2_V2: 2,
            HeteroKVCodecId.KIVI_K4_V4: 4,
            HeteroKVCodecId.KIVI_K8_V8: 8,
        }[resolved_id]
        residual_tokens = 32
        geometry = make_kivi_page_geometry(
            page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            key_bits=bit_width,
            value_bits=bit_width,
            key_group_size=32,
            value_group_size=32,
        )
        return HeteroKVCodecPlan(
            codec_id=resolved_id,
            name=geometry.name,
            page_bytes=page_bytes,
            tokens_per_page=geometry.token_capacity,
            external_bytes_per_sequence_per_layer=kivi_residual_bytes(
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                residual_tokens=residual_tokens,
            ),
            key_bits=bit_width,
            value_bits=bit_width,
            key_group_size=32,
            value_group_size=32,
            residual_tokens=residual_tokens,
        )

    if resolved_id in (
        HeteroKVCodecId.TURBOQUANT_K3_V2,
        HeteroKVCodecId.TURBOQUANT_K3_V4,
    ):
        value_bits = {
            HeteroKVCodecId.TURBOQUANT_K3_V2: 2,
            HeteroKVCodecId.TURBOQUANT_K3_V4: 4,
        }[resolved_id]
        ring_capacity = 128
        geometry = make_turboquant_page_geometry(
            page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            key_bits=3,
            value_bits=value_bits,
            value_group_size=32,
        )
        return HeteroKVCodecPlan(
            codec_id=resolved_id,
            name=geometry.name,
            page_bytes=page_bytes,
            tokens_per_page=geometry.tokens_per_page,
            external_bytes_per_sequence_per_layer=turboquant_ring_buffer_bytes(
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                ring_capacity=ring_capacity,
            ),
            key_bits=3,
            value_bits=value_bits,
            value_group_size=32,
            ring_capacity=ring_capacity,
        )

    raise AssertionError(f"Unhandled codec ID: {resolved_id}")
