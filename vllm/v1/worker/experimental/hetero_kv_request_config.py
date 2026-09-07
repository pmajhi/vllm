# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-level codec selection for experimental heterogeneous KV pages."""

from __future__ import annotations

import os

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    HeteroKVCodecId,
)
from vllm.v1.worker.experimental.hetero_kv_page_config import (
    is_hetero_kv_page_planning_enabled,
)


HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR = (
    "VLLM_EXPERIMENTAL_HETERO_KV_DEFAULT_QUANTIZER_ID"
)


def get_hetero_kv_default_quantizer_id() -> int:
    """Return the request codec ID.

    Baseline vLLM execution must always use codec 0 unless experimental
    heterogeneous KV-page planning is explicitly enabled. This guard prevents
    stale/default experimental codec settings from altering standard request
    handling when the feature is disabled.
    """
    if not is_hetero_kv_page_planning_enabled():
        return int(HeteroKVCodecId.BASELINE)

    configured_value = os.getenv(HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR)
    if configured_value is None:
        return int(HeteroKVCodecId.BASELINE)

    try:
        quantizer_id = int(configured_value)
    except ValueError as exc:
        raise ValueError(
            f"{HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR} must be an integer; "
            f"got {configured_value!r}"
        ) from exc

    try:
        HeteroKVCodecId(quantizer_id)
    except ValueError as exc:
        supported = ", ".join(str(int(codec)) for codec in HeteroKVCodecId)
        raise ValueError(
            f"{HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR} must be one of "
            f"{supported}; got {quantizer_id}"
        ) from exc

    return quantizer_id
