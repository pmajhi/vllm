# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for experimental CacheGen INT8 fused decode attention."""

from __future__ import annotations

import os

_CACHEGEN_INT8_ATTENTION_ENV = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION"
)
_CACHEGEN_INT8_ATTENTION_LAYER_ENV = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION_LAYER"
)
_CACHEGEN_INT8_ATTENTION_MAX_PAGES_ENV = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION_MAX_PAGES"
)
_CACHEGEN_INT8_ATTENTION_MODE_ENV = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION_MODE"
)

_VALID_ATTENTION_MODES = frozenset(("validate", "int8", "common_page_int8"))


def _parse_bool(value: str) -> bool:
    """Parse a strict environment boolean."""
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        "Expected one of 1/0, true/false, yes/no, or on/off; "
        f"got {value!r}"
    )


def is_cachegen_int8_attention_enabled() -> bool:
    """Return whether experimental INT8 fused decode validation is enabled."""
    value = os.environ.get(_CACHEGEN_INT8_ATTENTION_ENV, "0")
    return _parse_bool(value)


def get_cachegen_int8_attention_layer() -> str | None:
    """Return the raw selected-layer selector, if explicitly configured."""
    value = os.environ.get(_CACHEGEN_INT8_ATTENTION_LAYER_ENV)
    if value is None:
        return None

    selector = value.strip()
    if not selector:
        raise ValueError(
            f"{_CACHEGEN_INT8_ATTENTION_LAYER_ENV} must not be empty"
        )
    return selector


def get_cachegen_int8_attention_max_pages() -> int | None:
    """Return the bounded compact-page budget when INT8 attention is enabled."""
    if not is_cachegen_int8_attention_enabled():
        return None

    value = os.environ.get(_CACHEGEN_INT8_ATTENTION_MAX_PAGES_ENV, "256")
    try:
        max_pages = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{_CACHEGEN_INT8_ATTENTION_MAX_PAGES_ENV} must be an integer; "
            f"got {value!r}"
        ) from exc

    if max_pages <= 0:
        raise ValueError(
            f"{_CACHEGEN_INT8_ATTENTION_MAX_PAGES_ENV} must be positive; "
            f"got {max_pages}"
        )
    return max_pages


def get_cachegen_int8_attention_mode() -> str:
    """Return the selected experimental INT8 attention output mode.

    ``validate`` keeps native FlashAttention authoritative and records
    native-vs-fused error. ``int8`` will be used by the next implementation
    step to make fused output authoritative only for eligible decode calls.
    """
    value = os.environ.get(_CACHEGEN_INT8_ATTENTION_MODE_ENV, "validate")
    mode = value.strip().lower()

    if mode not in _VALID_ATTENTION_MODES:
        valid = ", ".join(sorted(_VALID_ATTENTION_MODES))
        raise ValueError(
            f"{_CACHEGEN_INT8_ATTENTION_MODE_ENV} must be one of "
            f"{valid}; got {mode!r}"
        )
    return mode
