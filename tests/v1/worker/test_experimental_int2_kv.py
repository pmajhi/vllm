import pytest
import torch

from vllm.v1.worker.experimental.int2_kv import (
    dequantize_affine_int2,
    pack_int2,
    quantize_affine_int2,
    unpack_int2,
)


def test_pack_and_unpack_int2_round_trip() -> None:
    values = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0]],
        dtype=torch.uint8,
    )

    packed = pack_int2(values)

    assert packed.dtype is torch.uint8
    assert packed.shape == (1, 2)
    torch.testing.assert_close(unpack_int2(packed), values)


def test_affine_int2_round_trip_for_evenly_spaced_values() -> None:
    values = torch.tensor(
        [[[-2.0, 0.0, 2.0, 4.0]]],
        dtype=torch.float32,
    )

    packed, scale, minimum = quantize_affine_int2(values)
    restored = dequantize_affine_int2(
        packed,
        scale,
        minimum,
        torch.float32,
    )

    assert packed.shape == (1, 1, 1)
    assert scale.shape == (1, 1, 1)
    assert minimum.shape == (1, 1, 1)
    torch.testing.assert_close(restored, values)


def test_affine_int2_handles_constant_values() -> None:
    values = torch.full((2, 3, 4), 7.0, dtype=torch.float32)

    packed, scale, minimum = quantize_affine_int2(values)
    restored = dequantize_affine_int2(
        packed,
        scale,
        minimum,
        torch.float32,
    )

    torch.testing.assert_close(restored, values)
    torch.testing.assert_close(scale, torch.ones_like(scale))
    torch.testing.assert_close(minimum, torch.full_like(minimum, 7.0))


@pytest.mark.parametrize(
    ("values", "error_message"),
    [
        (torch.tensor([0, 1, 2, 4], dtype=torch.uint8), "range"),
        (torch.tensor([0, 1, 2], dtype=torch.uint8), "divisible by 4"),
    ],
)
def test_pack_int2_rejects_invalid_values(
    values: torch.Tensor,
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        pack_int2(values)


def test_quantize_affine_int2_rejects_non_float_values() -> None:
    values = torch.zeros((1, 4), dtype=torch.int32)

    with pytest.raises(ValueError, match="floating-point"):
        quantize_affine_int2(values)
