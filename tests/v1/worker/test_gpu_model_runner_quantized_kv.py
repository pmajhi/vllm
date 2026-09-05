# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.worker.gpu_model_runner import (
    _require_baseline_quantizer_ids,
)


def test_baseline_quantizer_ids_are_accepted() -> None:
    _require_baseline_quantizer_ids(
        np.array([0, 0], dtype=np.int32)
    )


def test_nondefault_quantizer_id_is_rejected() -> None:
    with pytest.raises(
        NotImplementedError,
        match="Non-default quantizer_id",
    ):
        _require_baseline_quantizer_ids(
            np.array([0, 1], dtype=np.int32)
        )
