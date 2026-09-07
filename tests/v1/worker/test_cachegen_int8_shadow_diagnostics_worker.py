# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.gpu_worker import Worker


def test_worker_forwards_cachegen_shadow_diagnostics() -> None:
    worker = object.__new__(Worker)
    expected = {
        "callback_calls": 3,
        "tokens_written": 17,
    }
    worker.model_runner = SimpleNamespace(
        get_cachegen_int8_shadow_diagnostics=lambda: expected,
    )

    assert worker.get_cachegen_int8_shadow_diagnostics() == expected


def test_worker_forwards_disabled_cachegen_shadow_diagnostics() -> None:
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        get_cachegen_int8_shadow_diagnostics=lambda: None,
    )

    assert worker.get_cachegen_int8_shadow_diagnostics() is None


def test_worker_reports_attention_debug_state() -> None:
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.decoder.layers.0.self_attn": SimpleNamespace(
                    backend="FLASH_ATTN_VLLM_V1",
                    use_direct_call=False,
                ),
                "model.decoder.layers.0.fc1": SimpleNamespace(
                    backend="not-an-attention-layer",
                    use_direct_call=False,
                ),
            }
        )
    )

    assert worker.get_cachegen_int8_attention_debug_state() == [
        {
            "layer_name": "model.decoder.layers.0.self_attn",
            "backend": "FLASH_ATTN_VLLM_V1",
            "use_direct_call": False,
        }
    ]


def test_worker_forwards_layer_page_audit() -> None:
    worker = object.__new__(Worker)
    expected = {
        "layers.0.attn": {
            "page_id": 0,
            "page_offset": 1,
            "key_abs_max": 0.25,
            "value_abs_max": 0.5,
        }
    }
    worker.model_runner = SimpleNamespace(
        get_cachegen_int8_layer_page_audit=lambda: expected,
    )

    assert worker.get_cachegen_int8_layer_page_audit() == expected


def test_worker_forwards_cachegen_decode_route_debug_state() -> None:
    worker = object.__new__(Worker)
    expected = {
        "model.layers.0.self_attn.attn": {
            "num_actual_tokens": 1,
            "max_query_len": 1,
            "max_seq_len": 7,
            "seq_lens": [7],
            "query_start_loc": [0, 1],
            "block_table_shape": [1, 2],
            "active_block_table": [[3, 0]],
            "quantized_page_ids": [0],
            "quantized_page_offsets": [6],
        }
    }
    worker.model_runner = SimpleNamespace(
        get_cachegen_int8_decode_route_debug_state=lambda: expected,
    )

    assert worker.get_cachegen_int8_decode_route_debug_state() == expected
