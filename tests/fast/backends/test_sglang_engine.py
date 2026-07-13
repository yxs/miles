from __future__ import annotations

import pytest

from miles.backends.sglang_utils.sglang_engine import (
    _extract_omni_distributed_weight_update_transport,
    _validate_omni_server_info,
)


def _model_info(**updates):
    info = {
        "success": True,
        "model_path": ("/root/.cache/huggingface/hub/models--bosonai--higgs-audio-v3-tts-4b/" "snapshots/revision"),
        "weight_version": "default",
        "stages": [{"stage": "tts_engine", "data": {"tp_size": 1}}],
    }
    info.update(updates)
    return info


def test_omni_external_server_accepts_matching_hf_snapshot() -> None:
    _validate_omni_server_info(
        _model_info(),
        {"model_path": "bosonai/higgs-audio-v3-tts-4b", "tp_size": 1},
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_path": "/models/other"}, "model mismatch"),
        ({"stages": [{"data": {"tp_size": 2}}]}, "TP mismatch"),
        ({"weight_version": None}, "no weight_version"),
        ({"success": False}, "model_info failed"),
    ],
)
def test_omni_external_server_rejects_incompatible_identity(updates, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        _validate_omni_server_info(
            _model_info(**updates),
            {"model_path": "bosonai/higgs-audio-v3-tts-4b", "tp_size": 1},
        )


def test_extract_omni_distributed_weight_update_transport() -> None:
    transport = {
        "protocol_version": 1,
        "backend": "nccl",
        "nccl_version": "2.28.9",
        "nccl_cumem_enable": "default",
    }
    model_info = _model_info(
        stages=[
            {"stage": "preprocessing", "data": {}},
            {
                "stage": "tts_engine",
                "data": {
                    "supports_distributed_weight_update": True,
                    "distributed_weight_update": transport,
                },
            },
        ]
    )

    assert _extract_omni_distributed_weight_update_transport(model_info) == transport


def test_extract_omni_distributed_weight_update_transport_requires_descriptor() -> None:
    with pytest.raises(RuntimeError, match="no distributed weight-update transport"):
        _extract_omni_distributed_weight_update_transport(_model_info())
