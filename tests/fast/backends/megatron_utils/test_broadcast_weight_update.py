from __future__ import annotations

import pytest
import torch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import broadcast


def test_broadcast_materializes_noncontiguous_converted_views(monkeypatch) -> None:
    base = torch.arange(12).reshape(3, 4)
    converted = [("weight", base.transpose(0, 1))]
    received: dict = {}

    class Engine:
        class Method:
            @staticmethod
            def remote(**kwargs):
                received.update(kwargs)
                return "ref"

        update_weights_from_distributed = Method()

    def fake_broadcast(tensor, src, *, group):
        assert tensor.is_contiguous()
        received["tensor"] = tensor

    monkeypatch.setattr(broadcast.dist, "broadcast", fake_broadcast)

    refs = broadcast.update_weights_from_distributed(
        "group",
        object(),
        3,
        [Engine()],
        converted,
    )

    assert refs == ["ref"]
    assert received["names"] == ["weight"]
    assert received["shapes"] == [torch.Size([4, 3])]
    assert received["tensor"].is_contiguous()
    assert torch.equal(received["tensor"], base.transpose(0, 1))


def test_disconnect_releases_custom_group_once(monkeypatch) -> None:
    updater = object.__new__(broadcast.UpdateWeightFromDistributed)
    updater.args = object()
    updater._group_name = "miles-pp_0"
    updater._model_update_groups = object()
    updater.rollout_engines = [object()]
    calls = []

    monkeypatch.setattr(
        broadcast.UpdateWeightFromDistributed,
        "_is_source",
        property(lambda _self: True),
    )
    monkeypatch.setattr(
        broadcast,
        "disconnect_rollout_engines_from_distributed",
        lambda *args: calls.append(args),
    )

    updater.disconnect_rollout_engines()
    updater.disconnect_rollout_engines()

    assert len(calls) == 1
    assert calls[0][1] == "miles-pp_0"
    assert updater._model_update_groups is None


def test_weight_update_transport_matches_trainer(monkeypatch) -> None:
    monkeypatch.delenv("NCCL_CUMEM_ENABLE", raising=False)
    monkeypatch.setattr(broadcast.torch.cuda.nccl, "version", lambda: (2, 28, 9))

    broadcast._validate_distributed_weight_update_transports(
        [
            {
                "protocol_version": 1,
                "backend": "nccl",
                "nccl_version": "2.28.9",
                "nccl_cumem_enable": "default",
            }
        ]
    )


def test_weight_update_transport_rejects_cumem_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "0")
    monkeypatch.setattr(broadcast.torch.cuda.nccl, "version", lambda: (2, 28, 9))

    with pytest.raises(RuntimeError, match="transport mismatch"):
        broadcast._validate_distributed_weight_update_transports(
            [
                {
                    "protocol_version": 1,
                    "backend": "nccl",
                    "nccl_version": "2.28.9",
                    "nccl_cumem_enable": "default",
                }
            ]
        )


def test_weight_update_transport_allows_legacy_engines() -> None:
    broadcast._validate_distributed_weight_update_transports([None, None])
