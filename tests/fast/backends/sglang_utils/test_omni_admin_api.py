"""sglang-omni external-server admin contract: stages scoping + lifecycle no-ops.

The omni server accepts /init_weights_update_group, /update_weights_from_distributed and
/destroy_weights_update_group with an extra `stages` field (unset => the op fans out to
EVERY registered stage and breaks the NCCL world-size accounting), quiesces/flushes
internally during the update, and exposes no /flush_cache, /begin_weight_update or
/end_weight_update routes.
"""

from types import SimpleNamespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

pytest.importorskip("sglang")
from miles.backends.sglang_utils.sglang_engine import SGLangEngine


def _engine(monkeypatch, calls, **args_overrides):
    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234
    engine.args = SimpleNamespace(
        rollout_external_admin_api=args_overrides.pop("rollout_external_admin_api", "sglang"),
        rollout_weight_update_stages=args_overrides.pop("rollout_weight_update_stages", None),
        **args_overrides,
    )
    monkeypatch.setattr(
        SGLangEngine, "_make_request", lambda self, endpoint, payload=None: calls.append((endpoint, payload))
    )
    return engine


def _init_group(engine):
    return engine.init_weights_update_group("1.2.3.4", 29500, 1, 5, "miles-pp_0", "nccl")


def test_stages_forwarded_on_all_weight_group_requests(monkeypatch):
    calls = []
    engine = _engine(monkeypatch, calls, rollout_weight_update_stages=["thinker"])

    _init_group(engine)
    engine.update_weights_from_distributed(["w"], ["torch.bfloat16"], [[2, 2]], "miles-pp_0")
    engine.destroy_weights_update_group("miles-pp_0")

    by_endpoint = dict(calls)
    assert by_endpoint["init_weights_update_group"]["stages"] == ["thinker"]
    assert by_endpoint["update_weights_from_distributed"]["stages"] == ["thinker"]
    assert by_endpoint["destroy_weights_update_group"]["stages"] == ["thinker"]


def test_stages_absent_by_default(monkeypatch):
    calls = []
    engine = _engine(monkeypatch, calls)

    _init_group(engine)
    engine.update_weights_from_distributed(["w"], ["torch.bfloat16"], [[2, 2]], "miles-pp_0")
    engine.destroy_weights_update_group("miles-pp_0")

    for _, payload in calls:
        assert "stages" not in payload


def test_omni_admin_api_noops_missing_lifecycle_routes(monkeypatch):
    calls = []
    engine = _engine(monkeypatch, calls, rollout_external_admin_api="sglang-omni")
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("flush_cache must not hit HTTP"))
    )

    engine.flush_cache()
    engine.begin_weight_update()
    engine.end_weight_update()

    assert calls == [], f"omni admin api must not POST missing routes, got {calls}"


def test_omni_admin_api_delegates_flush_to_update_request(monkeypatch):
    calls = []
    engine = _engine(monkeypatch, calls, rollout_external_admin_api="sglang-omni")

    engine.update_weights_from_distributed(["w"], ["torch.bfloat16"], [[2, 2]], "miles-pp_0")

    [(endpoint, payload)] = calls
    assert endpoint == "update_weights_from_distributed"
    # the engine-level flush is a no-op under omni, so the server-internal one must run
    assert payload["flush_cache"] is True


def test_sglang_admin_api_keeps_default_lifecycle(monkeypatch):
    calls = []
    engine = _engine(monkeypatch, calls)

    engine.begin_weight_update()
    engine.end_weight_update()
    engine.update_weights_from_distributed(["w"], ["torch.bfloat16"], [[2, 2]], "miles-pp_0")

    endpoints = [e for e, _ in calls]
    assert endpoints == ["begin_weight_update", "end_weight_update", "update_weights_from_distributed"]
    assert dict(calls)["update_weights_from_distributed"]["flush_cache"] is False
