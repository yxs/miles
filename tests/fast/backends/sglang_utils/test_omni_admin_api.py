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
    engine.rank = 0
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


def test_omni_admin_api_external_init_skips_server_args_check(monkeypatch):
    import miles.backends.sglang_utils.sglang_engine as engine_mod

    calls = []
    engine = _engine(monkeypatch, calls, rollout_external_admin_api="sglang-omni")
    monkeypatch.setattr(engine_mod, "_wait_server_healthy", lambda **kwargs: calls.append("health"))
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("omni server has no /get_server_info")),
    )

    engine._init_external({"host": "fake-host", "port": 1234}, external_engine_need_check_fields=["port"])

    assert calls == ["health"]


def test_omni_admin_api_external_init_probes_health_only(monkeypatch):
    # the omni server exposes /health but neither /health_generate nor /flush_cache;
    # probing the sglang defaults would retry 404s forever and hang the attach
    import miles.backends.sglang_utils.sglang_engine as engine_mod

    captured = {}

    def fake_wait(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(engine_mod, "_wait_server_healthy", fake_wait)

    calls = []
    omni = _engine(monkeypatch, calls, rollout_external_admin_api="sglang-omni")
    omni._init_external({"host": "fake-host", "port": 1234}, external_engine_need_check_fields=[])
    assert captured["probe_paths"] == ("health",)

    monkeypatch.setattr(SGLangEngine, "_make_request", lambda self, endpoint, payload=None: None)
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: type("R", (), {"status_code": 200, "json": dict, "raise_for_status": lambda self: None})(),
    )
    default = _engine(monkeypatch, [], rollout_external_admin_api="sglang")
    default._init_external({"host": "fake-host", "port": 1234}, external_engine_need_check_fields=[])
    assert captured["probe_paths"] == ("health_generate", "flush_cache")


def test_wait_server_healthy_hits_only_requested_probe_paths(monkeypatch):
    from miles.backends.sglang_utils.sglang_engine import _wait_server_healthy

    seen = []

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None):
            seen.append(url)
            return type("R", (), {"status_code": 200})()

    monkeypatch.setattr("requests.Session", _FakeSession)

    _wait_server_healthy(
        base_url="http://x:1", api_key=None, is_process_alive=lambda: True, probe_paths=("health",)
    )

    assert seen == ["http://x:1/health"]


def test_sglang_admin_api_keeps_default_lifecycle(monkeypatch):
    calls = []
    engine = _engine(monkeypatch, calls)

    engine.begin_weight_update()
    engine.end_weight_update()
    engine.update_weights_from_distributed(["w"], ["torch.bfloat16"], [[2, 2]], "miles-pp_0")

    endpoints = [e for e, _ in calls]
    assert endpoints == ["begin_weight_update", "end_weight_update", "update_weights_from_distributed"]
    assert dict(calls)["update_weights_from_distributed"]["flush_cache"] is False
