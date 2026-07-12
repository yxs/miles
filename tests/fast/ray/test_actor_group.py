from miles.ray.actor_group import _build_train_actor_env_vars


def test_train_actor_does_not_invent_nccl_cumem_setting(monkeypatch) -> None:
    monkeypatch.delenv("NCCL_CUMEM_ENABLE", raising=False)

    env_vars = _build_train_actor_env_vars({})

    assert "NCCL_CUMEM_ENABLE" not in env_vars


def test_train_actor_preserves_explicit_nccl_cumem_setting(monkeypatch) -> None:
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")

    env_vars = _build_train_actor_env_vars({})

    assert env_vars["NCCL_CUMEM_ENABLE"] == "1"


def test_train_env_vars_override_process_nccl_cumem_setting(monkeypatch) -> None:
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")

    env_vars = _build_train_actor_env_vars({"NCCL_CUMEM_ENABLE": "0"})

    assert env_vars["NCCL_CUMEM_ENABLE"] == "0"
