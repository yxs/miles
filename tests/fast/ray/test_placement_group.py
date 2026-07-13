from types import SimpleNamespace

from miles.ray import placement_group


def test_external_rollout_reserves_only_training_gpus(monkeypatch) -> None:
    requested = []
    fake_pg = object()

    def fake_create(num_gpus):
        requested.append(num_gpus)
        return fake_pg, [0], [6]

    monkeypatch.setattr(placement_group, "_create_placement_group", fake_create)
    args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        rollout_external=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        rollout_num_gpus=1,
        use_critic=False,
    )

    groups = placement_group.create_placement_groups(args)

    assert requested == [1]
    assert groups["actor"] == (fake_pg, [0], [6])
    assert groups["rollout"] == (fake_pg, [], [])
