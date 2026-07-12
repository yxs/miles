from argparse import Namespace
from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.update_weight import common


def test_all_gather_param_skips_collective_for_tp1(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.arange(6).reshape(2, 3).float())
    parameter.tensor_model_parallel = True
    parameter.partition_dim = 0
    parameter.partition_stride = 1
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(size=1, group=object()),
        etp=SimpleNamespace(size=1, group=object()),
    )

    monkeypatch.setattr(common, "get_parallel_state", lambda: parallel_state)

    def unexpected_all_gather(*args, **kwargs):
        raise AssertionError("TP1 weight export must not enter a collective")

    monkeypatch.setattr(common.dist, "all_gather", unexpected_all_gather)

    gathered = common.all_gather_param(Namespace(), "module.weight", parameter)

    assert gathered.data_ptr() == parameter.data.data_ptr()
