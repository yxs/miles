"""get_batch: variable-length audio processor tensors must pad-and-concat, not crash."""

from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

import miles.backends.training_utils.cp_utils as cp_utils_mod
import miles.backends.training_utils.data as data_mod
from miles.backends.training_utils.data import get_batch

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

MEL = 8


class _FakeIterator:
    def __init__(self, batch: dict):
        self._batch = batch
        self.rollout_data = {}

    def get_next(self, keys):
        return {key: self._batch[key] for key in keys if key in self._batch}


@pytest.fixture(autouse=True)
def _stub_cuda(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu", raising=False)
    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=1), tp=SimpleNamespace(size=1))
    monkeypatch.setattr(data_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda: state)


KEYS = ["tokens", "loss_masks", "total_lengths", "response_lengths", "multimodal_train_inputs"]


def _make_batch(mm_dicts: list[dict | None]) -> _FakeIterator:
    lengths = [6, 4][: len(mm_dicts)]
    return _FakeIterator(
        {
            "tokens": [torch.arange(1, n + 1, dtype=torch.long) for n in lengths],
            "loss_masks": [torch.ones(n // 2, dtype=torch.int) for n in lengths],
            "total_lengths": lengths,
            "response_lengths": [n // 2 for n in lengths],
            "multimodal_train_inputs": mm_dicts,
        }
    )


def test_variable_length_audio_features_padded_and_concatenated():
    a_feat = torch.randn(1, MEL, 50)
    a_mask = torch.ones(1, 50, dtype=torch.long)
    b_feat = torch.randn(2, MEL, 30)
    b_mask = torch.ones(2, 30, dtype=torch.long)
    b_mask[1, 20:] = 0
    iterator = _make_batch(
        [
            {"input_features": a_feat, "feature_attention_mask": a_mask},
            {"input_features": b_feat, "feature_attention_mask": b_mask},
        ]
    )

    batch = get_batch(iterator, KEYS, pad_multiplier=1, qkv_format="thd")

    mm = batch["multimodal_train_inputs"]
    assert mm["input_features"].shape == (3, MEL, 50)
    assert torch.equal(mm["input_features"][0], a_feat[0])
    assert torch.equal(mm["input_features"][1:, :, :30], b_feat)
    assert torch.all(mm["input_features"][1:, :, 30:] == 0)
    assert mm["feature_attention_mask"].shape == (3, 50)
    # per-audio frame counts survive the padding: this is what feature_lens is computed from
    assert mm["feature_attention_mask"].sum(dim=1).tolist() == [50, 30, 20]


def test_same_length_audio_and_image_keys_concat_plainly():
    mm_a = {
        "input_features": torch.randn(1, MEL, 30),
        "feature_attention_mask": torch.ones(1, 30, dtype=torch.long),
        "pixel_values": torch.randn(4, 10),
    }
    mm_b = {
        "input_features": torch.randn(1, MEL, 30),
        "feature_attention_mask": torch.ones(1, 30, dtype=torch.long),
        "pixel_values": torch.randn(2, 10),
    }
    iterator = _make_batch([mm_a, mm_b])

    batch = get_batch(iterator, KEYS, pad_multiplier=1, qkv_format="thd")

    mm = batch["multimodal_train_inputs"]
    assert mm["input_features"].shape == (2, MEL, 30)
    assert mm["pixel_values"].shape == (6, 10)
    assert torch.equal(mm["pixel_values"], torch.cat([mm_a["pixel_values"], mm_b["pixel_values"]], dim=0))


def test_none_mm_entries_are_skipped():
    feat = torch.randn(1, MEL, 20)
    mask = torch.ones(1, 20, dtype=torch.long)
    iterator = _make_batch([None, {"input_features": feat, "feature_attention_mask": mask}])

    batch = get_batch(iterator, KEYS, pad_multiplier=1, qkv_format="thd")

    mm = batch["multimodal_train_inputs"]
    assert mm["input_features"].shape == (1, MEL, 20)
    assert torch.equal(mm["input_features"], feat)
