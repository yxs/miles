"""AVQA (r1aqa jsonl) -> miles jsonl conversion for the omni thinker audio-RL example."""

import importlib.util
import json
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

_REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = _REPO / "examples" / "omni_thinker" / "prepare_avqa.py"
    spec = importlib.util.spec_from_file_location("prepare_avqa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROW = {
    "id": 184,
    "question_text": "What is the main source of sound in the video?",
    "multi_choice": ["Car", "motorcycle", "siren", "gun fire"],
    "answer": 2,
    "dataset_name": "AVQA",
    "audio_path": "./Joysw909/AVQA/VGG10000/-HIPq7T3eFI_11.wav",
}


def test_convert_row_maps_answer_index_to_letter_and_audio_path():
    record = _load_tool().convert_row(ROW, audio_root="/data/avqa")

    assert record["label"] == "C"
    assert record["audios"] == ["/data/avqa/VGG10000/-HIPq7T3eFI_11.wav"]
    assert record["metadata"]["choices"] == ["Car", "motorcycle", "siren", "gun fire"]
    prompt = record["prompt"]
    assert prompt.startswith("<audio>")
    assert "A. Car" in prompt and "D. gun fire" in prompt
    assert record["prompt"].count("<audio>") == 1


def test_convert_row_reward_roundtrip_via_gpqa():
    from miles.rollout.rm_hub.gpqa import compute_gpqa_reward

    record = _load_tool().convert_row(ROW, audio_root="/data/avqa")

    assert compute_gpqa_reward("The answer is C.", record["label"], metadata=record["metadata"]) == 1.0
    assert compute_gpqa_reward("Answer: siren", record["label"], metadata=record["metadata"]) == 1.0
    assert compute_gpqa_reward("Answer: B", record["label"], metadata=record["metadata"]) == 0.0


def test_convert_file_writes_jsonl(tmp_path):
    src = tmp_path / "train_r1aqa_line.json"
    with open(src, "w") as f:
        f.write(json.dumps(ROW) + "\n")
        f.write(json.dumps({**ROW, "id": 185, "answer": 0}) + "\n")

    out = tmp_path / "avqa.jsonl"
    n = _load_tool().convert_file(src, out, audio_root="/data/avqa", max_samples=None)

    assert n == 2
    lines = [json.loads(line) for line in open(out)]
    assert lines[0]["label"] == "C"
    assert lines[1]["label"] == "A"


def test_convert_file_respects_max_samples(tmp_path):
    src = tmp_path / "train_r1aqa_line.json"
    with open(src, "w") as f:
        for i in range(5):
            f.write(json.dumps({**ROW, "id": i}) + "\n")

    out = tmp_path / "avqa.jsonl"
    n = _load_tool().convert_file(src, out, audio_root="/data/avqa", max_samples=3)

    assert n == 3
    assert sum(1 for _ in open(out)) == 3
