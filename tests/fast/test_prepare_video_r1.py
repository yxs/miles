"""Video-R1-260k (image+video MCQ) -> miles jsonl conversion for the VL thinker example."""

import importlib.util
import json
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

_REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = _REPO / "examples" / "omni_thinker" / "prepare_video_r1.py"
    spec = importlib.util.spec_from_file_location("prepare_video_r1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VIDEO_ROW = {
    "problem_id": 2,
    "problem": "What appears on the screen?",
    "data_type": "video",
    "problem_type": "multiple choice",
    "options": ["A. A notification", "B. A command", "C. A warning", "D. An update"],
    "solution": "<answer>A</answer>",
    "path": "./CLEVRER/video_validation/video_10000.mp4",
    "data_source": "",
}
IMAGE_ROW = {
    "problem_id": 1,
    "problem": "What role does the cylinder play?",
    "data_type": "image",
    "problem_type": "multiple choice",
    "options": ["A) focus", "B) weight", "C) casing", "D) medium"],
    "solution": "<answer>D</answer>",
    "path": "./Knowledge/ArxivQA/images/1702.jpg",
    "data_source": "",
}


def test_convert_video_row_builds_video_prompt_and_label():
    record = _load_tool().convert_row(VIDEO_ROW, media_root="/data/vr1")

    assert record["label"] == "A"
    assert record["videos"] == ["/data/vr1/CLEVRER/video_validation/video_10000.mp4"]
    assert record["images"] == []
    prompt = record["prompt"]
    assert prompt.startswith("<video>") and "<image>" not in prompt
    assert "A. A notification" in prompt and "D. An update" in prompt
    assert record["metadata"]["choices"] == VIDEO_ROW["options"]


def test_convert_image_row_builds_image_prompt():
    record = _load_tool().convert_row(IMAGE_ROW, media_root="/data/vr1")

    assert record["label"] == "D"
    assert record["images"] == ["/data/vr1/Knowledge/ArxivQA/images/1702.jpg"]
    assert record["videos"] == []
    assert record["prompt"].startswith("<image>")


def test_reward_roundtrip_via_gpqa():
    from miles.rollout.rm_hub.gpqa import compute_gpqa_reward

    record = _load_tool().convert_row(VIDEO_ROW, media_root="/data/vr1")

    assert compute_gpqa_reward("The answer is A.", record["label"], metadata=record["metadata"]) == 1.0
    assert compute_gpqa_reward("Answer: B", record["label"], metadata=record["metadata"]) == 0.0


def test_convert_file_filters_types_sources_and_existing_media(tmp_path):
    tool = _load_tool()
    src = tmp_path / "Video-R1-260k.json"
    other = {**VIDEO_ROW, "problem_id": 3, "problem_type": "numerical", "solution": "<answer>7</answer>"}
    missing = {**VIDEO_ROW, "problem_id": 4, "path": "./CLEVRER/missing.mp4"}
    off_source = {**VIDEO_ROW, "problem_id": 5, "path": "./LLaVA-Video-178K/x.mp4"}
    with open(src, "w") as f:
        json.dump([VIDEO_ROW, IMAGE_ROW, other, missing, off_source], f)
    media_root = tmp_path / "media"
    (media_root / "CLEVRER" / "video_validation").mkdir(parents=True)
    (media_root / "CLEVRER" / "video_validation" / "video_10000.mp4").write_bytes(b"x")

    out = tmp_path / "vr1.jsonl"
    n = tool.convert_file(
        src, out, media_root=media_root, data_types=("video",), path_prefixes=("CLEVRER",), require_media=True
    )

    assert n == 1
    [record] = [json.loads(line) for line in open(out)]
    assert record["label"] == "A" and record["videos"][0].endswith("video_10000.mp4")
