"""Convert AVQA (Joysw909/AVQA, r1aqa line-json) into a miles jsonl for audio-input GRPO.

Each output row is the miles multimodal convention: a `<audio>` placeholder prompt (paired
with --multimodal-keys '{"audio": "audios"}'), the option letter as label, and the raw
choices in metadata so `--rm-type gpqa` scores letter or option-text answers.

    python examples/omni_thinker/prepare_avqa.py \
        --src <dataset_dir>/train_r1aqa_line.json --dst <data_dir>/avqa.jsonl \
        --audio-root <dataset_dir> [--max-samples 5000]
"""

import argparse
import json
import string
from pathlib import Path

# AVQA questions say "in the video", but the r1aqa recipe feeds the audio track only; the
# questions are answerable from sound (VGGSound clips).
_PROMPT_TEMPLATE = (
    "<audio>{question}\n{options}\n"
    "Listen to the audio and choose the best option. Respond with the option letter in the form 'Answer: <letter>'."
)
_DATASET_PATH_PREFIX = "./Joysw909/AVQA/"


def convert_row(row: dict, audio_root: str) -> dict:
    choices = list(row["multi_choice"])
    answer_index = int(row["answer"])
    assert 0 <= answer_index < len(choices) <= len(string.ascii_uppercase), f"bad row: {row}"
    letters = string.ascii_uppercase[: len(choices)]
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices, strict=True))
    audio_rel = row["audio_path"].removeprefix(_DATASET_PATH_PREFIX)
    return {
        "prompt": _PROMPT_TEMPLATE.format(question=row["question_text"].strip(), options=options),
        "audios": [str(Path(audio_root) / audio_rel)],
        "label": letters[answer_index],
        "metadata": {"choices": choices},
    }


def convert_file(src, dst, audio_root: str, max_samples: int | None = None) -> int:
    n = skipped = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            if max_samples is not None and n >= max_samples:
                break
            record = convert_row(json.loads(line), audio_root)
            # the HF mirror has holes (delisted VGGSound clips); a missing wav must not
            # produce a row that crashes rollout preprocessing at startup
            if not all(Path(audio).exists() for audio in record["audios"]):
                skipped += 1
                continue
            fout.write(json.dumps(record) + "\n")
            n += 1
    assert n > 0, f"no rows converted from {src}"
    if skipped:
        print(f"[warn] skipped {skipped} rows with missing audio files")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="train_r1aqa_line.json from Joysw909/AVQA")
    parser.add_argument("--dst", required=True, help="output jsonl path")
    parser.add_argument("--audio-root", required=True, help="dataset dir containing VGG*/*.wav")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    n = convert_file(args.src, args.dst, audio_root=args.audio_root, max_samples=args.max_samples)
    print(f"[done] {n} samples -> {args.dst}")


if __name__ == "__main__":
    main()
