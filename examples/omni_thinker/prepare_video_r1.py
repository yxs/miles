"""Convert Video-R1-260k (mixed image/video MCQ with verifiable answers) into miles jsonl.

Rows carry lettered options and `<answer>X</answer>` solutions, so `--rm-type gpqa`
scores them directly. Both media columns are always present (one empty), pairing with
`--multimodal-keys '{"video": "videos", "image": "images"}'`.

    python examples/omni_thinker/prepare_video_r1.py \
        --src <dir>/Video-R1-260k.json --dst <data_dir>/video_r1.jsonl \
        --media-root <dir> [--data-types video image] [--path-prefixes CLEVRER] [--max-samples 5000]
"""

import argparse
import json
import re
from pathlib import Path

_PLACEHOLDER = {"video": "<video>", "image": "<image>"}
_PROMPT_TEMPLATE = (
    "{placeholder}{problem}\n{options}\n"
    "Watch carefully and choose the best option. Respond with the option letter in the form 'Answer: <letter>'."
)
_ANSWER_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>")


def convert_row(row: dict, media_root) -> dict:
    match = _ANSWER_RE.search(row["solution"])
    assert match, f"no letter answer in solution: {row['solution']!r} (problem_id={row.get('problem_id')})"
    letter = match.group(1).upper()
    media_path = str(Path(media_root) / row["path"].removeprefix("./"))
    data_type = row["data_type"]
    assert data_type in _PLACEHOLDER, f"unsupported data_type: {data_type}"
    return {
        "prompt": _PROMPT_TEMPLATE.format(
            placeholder=_PLACEHOLDER[data_type],
            problem=row["problem"].strip(),
            options="\n".join(row["options"]),
        ),
        "videos": [media_path] if data_type == "video" else [],
        "images": [media_path] if data_type == "image" else [],
        "label": letter,
        "metadata": {"choices": list(row["options"])},
    }


def convert_file(
    src,
    dst,
    media_root,
    data_types: tuple[str, ...] = ("video", "image"),
    path_prefixes: tuple[str, ...] | None = None,
    problem_types: tuple[str, ...] = ("multiple choice",),
    require_media: bool = False,
    max_samples: int | None = None,
) -> int:
    with open(src) as f:
        rows = json.load(f)

    n = 0
    with open(dst, "w") as fout:
        for row in rows:
            if max_samples is not None and n >= max_samples:
                break
            if row["data_type"] not in data_types or row["problem_type"] not in problem_types:
                continue
            rel = row["path"].removeprefix("./")
            if path_prefixes and not rel.startswith(tuple(path_prefixes)):
                continue
            if require_media and not (Path(media_root) / rel).exists():
                continue
            fout.write(json.dumps(convert_row(row, media_root)) + "\n")
            n += 1
    assert n > 0, f"no rows converted from {src}"
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Video-R1-260k.json")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--media-root", required=True, help="dir the media zips were extracted into")
    parser.add_argument("--data-types", nargs="+", default=["video", "image"])
    parser.add_argument("--path-prefixes", nargs="+", default=None, help="e.g. CLEVRER to match downloaded zips")
    parser.add_argument("--require-media", action="store_true", help="skip rows whose media file is absent")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    n = convert_file(
        args.src,
        args.dst,
        media_root=args.media_root,
        data_types=tuple(args.data_types),
        path_prefixes=tuple(args.path_prefixes) if args.path_prefixes else None,
        require_media=args.require_media,
        max_samples=args.max_samples,
    )
    print(f"[done] {n} samples -> {args.dst}")


if __name__ == "__main__":
    main()
