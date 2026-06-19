"""Text-only math-correctness reward for the GATE-A thinker RL smoke.

Loaded via ``--custom-rm-path miles_plugins.omni.math_reward.compute_math_reward``.
Returns 1.0 when the model's decoded response contains the gold answer
(``sample.label``), else 0.0. Deterministic and dependency-free, so it suits the first
``one_update_smoke``/``multi_step_stability`` run and the deterministic-reward criterion.
"""

from __future__ import annotations

import re

from miles.utils.types import Sample

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def _normalize(value: str) -> str:
    # 12.0 and 12 should compare equal as answers
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return value.strip()


async def compute_math_reward(args, sample: Sample, **kwargs) -> float:
    """1.0 if the response's answer matches the gold label, else 0.0."""
    label = "" if sample.label is None else str(sample.label).strip()
    if not label:
        return 0.0
    response = sample.response or ""

    gold = _normalize(label)
    # numeric match (handles "= 12", "12.0", trailing punctuation), then substring fallback
    if any(_normalize(n) == gold for n in _numbers(response)):
        return 1.0
    return 1.0 if label in response else 0.0
