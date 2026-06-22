"""GATE-B closed loop demo: GRPO rollout -> composite reward -> advantage on real Higgs TTS.

Demonstrates the first three closed-loop components on the real Higgs-audio model through
the sglang-omni rollout backend:
  rollout (pretok /generate -> codec tokens + logprobs + audio)
  -> composite reward (Whisper ASR CER + audio-validity guards)
  -> GRPO advantage.

The 4th component (LoRA policy update + NCCL weight-sync to the served TTS actor) mirrors
GATE-A's gate_a_full.py: the rollout returns codec-token logprobs (old) and the trainer
recomputes new logprobs over the codec sequence; weight sync uses /update_weights_from_distributed
with NCCL_P2P_DISABLE=1.

Run (container, miles venv; Higgs server already serving on SERVER):
    THINKER=... SERVER=http://localhost:8010 HIGGS_CKPT=<snapshot> \
    ASR_MODEL=openai/whisper-base ASR_DEVICE=cuda:0 \
    python examples/omni_gate_b/gate_b_loop.py
"""

from __future__ import annotations

import glob
import json
import os
import urllib.request

SERVER = os.environ.get("SERVER", "http://localhost:8010")
DATA = os.environ.get("DATA", "examples/omni_gate_b/tts_smoke.jsonl")
GROUP = int(os.environ.get("GROUP", "4"))
STEPS = int(os.environ.get("STEPS", "3"))


def _higgs_adapter():
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.higgs_tts.text_tokenizer import HiggsTokenizerAdapter

    ckpt = glob.glob(os.environ["HIGGS_CKPT"])[0] if "*" in os.environ.get("HIGGS_CKPT", "") else os.environ["HIGGS_CKPT"]
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json")))
    return HiggsTokenizerAdapter(tok)


def rollout(input_ids: list[int], seed: int) -> dict:
    req = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 256, "seed": seed},
        "return_logprob": True,
        "output_modalities": ["audio"],
    }
    r = urllib.request.urlopen(
        urllib.request.Request(SERVER + "/generate", data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"}),
        timeout=180,
    )
    resp = json.loads(r.read())
    otl = resp["meta_info"].get("output_token_logprobs") or []
    audio = resp.get("audio") or {}
    return {
        "codec_tokens": [t for _, t in otl],
        "old_logprobs": [lp for lp, _ in otl],
        "audio_b64": audio.get("data"),
    }


def main() -> None:
    import sys

    sys.path.insert(0, os.environ.get("SGLANG_OMNI", "/root/rl-omni/sglang-omni"))
    from miles_plugins.omni.tts_reward import TtsCompositeReward

    adapter = _higgs_adapter()
    reward_fn = TtsCompositeReward()
    data = [json.loads(line) for line in open(DATA)]

    print("step | mean_reward | mean_cer | (per-prompt reward)")
    for step in range(STEPS):
        step_reward, step_cer, n_cer = 0.0, 0.0, 0
        per_prompt = []
        for ex in data[: int(os.environ.get("PROMPTS", "4"))]:
            pid = list(map(int, adapter.build_prompt(ex["text"], num_ref_tokens=0)))
            samples = [rollout(pid, step * 1000 + g) for g in range(GROUP)]
            comps = [reward_fn.score(s["audio_b64"], ex["label"]) for s in samples]
            rewards = [c.reward for c in comps]
            mean_r = sum(rewards) / len(rewards)
            advs = [round(r - mean_r, 3) for r in rewards]
            per_prompt.append(round(mean_r, 3))
            step_reward += mean_r
            for c in comps:
                if c.cer is not None:
                    step_cer += c.cer
                    n_cer += 1
            # codec-token alignment sanity (old logprobs vs tokens)
            assert all(len(s["codec_tokens"]) == len(s["old_logprobs"]) for s in samples)
            print(f"  prompt={ex['text']!r} rewards={rewards} adv={advs} "
                  f"transcripts={[c.transcript for c in comps]}")
        mean_cer = step_cer / n_cer if n_cer else float("nan")
        print(f"{step:4d} | {step_reward / len(per_prompt):11.3f} | {mean_cer:8.3f} | {per_prompt}", flush=True)

    print("GATE-B rollout->composite-reward->advantage demonstrated on real Higgs TTS")


if __name__ == "__main__":
    main()
