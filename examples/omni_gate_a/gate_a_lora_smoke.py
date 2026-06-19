"""GATE-A closed-loop smoke: GRPO RL on the Qwen3-Omni Thinker (single GPU, LoRA).

Demonstrates all four closed-loop components on the real model end to end:
  rollout (sglang-omni /generate)  ->  reward (math correctness)
  ->  GRPO advantage  ->  LoRA policy-gradient update.

It is a deliberately minimal, self-contained harness (no Ray / FSDP / miles trainer)
so the loop mechanics can be verified on one 80GB GPU. The behavior policy is the
served base thinker; LoRA is updated locally (rollouts are not weight-synced back, which
is the documented next integration step), so this proves loop stability, not on-policy
convergence.

Run (inside the container, miles venv):
    THINKER=/root/qwen3-omni-thinker DATA=examples/omni_gate_a/math_smoke.jsonl \
    SERVER=http://localhost:8000/generate CUDA_VISIBLE_DEVICES=4 \
    python examples/omni_gate_a/gate_a_lora_smoke.py
"""

from __future__ import annotations

import json
import os
import urllib.request

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

SERVER = os.environ.get("SERVER", "http://localhost:8000/generate")
THINKER = os.environ["THINKER"]
DATA = os.environ["DATA"]
STEPS = int(os.environ.get("STEPS", "5"))
GROUP = int(os.environ.get("GROUP", "4"))
PROMPTS_PER_STEP = int(os.environ.get("PROMPTS_PER_STEP", "4"))
EPS = 0.2


def rollout(input_ids: list[int], seed: int):
    req = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 24, "seed": seed},
        "return_logprob": True,
    }
    r = urllib.request.urlopen(
        urllib.request.Request(SERVER, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}),
        timeout=120,
    )
    resp = json.loads(r.read())
    otl = resp["meta_info"]["output_token_logprobs"]
    return {
        "tokens": [t for _, t in otl],
        "old_logprobs": [lp for lp, _ in otl],
        "text": resp.get("text", ""),
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(THINKER, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        THINKER, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to("cuda:0")
    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    )
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-5)
    print(f"trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    data = [json.loads(line) for line in open(DATA)]
    print("step | mean_reward | avg_loss | (per-prompt rewards)")
    for step in range(STEPS):
        opt.zero_grad()
        step_reward, step_loss, n = 0.0, 0.0, 0
        per_prompt = []
        for ex in data[:PROMPTS_PER_STEP]:
            prompt_ids = tok.encode(ex["prompt"])
            samples = [rollout(prompt_ids, step * 1000 + g) for g in range(GROUP)]
            rewards = [1.0 if ex["label"] in s["text"] else 0.0 for s in samples]
            mean_r = sum(rewards) / len(rewards)
            per_prompt.append(mean_r)
            step_reward += mean_r
            advs = [r - mean_r for r in rewards]
            for s, adv in zip(samples, advs):
                rt = s["tokens"]
                if not rt or adv == 0.0:
                    continue
                full = torch.tensor([prompt_ids + rt], device="cuda:0")
                logits = model(input_ids=full).logits[0]
                p = len(prompt_ids)
                resp_logits = logits[p - 1 : p - 1 + len(rt)].float()
                logp = torch.log_softmax(resp_logits, dim=-1)
                idx = torch.tensor(rt, device="cuda:0")
                new_lp = logp[range(len(rt)), idx]
                old_lp = torch.tensor(s["old_logprobs"], device="cuda:0")
                ratio = torch.exp(new_lp - old_lp)
                loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv).mean()
                loss = loss / (GROUP * PROMPTS_PER_STEP)
                loss.backward()
                step_loss += loss.item() * (GROUP * PROMPTS_PER_STEP)
                n += 1
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        mean_reward = step_reward / PROMPTS_PER_STEP
        avg_loss = step_loss / max(n, 1)
        print(f"{step:4d} | {mean_reward:11.3f} | {avg_loss:8.4f} | {per_prompt}")

    print("GATE-A closed-loop smoke complete (rollout->reward->advantage->LoRA update over multiple steps)")


if __name__ == "__main__":
    main()
