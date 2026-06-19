"""Full on-policy GATE-A: GRPO LoRA RL on the Qwen3-Omni Thinker with per-step NCCL
weight-sync to the served sglang-omni thinker, so each step's rollouts are on-policy.

Beyond gate_a_lora_smoke.py this adds the 4th closed-loop component done *properly*:
after each optimizer step the LoRA-merged thinker weights are broadcast into the served
thinker stage via sglang-omni's distributed weight-update admin plane
(``/init_weights_update_group`` + ``/update_weights_from_distributed`` + ``stages=[thinker]``),
the exact pattern from sglang-omni's E2E refit test. The thinker stage's ``load_weights``
accepts plain ``model.*`` names, so the extracted-thinker names sync directly.

Run (container, miles venv, free GPU for the trainer; server already on another GPU):
    THINKER=/root/qwen3-omni-thinker DATA=examples/omni_gate_a/math_smoke.jsonl \
    SERVER=http://localhost:8000 MASTER_PORT=29555 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=4 python examples/omni_gate_a/gate_a_full.py
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

SERVER = os.environ.get("SERVER", "http://localhost:8000")
THINKER = os.environ["THINKER"]
DATA = os.environ["DATA"]
STEPS = int(os.environ.get("STEPS", "4"))
GROUP = int(os.environ.get("GROUP", "4"))
PROMPTS = int(os.environ.get("PROMPTS", "4"))
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29555"))
GROUP_NAME = "gate_a_wsync"
EPS = 0.2


def post(path: str, body: dict, timeout: int = 300):
    req = urllib.request.Request(
        SERVER + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def rollout(input_ids: list[int], seed: int):
    resp = post(
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 24, "seed": seed},
            "return_logprob": True,
        },
        timeout=120,
    )
    otl = resp["meta_info"]["output_token_logprobs"]
    return {"tokens": [t for _, t in otl], "old": [lp for lp, _ in otl], "text": resp.get("text", "")}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(THINKER, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        THINKER, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to("cuda:0")
    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    )
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)

    try:
        from sglang.srt.utils import init_custom_process_group
    except Exception:
        from sglang.srt.utils.common import init_custom_process_group

    # Rendezvous a 2-rank NCCL group with the served thinker (server joins as rank 1).
    init_err: list = []

    def _init_server():
        try:
            post(
                "/init_weights_update_group",
                {
                    "master_address": "localhost",
                    "master_port": MASTER_PORT,
                    "rank_offset": 1,
                    "world_size": 2,
                    "group_name": GROUP_NAME,
                    "backend": "nccl",
                    "stages": ["thinker"],
                },
                timeout=180,
            )
        except Exception as exc:  # noqa: BLE001
            init_err.append(exc)

    th = threading.Thread(target=_init_server)
    th.start()
    pg = init_custom_process_group(
        backend="nccl", init_method=f"tcp://localhost:{MASTER_PORT}", world_size=2, rank=0, group_name=GROUP_NAME
    )
    th.join()
    torch.cuda.synchronize()
    if init_err:
        raise init_err[0]
    print("WEIGHT_UPDATE_GROUP_READY", flush=True)

    def merged_lora_weights() -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):
                a = mod.lora_A["default"].weight
                b = mod.lora_B["default"].weight
                scaling = mod.scaling["default"]
                w = mod.base_layer.weight.data + scaling * (b @ a)
                hf = name.replace("base_model.model.", "") + ".weight"
                out[hf] = w.to(torch.bfloat16).contiguous()
        return out

    def sync_to_server() -> int:
        wd = merged_lora_weights()
        names = sorted(wd)
        spec = {
            "names": names,
            "dtypes": [str(wd[n].dtype).replace("torch.", "") for n in names],
            "shapes": [list(wd[n].shape) for n in names],
            "group_name": GROUP_NAME,
            "stages": ["thinker"],
        }
        err: list = []

        def _update():
            try:
                post("/update_weights_from_distributed", spec, timeout=300)
            except Exception as exc:  # noqa: BLE001
                err.append(exc)

        t = threading.Thread(target=_update)
        t.start()
        for n in names:
            torch.distributed.broadcast(wd[n], src=0, group=pg)
        torch.cuda.synchronize()
        t.join()
        if err:
            raise err[0]
        return len(names)

    data = [json.loads(line) for line in open(DATA)]
    print("step | mean_reward | avg_loss | synced_params")
    for step in range(STEPS):
        opt.zero_grad()
        step_reward, step_loss, n = 0.0, 0.0, 0
        for ex in data[:PROMPTS]:
            pid = tok.encode(ex["prompt"])
            samples = [rollout(pid, step * 1000 + g) for g in range(GROUP)]
            rewards = [1.0 if ex["label"] in s["text"] else 0.0 for s in samples]
            mean_r = sum(rewards) / len(rewards)
            step_reward += mean_r
            for s, adv in zip(samples, [r - mean_r for r in rewards]):
                if not s["tokens"] or adv == 0.0:
                    continue
                full = torch.tensor([pid + s["tokens"]], device="cuda:0")
                logits = model(input_ids=full).logits[0]
                p = len(pid)
                rl = logits[p - 1 : p - 1 + len(s["tokens"])].float()
                logp = torch.log_softmax(rl, dim=-1)
                rt = torch.tensor(s["tokens"], device="cuda:0")
                new = logp[range(len(s["tokens"])), rt]
                old = torch.tensor(s["old"], device="cuda:0")
                ratio = torch.exp(new - old)
                loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv).mean()
                loss = loss / (GROUP * PROMPTS)
                loss.backward()
                step_loss += loss.item() * (GROUP * PROMPTS)
                n += 1
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        synced = sync_to_server()  # next step's rollouts are on-policy
        print(f"{step:4d} | {step_reward / PROMPTS:11.3f} | {step_loss / max(n, 1):8.4f} | {synced}", flush=True)

    print("GATE-A FULL on-policy loop complete (per-step NCCL weight-sync to served thinker)")


if __name__ == "__main__":
    main()
