"""Full GATE-B closed loop: GRPO LoRA RL on the Higgs TTS actor with per-step NCCL
weight-sync to the served sglang-omni ``tts_engine`` stage, so each step's rollouts
are on-policy.

The fourth closed-loop component for TTS, mirroring gate_a_full.py:
  rollout (/generate -> codec tokens + codebook-0 logprobs + audio)
  -> composite reward (Whisper ASR CER + audio-validity guards)
  -> GRPO advantage over codebook-0 tokens
  -> LoRA policy update (trainer recomputes new logprobs via HiggsTtsActor)
  -> NCCL broadcast of LoRA-merged backbone weights into the served tts_engine stage
     (names in the checkpoint `body.*` convention; the server fuses q/k/v on load).

Run (container, miles venv, free GPU for the trainer; Higgs server on another GPU):
    SERVER=http://localhost:8010 HIGGS_CKPT='<snapshot glob>' MASTER_PORT=29641 \
    ASR_MODEL=openai/whisper-base ASR_DEVICE=cuda:0 CUDA_VISIBLE_DEVICES=4 \
    HF_HUB_OFFLINE=1 NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 \
    PYTHONPATH=/root/rl-omni/sglang-omni:/root/rl-omni/miles \
    python examples/omni_gate_b/gate_b_full.py

CRITICAL: set NCCL_P2P_DISABLE=1 on BOTH the server and the trainer (single-GPU masks
per process), exactly as in gate_a_full.py.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import urllib.request

import torch
from peft import LoraConfig, get_peft_model

SERVER = os.environ.get("SERVER", "http://localhost:8010")
HIGGS_CKPT = os.environ["HIGGS_CKPT"]
DATA = os.environ.get("DATA", "examples/omni_gate_b/tts_smoke.jsonl")
STEPS = int(os.environ.get("STEPS", "3"))
GROUP = int(os.environ.get("GROUP", "4"))
PROMPTS = int(os.environ.get("PROMPTS", "4"))
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29641"))
GROUP_NAME = os.environ.get("GROUP_NAME", "gate_b_wsync")
TEMP = float(os.environ.get("TEMP", "0.8"))
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
EPS = 0.2


def post(path: str, body: dict, timeout: int = 300):
    req = urllib.request.Request(
        SERVER + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def rollout(input_ids: list[int], seed: int) -> dict:
    resp = post(
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {"temperature": TEMP, "top_p": 0.95, "max_new_tokens": MAX_NEW, "seed": seed},
            "return_logprob": True,
            "output_modalities": ["audio"],
        },
        timeout=180,
    )
    meta = resp["meta_info"]
    otl = meta.get("output_token_logprobs") or []
    return {
        "old": [lp for lp, _ in otl],
        "codes": meta.get("output_codebook_tokens"),
        "audio": (resp.get("audio") or {}).get("data"),
    }


def main() -> None:
    torch.cuda.set_device(0)  # bind this process to its visible GPU for NCCL collectives

    from sglang_omni.models.higgs_tts.text_tokenizer import HiggsTokenizerAdapter
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    from miles_plugins.omni.higgs_actor import HiggsTtsActor
    from miles_plugins.omni.tts_reward import TtsCompositeReward

    ckpt = glob.glob(HIGGS_CKPT)[0] if "*" in HIGGS_CKPT else HIGGS_CKPT
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json")))
    adapter = HiggsTokenizerAdapter(tok)
    reward_fn = TtsCompositeReward()

    actor = HiggsTtsActor(ckpt, device="cuda:0")
    actor.backbone = get_peft_model(
        actor.backbone,
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=None),
    )
    actor.backbone.train()
    opt = torch.optim.AdamW([p for p in actor.backbone.parameters() if p.requires_grad], lr=2e-5)

    try:
        from sglang.srt.utils import init_custom_process_group
    except Exception:
        from sglang.srt.utils.common import init_custom_process_group

    # Rendezvous a 2-rank NCCL group with the served tts_engine stage (server = rank 1).
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
                    "stages": ["tts_engine"],
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
        for name, mod in actor.backbone.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):
                a = mod.lora_A["default"].weight
                b = mod.lora_B["default"].weight
                scaling = mod.scaling["default"]
                w = mod.base_layer.weight.data + scaling * (b @ a)
                # peft module name base_model.model.layers.N... -> ckpt body.layers.N...
                hf = name.replace("base_model.model.", "")
                out["body." + hf + ".weight"] = w.to(torch.bfloat16).contiguous()
        return out

    def sync_to_server() -> int:
        wd = merged_lora_weights()
        names = sorted(wd)
        spec = {
            "names": names,
            "dtypes": [str(wd[n].dtype).replace("torch.", "") for n in names],
            "shapes": [list(wd[n].shape) for n in names],
            "group_name": GROUP_NAME,
            "stages": ["tts_engine"],
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
    print("step | mean_reward | mean_cer | avg_loss | synced_params")
    for step in range(STEPS):
        opt.zero_grad()
        step_reward, step_cer, n_cer, step_loss, n = 0.0, 0.0, 0, 0.0, 0
        for ex in data[:PROMPTS]:
            pid = list(map(int, adapter.build_prompt(ex["text"], num_ref_tokens=0)))
            samples = [rollout(pid, step * 1000 + g) for g in range(GROUP)]
            comps = [reward_fn.score(s["audio"], ex["label"]) for s in samples]
            rewards = [c.reward for c in comps]
            mean_r = sum(rewards) / len(rewards)
            step_reward += mean_r
            for c in comps:
                if c.cer is not None:
                    step_cer += c.cer
                    n_cer += 1
            for s, adv in zip(samples, [r - mean_r for r in rewards]):
                codes = s["codes"]
                if not codes or adv == 0.0:
                    continue
                new = actor.codebook0_logprobs(pid, codes)
                T = min(len(new), len(s["old"]))
                new = new[:T]
                old = torch.tensor(s["old"][:T], device="cuda:0")
                ratio = torch.exp(new - old)
                loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv).mean()
                loss = loss / (GROUP * PROMPTS)
                loss.backward()
                step_loss += loss.item() * (GROUP * PROMPTS)
                n += 1
        torch.nn.utils.clip_grad_norm_([p for p in actor.backbone.parameters() if p.requires_grad], 1.0)
        opt.step()
        synced = sync_to_server()  # next step's rollouts are on-policy
        mean_cer = step_cer / n_cer if n_cer else float("nan")
        print(
            f"{step:4d} | {step_reward / PROMPTS:11.3f} | {mean_cer:8.3f} | "
            f"{step_loss / max(n, 1):8.4f} | {synced}",
            flush=True,
        )

    print("GATE-B FULL on-policy loop complete (per-step NCCL weight-sync to served tts_engine)")


if __name__ == "__main__":
    main()
