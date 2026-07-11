"""On-policy Higgs TTS GRPO with per-step SGLang-Omni weight sync.

Set ``TRAIN_MODE=lora`` (default) for the low-memory smoke path or ``full`` to
train and sync the complete backbone plus tied codebook embedding/head.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import urllib.request

import torch
from peft import LoraConfig, get_peft_model

from miles_plugins.omni.rollout_contract import (
    build_generate_payload,
    parse_generate_response,
    parse_omni_action_stream,
)

SERVER = os.environ.get("SERVER", "http://localhost:8010")
HIGGS_CKPT = os.environ["HIGGS_CKPT"]
DATA = os.environ.get("DATA", "examples/higgs_tts_rl/tts_smoke.jsonl")
STEPS = int(os.environ.get("STEPS", "3"))
GROUP = int(os.environ.get("GROUP", "4"))
PROMPTS = int(os.environ.get("PROMPTS", "4"))
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29641"))
GROUP_NAME = os.environ.get("GROUP_NAME", "higgs_tts_wsync")
TEMP = float(os.environ.get("TEMP", "0.8"))
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
TOP_K = int(os.environ["TOP_K"]) if os.environ.get("TOP_K") else None
TRAIN_MODE = os.environ.get("TRAIN_MODE", "lora").lower()
LR = float(os.environ.get("LR", "2e-5" if TRAIN_MODE == "lora" else "1e-6"))
EPS = 0.2


def post(path: str, body: dict, timeout: int = 300):
    req = urllib.request.Request(
        SERVER + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def rollout(input_ids: list[int], seed: int) -> dict:
    sampling_params = {
        "temperature": TEMP,
        "top_p": 0.95,
        "max_new_tokens": MAX_NEW,
        "seed": seed,
    }
    if TOP_K is not None:
        sampling_params["top_k"] = TOP_K
    resp = post(
        "/generate",
        build_generate_payload(
            input_ids,
            sampling_params,
            output_modalities=["audio"],
            return_omni_rollout=True,
        ),
        timeout=180,
    )
    result = parse_generate_response(resp)
    stream = parse_omni_action_stream(result.omni_rollout, "higgs_codes")
    if result.output_codebook_tokens != stream.actions:
        raise ValueError("Higgs output_codebook_tokens do not match omni_rollout actions")
    return {
        "old": stream.logprobs,
        "mask": stream.action_mask,
        "codes": stream.actions,
        "audio": (result.audio or {}).get("data"),
    }


def main() -> None:
    torch.cuda.set_device(0)  # bind this process to its visible GPU for NCCL collectives

    from sglang_omni.models.higgs_tts.text_tokenizer import HiggsTokenizerAdapter
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    from miles_plugins.omni.higgs_actor import HiggsTtsActor, clipped_grpo_loss
    from miles_plugins.omni.tts_reward import TtsCompositeReward

    ckpt = glob.glob(HIGGS_CKPT)[0] if "*" in HIGGS_CKPT else HIGGS_CKPT
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json")))
    adapter = HiggsTokenizerAdapter(tok)
    reward_fn = TtsCompositeReward()

    actor = HiggsTtsActor(ckpt, device="cuda:0")
    if TRAIN_MODE == "lora":
        actor.fused_embed.requires_grad_(False)
        actor.backbone = get_peft_model(
            actor.backbone,
            LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=None),
        )
    elif TRAIN_MODE != "full":
        raise ValueError(f"TRAIN_MODE must be 'lora' or 'full', got {TRAIN_MODE!r}")
    actor.train()
    trainable_params = [param for param in actor.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=LR)

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

    @torch.no_grad()
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

    @torch.no_grad()
    def weights_to_sync() -> dict[str, torch.Tensor]:
        if TRAIN_MODE == "lora":
            return merged_lora_weights()
        return {
            name: tensor.detach().to(torch.bfloat16).contiguous()
            for name, tensor in actor.full_server_weights().items()
        }

    def sync_to_server() -> int:
        wd = weights_to_sync()
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
            for s, adv in zip(samples, [r - mean_r for r in rewards], strict=True):
                codes = s["codes"]
                if not codes or adv == 0.0:
                    continue
                new = actor.codebook_logprobs(pid, codes, temperature=TEMP, top_k=TOP_K)
                old = torch.tensor(s["old"], dtype=new.dtype, device="cuda:0")
                mask = torch.tensor(s["mask"], dtype=torch.bool, device="cuda:0")
                loss = clipped_grpo_loss(new, old, mask, advantage=adv, clip_eps=EPS)
                loss = loss / (GROUP * PROMPTS)
                loss.backward()
                step_loss += loss.item() * (GROUP * PROMPTS)
                n += 1
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        synced = sync_to_server()  # next step's rollouts are on-policy
        mean_cer = step_cer / n_cer if n_cer else float("nan")
        print(
            f"{step:4d} | {step_reward / PROMPTS:11.3f} | {mean_cer:8.3f} | "
            f"{step_loss / max(n, 1):8.4f} | {synced}",
            flush=True,
        )

    print("Higgs TTS on-policy loop complete (per-step NCCL weight-sync to served tts_engine)")


if __name__ == "__main__":
    main()
