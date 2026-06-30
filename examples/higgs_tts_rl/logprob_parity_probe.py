"""Logprob-parity gate for the GATE-B trainable TTS actor.

Right after load the trainer-side actor and the served model are the same policy,
so the actor's recomputed codebook-0 log-probs must match the rollout's
`output_token_logprobs`. This is the make-or-break correctness check before any
GRPO update.

Run (container, miles venv; Higgs server serving on SERVER):
    SERVER=http://localhost:8010 HIGGS_CKPT='<snapshot glob>' CUDA_VISIBLE_DEVICES=4 \
    PYTHONPATH=/root/rl-omni/sglang-omni:/root/rl-omni/miles \
    python examples/omni_gate_b/gate_b_parity_probe.py
"""

from __future__ import annotations

import glob
import json
import os
import urllib.request

SERVER = os.environ.get("SERVER", "http://localhost:8010")
# Gate on mean|Δ|: the residual is the served model's bf16 + sglang-kernel numeric
# floor (an fp32 trainer gives the SAME ~0.05 residual), so per-token max|Δ| of ~0.2
# is irreducible cross-implementation noise, not a reconstruction error. exp(0.2)≈1.22
# sits at the GRPO clip boundary and only biases the first ratio after each sync.
TOL = float(os.environ.get("PARITY_TOL", "0.10"))


def _rollout(prompt_ids: list[int], seed: int) -> dict:
    req = {
        "input_ids": prompt_ids,
        "sampling_params": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 256, "seed": seed},
        "return_logprob": True,
        "output_modalities": ["audio"],
    }
    resp = json.loads(urllib.request.urlopen(
        urllib.request.Request(SERVER + "/generate", data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"}),
        timeout=180,
    ).read())
    meta = resp["meta_info"]
    otl = meta.get("output_token_logprobs") or []
    return {
        "old_logprobs": [float(lp) for lp, _ in otl],
        "cb0_tokens": [int(t) for _, t in otl],
        "codebook_tokens": meta.get("output_codebook_tokens"),
    }


def main() -> None:
    import torch
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.higgs_tts.text_tokenizer import HiggsTokenizerAdapter

    from miles_plugins.omni.higgs_actor import HiggsTtsActor

    ckpt = glob.glob(os.environ["HIGGS_CKPT"])[0] if "*" in os.environ["HIGGS_CKPT"] else os.environ["HIGGS_CKPT"]
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json")))
    adapter = HiggsTokenizerAdapter(tok)
    device = os.environ.get("ACTOR_DEVICE", "cuda:0")

    dtype = torch.float32 if os.environ.get("ACTOR_DTYPE") == "fp32" else torch.bfloat16
    actor = HiggsTtsActor(ckpt, device=device, dtype=dtype)
    print("actor loaded; backbone dtype", actor.dtype)

    texts = ["Hello world.", "The quick brown fox."]
    worst = 0.0
    worst_mean = 0.0
    for i, text in enumerate(texts):
        pid = list(map(int, adapter.build_prompt(text, num_ref_tokens=0)))
        r = _rollout(pid, seed=1000 + i)
        codes = r["codebook_tokens"]
        old = r["old_logprobs"]
        if not codes or not old:
            print(f"[{text!r}] no codes/logprobs returned -> SKIP (server missing Step-1 fix?)")
            continue
        assert all(row[0] == t for row, t in zip(codes, r["cb0_tokens"])), "cb0 mismatch codes vs logprob tokens"

        with torch.no_grad():
            new = actor.codebook0_logprobs(pid, codes).tolist()
        n = min(len(new), len(old))
        diffs = [abs(new[j] - old[j]) for j in range(n)]
        max_d = max(diffs)
        mean_d = sum(diffs) / n
        worst = max(worst, max_d)
        worst_mean = max(worst_mean, mean_d)
        print(f"[{text!r}] T={n} max|Δ|={max_d:.4f} mean|Δ|={mean_d:.4f}")
        print(f"   old[:5]={[round(x,3) for x in old[:5]]}")
        print(f"   new[:5]={[round(x,3) for x in new[:5]]}")

    print(f"WORST_MEAN_ABS_DIFF: {worst_mean:.4f}  (tol={TOL})   WORST_MAX_ABS_DIFF: {worst:.4f}")
    print(f"PARITY_OK: {worst_mean < TOL}")


if __name__ == "__main__":
    main()
