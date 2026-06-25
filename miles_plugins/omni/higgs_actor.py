"""Trainer-side Higgs TTS actor: a gradient-enabled teacher-forced forward that
reproduces the served model's per-step codebook-0 log-probs over a sampled codec
sequence, so the RL trainer can recompute new-policy log-probs for GRPO.

The served `HiggsTTSModel` backbone is sglang's inference `Qwen3ForCausalLM`
(paged attention / CUDA graph, no autograd), so it cannot be trained directly.
This rebuilds the same policy from the checkpoint with a plain `transformers`
Qwen3 backbone + the fused codec embedding/head, which IS differentiable.

Correctness is gated by a logprob-parity check against the server (see
`examples/omni_gate_b/gate_b_parity_probe.py`): right after load the trainer and
the server are the same policy, so recomputed log-probs must match.
"""

from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn.functional as F


# Checkpoint-name → transformers Qwen3Model state-dict-name (mirrors the server's
# DiscreteWeightMapper + _BACKBONE_PREFIX_MAP, but targets a plain Qwen3Model).
_BACKBONE_RENAME = {
    "tied.embedding.text_embedding.": "embed_tokens.",
    "body.layers.": "layers.",
    "body.norm.": "norm.",
}
_FUSED_EMBED_KEY = "tied.embedding.modality_embeddings.0.embedding.weight"


def _resolve_ckpt_dir(path_or_glob: str) -> str:
    if "*" in path_or_glob:
        matches = glob.glob(path_or_glob)
        if not matches:
            raise FileNotFoundError(f"no checkpoint dir matches {path_or_glob!r}")
        return matches[0]
    return path_or_glob


class HiggsTtsActor:
    """Differentiable Higgs codec policy (Qwen3 backbone + fused codebook head)."""

    def __init__(self, ckpt_dir: str, device: str = "cuda:0", dtype=torch.bfloat16):
        from safetensors import safe_open
        from transformers import Qwen3Config, Qwen3Model

        ckpt_dir = _resolve_ckpt_dir(ckpt_dir)
        self.device = device
        self.dtype = dtype

        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        text_cfg = cfg["text_config"]
        enc_cfg = cfg["audio_encoder_config"]
        self.num_codebooks = int(enc_cfg["num_codebooks"])
        self.codebook_vocab = int(enc_cfg["vocab_size"])

        backbone = Qwen3Model(Qwen3Config(**text_cfg)).to(device=device, dtype=dtype).eval()
        self.backbone = backbone

        # Stream the shards once: route backbone tensors into a state dict, grab the
        # fused codec embedding weight, and drop the (skipped) audio-encoder tensors.
        backbone_sd: dict[str, torch.Tensor] = {}
        fused_embed: torch.Tensor | None = None
        index = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))
        for shard in sorted(set(index["weight_map"].values())):
            with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
                for key in f.keys():
                    if key == _FUSED_EMBED_KEY:
                        fused_embed = f.get_tensor(key)
                        continue
                    renamed = self._rename_backbone(key)
                    if renamed is not None:
                        backbone_sd[renamed] = f.get_tensor(key)

        if fused_embed is None:
            raise KeyError(f"fused codec embedding {_FUSED_EMBED_KEY!r} not in checkpoint")
        missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
        # Qwen3Model ties embed_tokens; lm_head/text_head is intentionally absent here.
        unexpected = [u for u in unexpected if "lm_head" not in u and "text_head" not in u]
        if unexpected:
            raise RuntimeError(f"unexpected backbone keys: {unexpected[:5]}")
        real_missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
        if real_missing:
            raise RuntimeError(f"missing backbone keys: {real_missing[:5]}")

        # Fused codebook weight [N*V, D]: input embedding (sum over codebooks) and,
        # tied, the codebook-0 head = its first V rows.
        self.fused_embed = fused_embed.to(device=device, dtype=dtype)
        self._cb_offsets = (
            torch.arange(self.num_codebooks, device=device) * self.codebook_vocab
        )

    def _rename_backbone(self, key: str) -> str | None:
        if key.startswith("tied.embedding.modality_embeddings.0.model."):
            return None  # audio encoder — not part of the AR policy
        for src, dst in _BACKBONE_RENAME.items():
            if key.startswith(src):
                return dst + key[len(src) :]
        return None  # text_head / anything else: skip

    def _embed_codes(self, codes_LN: torch.Tensor) -> torch.Tensor:
        """[L, N] codebook ids → [L, D] fused embedding (mirrors the served model)."""
        fused_ids = codes_LN + self._cb_offsets
        return F.embedding(fused_ids, self.fused_embed).sum(dim=-2)

    def codebook0_logprobs(
        self, prompt_ids: list[int], codebook_tokens: list[list[int]]
    ) -> torch.Tensor:
        """Teacher-forced new-policy log-probs of each step's sampled codebook-0 token.

        ``codebook_tokens`` is ``[T, num_codebooks]`` (the full per-step codes the
        server fed back). Returns ``[T]`` log-probs aligned with the rollout's
        ``output_token_logprobs`` (codebook-0).
        """
        device = self.device
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        codes = torch.tensor(codebook_tokens, dtype=torch.long, device=device)  # [T, N]
        T = int(codes.shape[0])
        P = int(prompt.shape[0])

        text_emb = self.backbone.embed_tokens(prompt)  # [P, D]
        # Teacher forcing: step t (t>=1) is predicted from the embedding of step t-1's
        # full codes; step 0 is predicted from the last prompt token. So feed prompt +
        # codes[0..T-2]; read hidden at positions P-1 .. P+T-2 for steps 0 .. T-1.
        if T > 1:
            codec_emb = self._embed_codes(codes[: T - 1])  # [T-1, D]
            inputs_embeds = torch.cat([text_emb, codec_emb], dim=0)
        else:
            inputs_embeds = text_emb
        L = inputs_embeds.shape[0]
        positions = torch.arange(L, device=device).unsqueeze(0)

        out = self.backbone(
            inputs_embeds=inputs_embeds.unsqueeze(0),
            position_ids=positions,
            use_cache=False,
        )
        hidden = out.last_hidden_state[0]  # [L, D]
        step_hidden = hidden[P - 1 : P - 1 + T]  # [T, D]
        cb0_logits = F.linear(step_hidden.float(), self.fused_embed[: self.codebook_vocab].float())
        logp = torch.log_softmax(cb0_logits, dim=-1)  # [T, V]
        sampled_cb0 = codes[:, 0]  # [T]
        return logp[torch.arange(T, device=device), sampled_cb0]
