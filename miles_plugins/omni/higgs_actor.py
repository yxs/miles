"""Trainer-side Higgs TTS actor with gradient-enabled codebook logprob replay.

The served `HiggsTTSModel` backbone is sglang's inference `Qwen3ForCausalLM`
(paged attention / CUDA graph, no autograd), so it cannot be trained directly.
This rebuilds the same policy from the checkpoint with a plain `transformers`
Qwen3 backbone + the fused codec embedding/head, which IS differentiable.

Correctness is gated by a logprob-parity check against the server (see
`examples/higgs_tts_rl/logprob_parity_probe.py`): right after load the trainer and
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
_GREEDY_TEMP_THRESHOLD = 1e-5


def backbone_parameter_to_checkpoint_name(name: str) -> str:
    """Map a plain ``Qwen3Model`` parameter name back to the Higgs checkpoint."""
    if name.startswith("embed_tokens."):
        return "tied.embedding.text_embedding." + name[len("embed_tokens.") :]
    if name.startswith("layers."):
        return "body.layers." + name[len("layers.") :]
    if name.startswith("norm."):
        return "body.norm." + name[len("norm.") :]
    raise ValueError(f"unsupported Higgs actor backbone parameter {name!r}")


def build_full_server_weights(backbone, fused_embed: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return full-parameter actor weights using names accepted by the server."""
    weights = {backbone_parameter_to_checkpoint_name(name): param for name, param in backbone.named_parameters()}
    weights[_FUSED_EMBED_KEY] = fused_embed
    return weights


def selected_codebook_logprobs(
    step_hidden: torch.Tensor,
    fused_embed: torch.Tensor,
    codes: torch.Tensor,
    *,
    num_codebooks: int,
    codebook_vocab: int,
    temperature: float,
    top_k: int | None = None,
) -> torch.Tensor:
    """Compute selected-action logprobs for every cell in a codebook lattice."""
    if codes.ndim != 2 or tuple(codes.shape) != (step_hidden.shape[0], num_codebooks):
        raise ValueError(f"codes shape {tuple(codes.shape)} must be {(step_hidden.shape[0], num_codebooks)}")
    expected_rows = num_codebooks * codebook_vocab
    if fused_embed.ndim != 2 or fused_embed.shape[0] != expected_rows:
        raise ValueError(f"fused_embed shape {tuple(fused_embed.shape)} must start with {expected_rows} rows")

    logits = F.linear(step_hidden.float(), fused_embed.float()).view(
        step_hidden.shape[0], num_codebooks, codebook_vocab
    )
    greedy = temperature <= _GREEDY_TEMP_THRESHOLD or top_k == 1
    effective_temperature = 1.0 if greedy else max(float(temperature), _GREEDY_TEMP_THRESHOLD)
    logprobs = torch.log_softmax(logits / effective_temperature, dim=-1)
    return logprobs.gather(-1, codes.long().unsqueeze(-1)).squeeze(-1)


def clipped_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    advantage: float | torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """Per-action clipped GRPO loss over the trainable codebook cells."""
    if current_logprobs.shape != old_logprobs.shape or current_logprobs.shape != action_mask.shape:
        raise ValueError("current logprobs, old logprobs, and action mask must have the same shape")
    action_mask = action_mask.to(device=current_logprobs.device, dtype=torch.bool)
    if not bool(action_mask.any()):
        raise ValueError("GRPO action mask contains no trainable actions")

    ratio = torch.exp(current_logprobs - old_logprobs)
    advantage_t = torch.as_tensor(advantage, dtype=ratio.dtype, device=ratio.device)
    unclipped = ratio * advantage_t
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage_t
    return -torch.minimum(unclipped, clipped)[action_mask].mean()


def _resolve_ckpt_dir(path_or_glob: str) -> str:
    if "*" in path_or_glob:
        matches = glob.glob(path_or_glob)
        if not matches:
            raise FileNotFoundError(f"no checkpoint dir matches {path_or_glob!r}")
        return matches[0]
    return path_or_glob


class HiggsTtsActor(torch.nn.Module):
    """Differentiable Higgs codec policy (Qwen3 backbone + fused codebook head)."""

    def __init__(self, ckpt_dir: str, device: str = "cuda:0", dtype=torch.bfloat16):
        super().__init__()
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

        # Fused codebook weight [N*V, D], tied between the summed input embedding
        # and all per-codebook output heads.
        self.fused_embed = torch.nn.Parameter(fused_embed.to(device=device, dtype=dtype))
        self._cb_offsets = torch.arange(self.num_codebooks, device=device) * self.codebook_vocab

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

    def codebook_logprobs(
        self,
        prompt_ids: list[int],
        codebook_tokens: list[list[int]],
        *,
        temperature: float,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Teacher-forced selected-action logprobs for all sampled codebooks."""
        device = self.device
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        codes = torch.tensor(codebook_tokens, dtype=torch.long, device=device)  # [T, N]
        T = int(codes.shape[0])
        P = int(prompt.shape[0])

        text_emb = self.backbone.get_input_embeddings()(prompt)  # [P, D]; LoRA-wrap safe
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
        return selected_codebook_logprobs(
            step_hidden,
            self.fused_embed,
            codes,
            num_codebooks=self.num_codebooks,
            codebook_vocab=self.codebook_vocab,
            temperature=temperature,
            top_k=top_k,
        )

    def codebook0_logprobs(self, prompt_ids: list[int], codebook_tokens: list[list[int]]) -> torch.Tensor:
        """Raw codebook-0 logprobs retained for server parity diagnostics."""
        return self.codebook_logprobs(prompt_ids, codebook_tokens, temperature=1.0)[:, 0]

    def full_server_weights(self) -> dict[str, torch.Tensor]:
        """Expose every full-training weight with a server-compatible name."""
        return build_full_server_weights(self.backbone, self.fused_embed)
