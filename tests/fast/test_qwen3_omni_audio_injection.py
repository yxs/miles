"""Frozen-audio-encoder injection for the Qwen3-Omni thinker text backbone."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from miles_plugins.models.qwen3_omni_thinker import (
    compute_audio_embeddings,
    install_audio_injection,
    load_frozen_audio_encoder,
    resolve_audio_token_id,
    scatter_audio_embeddings,
)

AUDIO_TOKEN_ID = 151675
HIDDEN = 16
MEL = 8


def _tiny_encoder_config() -> Qwen3OmniMoeAudioEncoderConfig:
    return Qwen3OmniMoeAudioEncoderConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=MEL,
        output_dim=HIDDEN,
        downsample_hidden_size=24,
    )


@pytest.fixture(scope="module")
def tiny_encoder() -> Qwen3OmniMoeAudioEncoder:
    torch.manual_seed(0)
    return Qwen3OmniMoeAudioEncoder(_tiny_encoder_config()).eval()


def _audio_batch(tiny_encoder, num_frames: int = 40):
    """input_features [1, mel, T] + full attention mask, and the token count it expands to."""
    torch.manual_seed(1)
    input_features = torch.randn(1, MEL, num_frames)
    feature_attention_mask = torch.ones(1, num_frames, dtype=torch.long)
    # the module-level formula is what the processor uses for placeholder counts
    out_lens = _get_feat_extract_output_lengths(torch.tensor([num_frames]))
    return input_features, feature_attention_mask, int(out_lens.item())


class _StubGPT(torch.nn.Module):
    """Duck-typed mcore GPT chunk: embedding + forward, [s, b, h] layout."""

    def __init__(self, pre_process: bool = True):
        super().__init__()
        self.pre_process = pre_process
        torch.manual_seed(2)
        self.word_embeddings = torch.nn.Embedding(152064, HIDDEN)
        self.captured = None

    def embedding(self, input_ids, position_ids=None):
        return self.word_embeddings(input_ids).transpose(0, 1)  # [s, b, h]

    def forward(self, **kwargs):
        self.captured = kwargs
        return torch.zeros(1)


def _args(**overrides):
    defaults = dict(sequence_parallel=False, context_parallel_size=1, qwen3_omni_audio_encoder_path="/nonexistent")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _install(model, tiny_encoder, **arg_overrides):
    return install_audio_injection(
        model,
        _args(**arg_overrides),
        encoder_loader=lambda device, dtype: tiny_encoder,
        audio_token_id=AUDIO_TOKEN_ID,
    )


def _packed_input_ids(audio_token_counts: list[int], text_len: int = 3) -> torch.Tensor:
    """[1, s] packed sequence: text_len text tokens then N audio tokens, per sample."""
    ids = []
    for n in audio_token_counts:
        ids += [7] * text_len + [AUDIO_TOKEN_ID] * n
    return torch.tensor([ids], dtype=torch.long)


# ------------------------------- scatter -------------------------------


def test_scatter_replaces_audio_positions_in_order():
    input_ids = _packed_input_ids([2, 3])
    s = input_ids.size(1)
    hidden = torch.arange(s * HIDDEN, dtype=torch.float32).reshape(s, 1, HIDDEN)
    audio_embeds = -torch.arange(1, 5 * HIDDEN + 1, dtype=torch.float32).reshape(5, HIDDEN)

    out = scatter_audio_embeddings(hidden, input_ids, audio_embeds, AUDIO_TOKEN_ID)

    mask = (input_ids[0] == AUDIO_TOKEN_ID).tolist()
    audio_row = 0
    for pos, is_audio in enumerate(mask):
        if is_audio:
            assert torch.equal(out[pos, 0], audio_embeds[audio_row]), pos
            audio_row += 1
        else:
            assert torch.equal(out[pos, 0], hidden[pos, 0]), pos
    assert audio_row == 5
    assert torch.equal(hidden[3, 0], torch.arange(3 * HIDDEN, 4 * HIDDEN, dtype=torch.float32)), "input mutated"


def test_scatter_count_mismatch_fails_loud():
    input_ids = _packed_input_ids([2])
    hidden = torch.zeros(input_ids.size(1), 1, HIDDEN)
    with pytest.raises(AssertionError, match="audio"):
        scatter_audio_embeddings(hidden, input_ids, torch.zeros(3, HIDDEN), AUDIO_TOKEN_ID)


def test_scatter_requires_packed_batch_dim_one():
    input_ids = torch.full((2, 4), AUDIO_TOKEN_ID)
    hidden = torch.zeros(4, 2, HIDDEN)
    with pytest.raises(AssertionError, match="packed"):
        scatter_audio_embeddings(hidden, input_ids, torch.zeros(8, HIDDEN), AUDIO_TOKEN_ID)


# ------------------------------- encoder plumbing -------------------------------


def test_compute_audio_embeddings_matches_direct_encoder_call(tiny_encoder):
    input_features, feature_attention_mask, n_tokens = _audio_batch(tiny_encoder)

    embeds = compute_audio_embeddings(tiny_encoder, input_features, feature_attention_mask, None)

    assert embeds.shape == (n_tokens, HIDDEN)
    lengths = feature_attention_mask.sum(dim=1)
    flat = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    expected = tiny_encoder(flat, feature_lens=lengths).last_hidden_state
    assert torch.allclose(embeds, expected)


def test_compute_audio_embeddings_concatenates_multiple_audios(tiny_encoder):
    torch.manual_seed(3)
    input_features = torch.randn(2, MEL, 40)
    feature_attention_mask = torch.ones(2, 40, dtype=torch.long)
    feature_attention_mask[1, 24:] = 0  # second audio shorter

    embeds = compute_audio_embeddings(tiny_encoder, input_features, feature_attention_mask, None)

    out_lens = _get_feat_extract_output_lengths(torch.tensor([40, 24]))
    assert embeds.shape == (int(out_lens.sum()), HIDDEN)


# ------------------------------- forward patch -------------------------------


def test_install_injects_decoder_input(tiny_encoder):
    model = _StubGPT()
    _install(model, tiny_encoder)
    input_features, feature_attention_mask, n_tokens = _audio_batch(tiny_encoder)
    input_ids = _packed_input_ids([n_tokens])

    model.forward(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=None,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
    )

    captured = model.captured
    assert "input_features" not in captured and "feature_attention_mask" not in captured
    decoder_input = captured["decoder_input"]
    assert decoder_input.shape == (input_ids.size(1), 1, HIDDEN)
    expected_embeds = compute_audio_embeddings(tiny_encoder, input_features, feature_attention_mask, None)
    audio_positions = (input_ids[0] == AUDIO_TOKEN_ID).nonzero().squeeze(1)
    assert torch.allclose(decoder_input[audio_positions, 0], expected_embeds.to(decoder_input.dtype))
    text_positions = (input_ids[0] != AUDIO_TOKEN_ID).nonzero().squeeze(1)
    ref = model.embedding(input_ids)
    assert torch.equal(decoder_input[text_positions, 0], ref[text_positions, 0])


def test_install_passthrough_without_audio(tiny_encoder):
    model = _StubGPT()

    def _fail_loader(device, dtype):
        raise AssertionError("encoder must not load on text-only batches")

    install_audio_injection(model, _args(), encoder_loader=_fail_loader, audio_token_id=AUDIO_TOKEN_ID)
    model.forward(input_ids=_packed_input_ids([0]), position_ids=None, attention_mask=None)
    assert "decoder_input" not in model.captured

    model.forward(input_ids=_packed_input_ids([0]), input_features=None)
    assert "decoder_input" not in model.captured
    assert "input_features" not in model.captured


def test_install_rejects_vision_keys(tiny_encoder):
    model = _StubGPT()
    _install(model, tiny_encoder)
    with pytest.raises(AssertionError, match="pixel_values"):
        model.forward(input_ids=_packed_input_ids([1]), pixel_values=torch.zeros(1, 3))


def test_install_non_pre_process_swallows_audio_kwargs(tiny_encoder):
    model = _StubGPT(pre_process=False)
    _install(model, tiny_encoder)
    input_features, feature_attention_mask, n_tokens = _audio_batch(tiny_encoder)

    model.forward(
        input_ids=_packed_input_ids([n_tokens]),
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
    )

    assert "input_features" not in model.captured
    assert "decoder_input" not in model.captured


def test_install_rejects_sequence_parallel_with_audio(tiny_encoder):
    model = _StubGPT()
    _install(model, tiny_encoder, sequence_parallel=True)
    input_features, feature_attention_mask, n_tokens = _audio_batch(tiny_encoder)
    with pytest.raises(AssertionError, match="sequence.parallel"):
        model.forward(
            input_ids=_packed_input_ids([n_tokens]),
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
    # text-only batches stay allowed under SP
    model.forward(input_ids=_packed_input_ids([0]))
    assert model.captured is not None


# ------------------------------- checkpoint loader -------------------------------


def test_load_frozen_audio_encoder_from_omni_checkpoint(tmp_path, tiny_encoder):
    ckpt = tmp_path / "omni"
    ckpt.mkdir()
    config = {
        "model_type": "qwen3_omni_moe",
        "thinker_config": {
            "audio_token_id": AUDIO_TOKEN_ID,
            "audio_config": _tiny_encoder_config().to_dict(),
        },
    }
    with open(ckpt / "config.json", "w") as f:
        json.dump(config, f)
    tensors = {f"thinker.audio_tower.{k}": v for k, v in tiny_encoder.state_dict().items()}
    tensors["thinker.model.embed_tokens.weight"] = torch.zeros(4, 4)  # decoys the loader must skip
    tensors["talker.model.norm.weight"] = torch.zeros(4)
    save_file(tensors, ckpt / "model.safetensors", metadata={"format": "pt"})

    encoder = load_frozen_audio_encoder(ckpt, device=torch.device("cpu"), dtype=torch.float32)

    assert not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())
    input_features, feature_attention_mask, _ = _audio_batch(tiny_encoder)
    expected = compute_audio_embeddings(tiny_encoder, input_features, feature_attention_mask, None)
    actual = compute_audio_embeddings(encoder, input_features, feature_attention_mask, None)
    assert torch.allclose(actual, expected, atol=1e-5)

    assert resolve_audio_token_id(config) == AUDIO_TOKEN_ID
