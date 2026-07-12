import json
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.higgs_policy import (
    build_higgs_teacher_embeddings,
    collate_higgs_policy_batch,
    get_higgs_joint_log_probs,
    higgs_joint_policy_loss,
    higgs_policy_loss_from_logits,
    selected_higgs_logprobs,
    validate_higgs_hf_config,
    validate_higgs_logprob_parity,
    validate_higgs_single_device_config,
    validate_higgs_weight_versions,
)


class _Trace:
    def __init__(self, actions, logprobs, mask, *, vocab_size=4):
        self.model_family = "higgs_tts"
        self.action_streams = [
            SimpleNamespace(
                name="higgs_codes",
                action_type="multi_discrete",
                layout="time_codebook",
                shape=[len(actions), len(actions[0])],
                vocab_size=vocab_size,
                actions=actions,
                policy_logprobs=logprobs,
                action_mask=mask,
            )
        ]

    def validate(self):
        return None


def _config(**overrides):
    values = {
        "structured_policy_model_family": "higgs_tts",
        "hf_checkpoint": "bosonai/higgs-audio-v3-tts-4b",
        "train_backend": "megatron",
        "qkv_format": "bshd",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": None,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "advantage_estimator": "grpo",
        "bf16": True,
        "fp16": False,
        "true_on_policy_mode": False,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "masked_softmax_fusion": False,
        "vocab_size": 151936,
        "padded_vocab_size": 151936,
        "group_query_attention": True,
        "num_query_groups": 8,
        "kv_channels": 128,
        "qk_layernorm": True,
        "swiglu": True,
        "add_bias_linear": False,
        "untie_embeddings_and_output_weights": False,
        "normalization": "RMSNorm",
        "use_rotary_position_embeddings": True,
        "rotary_percent": 1.0,
        "rotary_interleaved": False,
        "use_rope_scaling": False,
        "use_rollout_logprobs": True,
        "megatron_to_hf_mode": "raw",
        "lora_rank": 0,
        "kl_coef": 0.0,
        "compute_advantages_and_returns": True,
        "debug_train_only": False,
        "higgs_logprob_parity_atol": 0.05,
        "rewards_normalization": True,
        "n_samples_per_prompt": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _hf_config(**audio_overrides):
    audio = {
        "encoder_type": "discrete",
        "num_codebooks": 8,
        "vocab_size": 1026,
        "out_dim": 2560,
        "tie_word_embeddings": True,
        "use_delay_pattern": True,
    }
    audio.update(audio_overrides)
    return {
        "model_type": "higgs_multimodal_qwen3",
        "architectures": ["HiggsMultimodalQwen3ForConditionalGeneration"],
        "audio_encoder_config": audio,
        "text_config": {
            "model_type": "qwen3",
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 9728,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "hidden_act": "silu",
            "tie_word_embeddings": True,
            "attention_dropout": 0.0,
            "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
            "dtype": "bfloat16",
        },
    }


def test_collation_teacher_forces_complete_prior_rows_including_forced_cells():
    trace = _Trace(
        actions=[[1, 3], [2, 0], [3, 1]],
        logprobs=[[-0.2, 0.0], [-0.3, -0.4], [-0.5, 0.0]],
        mask=[[True, False], [True, True], [True, False]],
    )

    batch = collate_higgs_policy_batch([[10, 11, 12]], [trace], advantages=[0.75])

    assert batch.input_ids.tolist() == [[10, 11, 12, 0, 0]]
    assert batch.codec_position_mask.tolist() == [[False, False, False, True, True]]
    assert batch.prior_codes[0, 3:].tolist() == [[1, 3], [2, 0]]
    assert batch.prediction_positions.tolist() == [[2, 3, 4]]
    assert batch.actions.tolist() == [[[1, 3], [2, 0], [3, 1]]]
    assert torch.allclose(
        batch.old_cell_logprobs,
        torch.tensor([[[-0.2, 0.0], [-0.3, -0.4], [-0.5, 0.0]]]),
    )
    assert torch.allclose(batch.old_joint_logprobs, torch.tensor([[-0.2, -0.7, -0.5]]))
    assert torch.allclose(batch.advantages, torch.full((1, 3), 0.75))


def test_collation_right_pads_variable_prompt_and_action_lengths():
    first = _Trace([[0, 1]], [[-0.1, -0.2]], [[True, True]])
    second = _Trace(
        [[1, 2], [2, 3], [0, 1]],
        [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
        [[True, True], [True, True], [True, True]],
    )

    batch = collate_higgs_policy_batch([[7, 8, 9], [4]], [first, second])

    assert batch.input_ids.shape == (2, 3)
    assert batch.sequence_mask.tolist() == [[True, True, True], [True, True, True]]
    assert batch.prediction_positions.tolist() == [[2, 0, 0], [0, 1, 2]]
    assert batch.row_mask.tolist() == [[True, False, False], [True, True, True]]


def test_collation_uses_recomputed_joint_logprobs_as_old_policy_baseline():
    trace = _Trace(
        [[0, 1], [1, 2]],
        [[-1.0, -2.0], [-3.0, -4.0]],
        [[True, True], [True, True]],
        vocab_size=3,
    )

    batch = collate_higgs_policy_batch(
        [[7, 8]],
        [trace],
        advantages=[1.0],
        old_policy_joint_logprobs=[torch.tensor([-2.75, -6.5])],
    )

    assert torch.allclose(batch.old_cell_logprobs.sum(-1), torch.tensor([[-3.0, -7.0]]))
    assert torch.allclose(batch.old_joint_logprobs, torch.tensor([[-2.75, -6.5]]))


@pytest.mark.parametrize(
    "baseline",
    [
        [torch.tensor([-1.0])],
        [torch.tensor([-1.0, float("nan")])],
    ],
)
def test_collation_rejects_invalid_recomputed_old_policy_baseline(baseline):
    trace = _Trace(
        [[0, 1], [1, 2]],
        [[-1.0, -2.0], [-3.0, -4.0]],
        [[True, True], [True, True]],
        vocab_size=3,
    )

    with pytest.raises(ValueError, match="old-policy joint logprob"):
        collate_higgs_policy_batch(
            [[7, 8]],
            [trace],
            old_policy_joint_logprobs=baseline,
        )


def test_teacher_embedding_uses_channel_offsets_and_sums_codebooks():
    text_embeddings = torch.tensor([[[100.0], [200.0]]])
    # Two codebooks, vocabulary three.  Row [1, 2] maps to weights 1 and 5.
    codec_weight = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    prior_codes = torch.tensor([[[0, 0], [1, 2]]])
    codec_mask = torch.tensor([[False, True]])

    actual = build_higgs_teacher_embeddings(text_embeddings, codec_weight, prior_codes, codec_mask)

    assert actual.tolist() == [[[100.0], [6.0]]]


def test_selected_logprobs_use_fp32_full_vocabulary_and_zero_forced_cells():
    logits = torch.tensor(
        [[[[2.0, 1.0, -1.0], [0.0, 3.0, 1.0]], [[-1.0, 0.0, 2.0], [1.0, 2.0, 3.0]]]],
        dtype=torch.bfloat16,
    )
    actions = torch.tensor([[[0, 1], [2, 0]]])
    mask = torch.tensor([[[True, True], [True, False]]])

    cell, joint, row_mask = selected_higgs_logprobs(logits, actions, mask)

    expected = torch.log_softmax(logits.float(), dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    expected = torch.where(mask, expected, torch.zeros_like(expected))
    assert cell.dtype == torch.float32
    assert torch.allclose(cell, expected)
    assert torch.allclose(joint, expected.sum(-1))
    assert row_mask.tolist() == [[True, True]]
    assert cell[0, 1, 1].item() == 0.0


def test_grpo_clips_the_joint_row_ratio_not_individual_codebook_ratios():
    # Factor ratios 2.0 and 0.55 would be clipped independently, but their
    # joint ratio is 1.1 and must remain unclipped.
    current = torch.tensor([[torch.log(torch.tensor(1.1))]])
    old = torch.zeros_like(current)

    loss, metrics = higgs_joint_policy_loss(
        current,
        old,
        advantages=torch.ones_like(current),
        row_mask=torch.ones_like(current, dtype=torch.bool),
        eps_clip=0.2,
    )

    assert torch.allclose(loss, torch.tensor(-1.1))
    assert metrics["pg_clipfrac"].item() == 0.0


def test_higgs_loss_backpropagates_through_each_active_codebook_only():
    trace = _Trace(
        actions=[[0, 1], [2, 0]],
        logprobs=[[-1.0, -1.0], [-1.0, 0.0]],
        mask=[[True, True], [True, False]],
        vocab_size=3,
    )
    batch = collate_higgs_policy_batch([[5, 6]], [trace], advantages=[1.0])
    logits = torch.randn(1, 2, 2, 3, requires_grad=True)
    _, initial_joint, _ = selected_higgs_logprobs(logits.detach(), batch.actions, batch.action_mask)
    batch.old_joint_logprobs.copy_(initial_joint)

    loss, _, _ = higgs_policy_loss_from_logits(logits, batch, eps_clip=0.2)
    loss.backward()

    assert logits.grad is not None
    assert bool((logits.grad[0, 0, 0] != 0).any())
    assert bool((logits.grad[0, 0, 1] != 0).any())
    assert bool((logits.grad[0, 1, 0] != 0).any())
    assert torch.count_nonzero(logits.grad[0, 1, 1]).item() == 0


@pytest.mark.parametrize("advantage", [float("nan"), float("inf"), float("-inf")])
def test_higgs_loss_rejects_nonfinite_active_advantages(advantage):
    current = torch.zeros((1, 1))
    with pytest.raises(RuntimeError, match="advantages must be finite"):
        higgs_joint_policy_loss(
            current,
            current,
            advantages=torch.tensor([[advantage]]),
            row_mask=torch.ones_like(current, dtype=torch.bool),
            eps_clip=0.2,
        )


def test_forward_logprob_collection_returns_joint_rows_per_sample():
    trace = _Trace([[0, 1], [1, 0]], [[-1.0, -1.0], [-1.0, -1.0]], [[True, True], [True, True]], vocab_size=2)
    batch = collate_higgs_policy_batch([[4]], [trace])
    logits = torch.tensor([[[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [2.0, 0.0]]]])

    result = get_higgs_joint_log_probs(logits, higgs_batch=batch)

    _, expected, _ = selected_higgs_logprobs(logits, batch.actions, batch.action_mask)
    assert len(result["log_probs"]) == 1
    assert torch.allclose(result["log_probs"][0], expected[0])


def test_training_batch_requires_and_uses_preupdate_megatron_logprobs(monkeypatch):
    from miles.backends.training_utils import data as data_module

    trace = _Trace(
        [[0, 1], [1, 2]],
        [[-1.0, -2.0], [-3.0, -4.0]],
        [[True, True], [True, True]],
        vocab_size=3,
    )
    rollout_data = {
        "tokens": [torch.tensor([7, 8])],
        "action_traces": [trace],
        "advantages": [torch.ones(2)],
        "log_probs": [torch.tensor([-2.75, -6.5])],
    }

    class Iterator:
        def get_next(self, keys):
            return {key: rollout_data.get(key) for key in keys}

    monkeypatch.setattr(
        data_module,
        "get_parallel_state",
        lambda: SimpleNamespace(intra_dp=SimpleNamespace(size=1)),
    )
    args = _config(
        higgs_num_codebooks=2,
        higgs_codebook_vocab_size=3,
        seq_length=16,
    )

    batch = data_module.get_higgs_batch(Iterator(), args, require_advantages=True)

    assert torch.allclose(batch.old_joint_logprobs, torch.tensor([[-2.75, -6.5]]))

    rollout_data["log_probs"] = None
    with pytest.raises(ValueError, match="pre-update Megatron joint logprobs"):
        data_module.get_higgs_batch(Iterator(), args, require_advantages=True)


def test_structured_grpo_advantage_is_broadcast_to_action_rows(monkeypatch):
    from miles.backends.training_utils import loss as loss_module

    trace = _Trace(
        [[0, 1], [1, 2], [2, 0]],
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]],
        [[True, True], [True, True], [True, True]],
        vocab_size=3,
    )
    monkeypatch.setattr(
        loss_module,
        "get_parallel_state",
        lambda: SimpleNamespace(intra_dp=SimpleNamespace(size=1)),
    )
    rollout_data = {
        "action_traces": [trace],
        "rewards": [0.625],
        "tokens": [torch.tensor([4, 5])],
    }

    loss_module.compute_advantages_and_returns(_config(), rollout_data)

    assert torch.allclose(rollout_data["advantages"][0], torch.full((3,), 0.625))
    assert torch.equal(rollout_data["returns"][0], rollout_data["advantages"][0])


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf")])
def test_structured_grpo_rejects_nonfinite_rewards(monkeypatch, reward):
    from miles.backends.training_utils import loss as loss_module

    trace = _Trace([[0, 1]], [[-1.0, -1.0]], [[True, True]])
    monkeypatch.setattr(
        loss_module,
        "get_parallel_state",
        lambda: SimpleNamespace(intra_dp=SimpleNamespace(size=1)),
    )
    with pytest.raises(ValueError, match="reward must be finite"):
        loss_module.compute_advantages_and_returns(
            _config(),
            {
                "action_traces": [trace],
                "rewards": [reward],
                "tokens": [torch.tensor([4, 5])],
            },
        )


def test_structured_rollout_logging_uses_action_rows_not_empty_text_masks():
    from miles.backends.training_utils.log_utils import _get_higgs_rollout_log_dict

    trace = _Trace(
        [[0, 1], [1, 2], [2, 0]],
        [[-1.0, -2.0], [-3.0, 0.0], [0.0, 0.0]],
        [[True, True], [True, False], [False, False]],
        vocab_size=3,
    )
    metrics = _get_higgs_rollout_log_dict(
        {
            "action_traces": [trace],
            "advantages": [torch.tensor([0.5, 1.5, 999.0])],
            "returns": [torch.tensor([0.5, 1.5, 999.0])],
            "rewards": [0.75],
            "response_lengths": [0],
            "total_lengths": [2],
            "loss_masks": [torch.empty(0)],
        }
    )

    assert metrics["action_rows"] == 3.0
    assert metrics["active_action_rows"] == 2.0
    assert metrics["sampled_action_cells"] == 3.0
    assert metrics["rollout_joint_log_probs"] == -3.0
    assert metrics["advantages"] == 1.0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size"),
        ({"actor_num_gpus_per_node": 2}, "actor_num_gpus_per_node"),
        ({"qkv_format": "thd"}, "qkv_format"),
        ({"bf16": False}, "bf16"),
        ({"fp16": True}, "fp16"),
        ({"true_on_policy_mode": True}, "true_on_policy_mode"),
        ({"hidden_dropout": 0.1}, "hidden_dropout"),
        ({"attention_dropout": 0.1}, "attention_dropout"),
        ({"vocab_size": 151727}, "vocab_size"),
        ({"padded_vocab_size": 151744}, "padded_vocab_size"),
        ({"group_query_attention": False}, "group_query_attention"),
        ({"num_query_groups": 32}, "num_query_groups"),
        ({"kv_channels": 80}, "kv_channels"),
        ({"qk_layernorm": False}, "qk_layernorm"),
        ({"swiglu": False}, "swiglu"),
        ({"add_bias_linear": True}, "add_bias_linear"),
        ({"untie_embeddings_and_output_weights": True}, "untie_embeddings_and_output_weights"),
        ({"normalization": "LayerNorm"}, "normalization"),
        ({"use_rotary_position_embeddings": False}, "use_rotary_position_embeddings"),
        ({"rotary_percent": 0.5}, "rotary_percent"),
        ({"rotary_interleaved": True}, "rotary_interleaved"),
        ({"use_rope_scaling": True}, "use_rope_scaling"),
        ({"masked_softmax_fusion": True}, "masked_softmax_fusion"),
        ({"num_experts": 8}, "num_experts"),
        ({"mtp_num_layers": 1}, "mtp_num_layers"),
        ({"spec": ["custom", "provider"]}, "layer specs"),
        ({"sequence_parallel": True}, "sequence_parallel"),
        ({"megatron_to_hf_mode": "bridge"}, "checkpoint mapping"),
        ({"hf_checkpoint": None}, "concrete Higgs"),
        ({"save_hf": "/tmp/higgs-hf"}, "standalone Higgs HF snapshot export"),
        ({"lora_rank": 8}, "lora_rank"),
        ({"higgs_logprob_parity_atol": None}, "measured tolerance"),
        ({"higgs_logprob_parity_atol": "0.05"}, "measured tolerance"),
        ({"rewards_normalization": False}, "rewards_normalization"),
        ({"n_samples_per_prompt": 1}, "nonzero grouped GRPO signal"),
    ],
)
def test_single_device_config_fails_closed(override, message):
    with pytest.raises(ValueError, match=message):
        validate_higgs_single_device_config(_config(**override), data_parallel_size=1)


def test_single_device_config_accepts_naive_megatron_profile():
    validate_higgs_single_device_config(_config(), data_parallel_size=1)


def test_higgs_checkpoint_config_matches_v3_tts_policy_architecture():
    validate_higgs_hf_config(_hf_config(), num_codebooks=8, codebook_vocab_size=1026)


def test_higgs_qwen3_config_normalizes_explicit_null_rope_theta():
    from miles_plugins.models.higgs_tts import build_higgs_text_config

    text_config = dict(_hf_config()["text_config"])
    text_config["rope_parameters"] = {"rope_theta": None, "rope_type": "default"}

    normalized = build_higgs_text_config(text_config)

    assert normalized.rope_parameters["rope_theta"] == 1_000_000


def test_higgs_qwen3_config_normalizes_legacy_top_level_null_rope_theta():
    from miles_plugins.models.higgs_tts import build_higgs_text_config

    text_config = dict(_hf_config()["text_config"])
    text_config.pop("rope_parameters")
    text_config["rope_theta"] = None

    normalized = build_higgs_text_config(text_config)

    assert normalized.rope_theta == 1_000_000
    assert normalized.rope_parameters["rope_theta"] == 1_000_000


def test_miles_hf_loader_registers_higgs_composition_config(tmp_path):
    from miles.utils.hf_config import load_hf_config

    config = _hf_config()
    config["text_config"]["rope_parameters"]["rope_theta"] = None
    (tmp_path / "config.json").write_text(json.dumps(config))

    loaded = load_hf_config(str(tmp_path))

    assert loaded.model_type == "higgs_multimodal_qwen3"
    assert loaded.text_config.hidden_size == 2560
    assert loaded.text_config.rope_parameters["rope_theta"] == 1_000_000
    validate_higgs_hf_config(loaded, num_codebooks=8, codebook_vocab_size=1026)


@pytest.mark.parametrize(
    "audio_override",
    [
        {"num_codebooks": 12},
        {"vocab_size": 1024},
        {"tie_word_embeddings": False},
        {"encoder_type": "whisper"},
    ],
)
def test_higgs_checkpoint_config_rejects_a_different_audio_policy(audio_override):
    with pytest.raises(ValueError, match="unsupported Higgs checkpoint"):
        validate_higgs_hf_config(_hf_config(**audio_override), num_codebooks=8, codebook_vocab_size=1026)


def test_logprob_parity_compares_joint_active_rows_and_accepts_within_tolerance():
    trace = _Trace(
        [[0, 1], [1, 2], [2, 0]],
        [[-1.0, -2.0], [-3.0, 0.0], [0.0, 0.0]],
        [[True, True], [True, False], [False, False]],
        vocab_size=3,
    )

    metrics = validate_higgs_logprob_parity(
        [trace],
        [torch.tensor([-2.99, -3.02, 123.0])],
        atol=0.03,
    )

    assert metrics["max_abs_diff"] == pytest.approx(0.02)
    assert metrics["mean_abs_diff"] == pytest.approx(0.015)
    assert metrics["within_tolerance"] is True


def test_logprob_parity_reports_finite_joint_row_mismatch_without_blocking_training():
    trace = _Trace([[0, 1]], [[-1.0, -2.0]], [[True, True]], vocab_size=2)

    metrics = validate_higgs_logprob_parity([trace], [torch.tensor([-2.8])], atol=0.1)

    assert metrics == {
        "max_abs_diff": pytest.approx(0.2),
        "mean_abs_diff": pytest.approx(0.2),
        "within_tolerance": False,
    }


def test_weight_provenance_requires_one_current_equal_version_per_sample():
    assert validate_higgs_weight_versions([["7"], ["7"]], trainer_weight_version=7) == "7"


def test_weight_provenance_accepts_server_default_only_for_initial_trainer_version():
    assert validate_higgs_weight_versions([["default"], ["default"]], trainer_weight_version=0) == "default"

    with pytest.raises(RuntimeError, match="trainer expects '1'"):
        validate_higgs_weight_versions([["default"]], trainer_weight_version=1)


@pytest.mark.parametrize(
    ("versions", "message"),
    [
        (None, "requires one rollout"),
        ([[]], "exactly one"),
        ([["1", "2"]], "exactly one"),
        ([["1"], ["2"]], "mixes rollout"),
        ([["1"]], "trainer expects '2'"),
    ],
)
def test_weight_provenance_rejects_missing_mixed_or_stale_versions(versions, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        validate_higgs_weight_versions(versions, trainer_weight_version=2)
