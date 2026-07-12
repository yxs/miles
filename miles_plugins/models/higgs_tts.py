"""Hugging Face config registration for the Higgs v3 discrete TTS policy."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig, Qwen3Config


def build_higgs_text_config(text_config: Qwen3Config | dict[str, Any]) -> Qwen3Config:
    """Normalize the v3 checkpoint's Qwen3 RoPE theta."""

    if isinstance(text_config, Qwen3Config):
        rope_parameters = dict(text_config.rope_parameters or {})
        if rope_parameters.get("rope_theta") is None:
            rope_parameters["rope_theta"] = 1_000_000
            text_config.rope_parameters = rope_parameters
        if vars(text_config).get("rope_theta") is None:
            text_config.rope_theta = 1_000_000
        return text_config
    values = dict(text_config)
    rope_parameters = dict(values.get("rope_parameters") or {})
    if rope_parameters.get("rope_theta") is None:
        rope_parameters["rope_theta"] = 1_000_000
    values["rope_parameters"] = rope_parameters
    if values.get("rope_theta") is None:
        values["rope_theta"] = 1_000_000
    normalized = Qwen3Config(**values)
    if vars(normalized).get("rope_theta") is None:
        normalized.rope_theta = 1_000_000
    return normalized


class HiggsMultimodalQwen3Config(PretrainedConfig):
    """Minimal composition config used by Miles' Megatron provider.

    The audio codec implementation is not loaded into the trainer.  Keeping its
    configuration as a mapping is sufficient to validate and construct the
    tied discrete codebook policy.
    """

    model_type = "higgs_multimodal_qwen3"
    sub_configs = {"text_config": Qwen3Config}
    is_composition = True

    def __init__(
        self,
        text_config: Qwen3Config | dict[str, Any] | None = None,
        audio_encoder_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
        self.text_config = build_higgs_text_config(text_config)
        self.audio_encoder_config = dict(audio_encoder_config or {})


__all__ = ["HiggsMultimodalQwen3Config", "build_higgs_text_config"]
