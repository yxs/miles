"""Megatron-native Higgs codec policy wrapper.

Megatron imports are intentionally local so the structured tensor contract can
be unit-tested in environments that do not install Megatron-LM.
"""

from __future__ import annotations

from typing import Any

import torch

from miles.backends.training_utils.higgs_policy import build_higgs_teacher_embeddings


def build_higgs_megatron_model(
    *,
    gpt_model_kwargs: dict[str, Any],
    num_codebooks: int,
    codebook_vocab_size: int,
):
    """Build a Qwen3 GPTModel with one tied fused codec embedding/head.

    The initial backend is deliberately TP=PP=CP=DP=1.  It still uses
    Megatron modules so DDP wrapping, optimizer construction, scheduling, and
    checkpoint lifecycle remain owned by Megatron.
    """

    from megatron.core import tensor_parallel
    from megatron.core.models.gpt import GPTModel

    if num_codebooks <= 0 or codebook_vocab_size <= 0:
        raise ValueError("Higgs codebook dimensions must be positive")

    class HiggsMegatronModel(GPTModel):
        def setup_embeddings_and_output_layer(self) -> None:
            """Install parameter attributes without tying the temporary text head.

            ``GPTModel.__init__`` calls this hook before the codec modules exist.
            The initial call must therefore avoid the virtual shared-weight
            lookup; the second call below completes setup after construction.
            """

            if self.pre_process:
                self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
            if not hasattr(self, "codec_embeddings"):
                return
            self.codec_embeddings.weight.is_embedding_or_output_parameter = True
            self.codec_embeddings.weight.zero_out_wgrad = True

        def __init__(self) -> None:
            super().__init__(**gpt_model_kwargs)
            if not self.pre_process or not self.post_process:
                raise ValueError("the initial Higgs Megatron model requires pipeline parallel size 1")
            if self.config.sequence_parallel:
                raise ValueError("the initial Higgs Megatron model does not support sequence parallelism")

            self.num_codebooks = int(num_codebooks)
            self.codebook_vocab_size = int(codebook_vocab_size)
            codec_rows = self.num_codebooks * self.codebook_vocab_size
            self.codec_embeddings = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=codec_rows,
                embedding_dim=self.config.hidden_size,
                init_method=self.config.embedding_init_method,
                config=self.config,
                tp_group=self.pg_collection.tp,
            )
            # The checkpoint ties modality input and output weights.  Allocate
            # the parameter once on codec_embeddings and pass it to this head.
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                codec_rows,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                gather_output=False,
                skip_bias_add=False,
                skip_weight_param_allocation=True,
                tp_group=self.pg_collection.tp,
            )
            self.share_embeddings_and_output_weights = True
            self.vocab_size = codec_rows
            self.setup_embeddings_and_output_layer()

        def shared_embedding_or_output_weight(self) -> torch.Tensor:
            return self.codec_embeddings.weight

        def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            *,
            higgs_prior_codes: torch.Tensor | None = None,
            higgs_codec_position_mask: torch.Tensor | None = None,
            higgs_sequence_mask: torch.Tensor | None = None,
            higgs_prediction_positions: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            required = (
                higgs_prior_codes,
                higgs_codec_position_mask,
                higgs_sequence_mask,
                higgs_prediction_positions,
            )
            if any(value is None for value in required):
                raise ValueError("HiggsMegatronModel requires the complete structured teacher-forcing batch")
            if input_ids.ndim != 2:
                raise ValueError("Higgs input_ids must have shape [batch, sequence]")
            if higgs_sequence_mask.shape != input_ids.shape:
                raise ValueError("Higgs sequence mask must match input_ids")
            if higgs_prediction_positions.ndim != 2 or higgs_prediction_positions.shape[0] != input_ids.shape[0]:
                raise ValueError("Higgs prediction positions must have shape [batch, action_rows]")

            text_embeddings = self.embedding.word_embeddings(input_ids)
            embeddings = build_higgs_teacher_embeddings(
                text_embeddings,
                self.codec_embeddings.weight,
                higgs_prior_codes,
                higgs_codec_position_mask,
            )
            decoder_input = embeddings.transpose(0, 1).contiguous()
            if self.config.fp32_residual_connection:
                decoder_input = decoder_input.float()
            decoder_input = self.embedding.embedding_dropout(decoder_input)

            logits = super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=None,
                packed_seq_params=None,
                padding_mask=~higgs_sequence_mask,
                **kwargs,
            )
            if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
                raise RuntimeError(f"unexpected Higgs Megatron logits shape {tuple(logits.shape)}")
            if logits.shape[-1] != self.num_codebooks * self.codebook_vocab_size:
                raise RuntimeError("Higgs codec head returned the wrong flattened vocabulary size")

            gather_index = higgs_prediction_positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
            action_logits = logits.gather(dim=1, index=gather_index)
            return action_logits.reshape(
                input_ids.shape[0],
                higgs_prediction_positions.shape[1],
                self.num_codebooks,
                self.codebook_vocab_size,
            )

    return HiggsMegatronModel()


def higgs_model_forward_kwargs(batch: Any) -> dict[str, torch.Tensor]:
    """Translate a ``HiggsPolicyBatch`` into the model's explicit API."""

    return {
        "input_ids": batch.input_ids,
        "position_ids": None,
        "attention_mask": None,
        "higgs_prior_codes": batch.prior_codes,
        "higgs_codec_position_mask": batch.codec_position_mask,
        "higgs_sequence_mask": batch.sequence_mask,
        "higgs_prediction_positions": batch.prediction_positions,
    }


__all__ = ["build_higgs_megatron_model", "higgs_model_forward_kwargs"]
