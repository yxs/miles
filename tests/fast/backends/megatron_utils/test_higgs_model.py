import sys
from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as F

from miles.backends.megatron_utils.higgs_model import build_higgs_megatron_model


class _FakeVocabParallelEmbedding(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **_):
        super().__init__(num_embeddings, embedding_dim)


class _FakeColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, *, skip_weight_param_allocation, **_):
        super().__init__()
        assert skip_weight_param_allocation
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_parallel = False

    def forward(self, input_, weight=None, runtime_gather_output=None):
        assert weight is not None
        assert runtime_gather_output is None
        return F.linear(input_, weight), None


class _FakeGPTModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs["config"]
        self.pre_process = kwargs["pre_process"]
        self.post_process = kwargs["post_process"]
        self.pg_collection = SimpleNamespace(tp=object())
        self.embedding = SimpleNamespace(
            word_embeddings=torch.nn.Embedding(kwargs["vocab_size"], self.config.hidden_size),
            embedding_dropout=torch.nn.Identity(),
        )
        self.output_layer = torch.nn.Linear(self.config.hidden_size, kwargs["vocab_size"], bias=False)
        self.share_embeddings_and_output_weights = kwargs["share_embeddings_and_output_weights"]
        self.vocab_size = kwargs["vocab_size"]
        self.setup_embeddings_and_output_layer()

    def setup_embeddings_and_output_layer(self):
        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        if self.share_embeddings_and_output_weights:
            self.shared_embedding_or_output_weight().zero_out_wgrad = True

    def forward(self, *, decoder_input, **_):
        logits, _ = self.output_layer(decoder_input, weight=self.shared_embedding_or_output_weight())
        return logits.transpose(0, 1).contiguous()


def _install_fake_megatron(monkeypatch):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    models = ModuleType("megatron.core.models")
    gpt = ModuleType("megatron.core.models.gpt")
    core.tensor_parallel = SimpleNamespace(
        VocabParallelEmbedding=_FakeVocabParallelEmbedding,
        ColumnParallelLinear=_FakeColumnParallelLinear,
    )
    gpt.GPTModel = _FakeGPTModel
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.models", models)
    monkeypatch.setitem(sys.modules, "megatron.core.models.gpt", gpt)


def test_megatron_wrapper_teacher_forces_prior_rows_and_uses_tied_codec_head(monkeypatch):
    _install_fake_megatron(monkeypatch)
    config = SimpleNamespace(
        hidden_size=2,
        sequence_parallel=False,
        embedding_init_method=lambda weight: None,
        init_method=lambda weight: None,
        fp32_residual_connection=False,
    )
    model = build_higgs_megatron_model(
        gpt_model_kwargs={
            "config": config,
            "transformer_layer_spec": object(),
            "vocab_size": 8,
            "max_sequence_length": 16,
            "pre_process": True,
            "post_process": True,
            "share_embeddings_and_output_weights": True,
        },
        num_codebooks=2,
        codebook_vocab_size=3,
    )
    with torch.no_grad():
        model.embedding.word_embeddings.weight.zero_()
        model.embedding.word_embeddings.weight[4] = torch.tensor([1.0, 0.0])
        model.embedding.word_embeddings.weight[5] = torch.tensor([0.0, 1.0])
        model.codec_embeddings.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [0.0, 3.0],
                ]
            )
        )

    logits = model(
        input_ids=torch.tensor([[4, 5, 0]]),
        higgs_prior_codes=torch.tensor([[[0, 0], [0, 0], [1, 2]]]),
        higgs_codec_position_mask=torch.tensor([[False, False, True]]),
        higgs_sequence_mask=torch.tensor([[True, True, True]]),
        higgs_prediction_positions=torch.tensor([[1, 2]]),
    )

    expected_hidden = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    expected = F.linear(expected_hidden, model.codec_embeddings.weight).reshape(2, 2, 3)
    assert logits.shape == (1, 2, 2, 3)
    assert torch.allclose(logits[0], expected)
    assert model.shared_embedding_or_output_weight() is model.codec_embeddings.weight
    assert model.codec_embeddings.weight.is_embedding_or_output_parameter is True
    assert model.codec_embeddings.weight.zero_out_wgrad is True

    logits.sum().backward()
    assert model.codec_embeddings.weight.grad is not None
