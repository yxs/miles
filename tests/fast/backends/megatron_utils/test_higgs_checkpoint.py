import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.ci.ci_register import register_cpu_ci

from miles.backends.megatron_utils import higgs_checkpoint

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

_CACHED_HIGGS_REVISION = Path(
    "/root/.cache/huggingface/hub/models--bosonai--higgs-audio-v3-tts-4b/"
    "snapshots/7556c17e05201fccd9c8cc120bc216dcc7b5d561"
)


def _load_higgs_converter():
    package_name = "_miles_higgs_converter_test"
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    base = Path(__file__).resolve().parents[4] / "miles" / "backends" / "megatron_utils" / "megatron_to_hf"
    for module_name in ("qwen2", "higgs_tts"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified_name, base / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.higgs_tts"].convert_higgs_to_hf


@pytest.fixture(scope="module")
def convert_higgs_to_hf():
    return _load_higgs_converter()


def _converter_args() -> Namespace:
    return Namespace(
        hidden_size=4,
        kv_channels=1,
        num_attention_heads=4,
        num_query_groups=2,
        higgs_num_codebooks=2,
        higgs_codebook_vocab_size=3,
    )


def test_higgs_converter_emits_canonical_embedding_names(convert_higgs_to_hf):
    args = _converter_args()
    text_embedding = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    codec_embedding = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    assert convert_higgs_to_hf(
        args,
        "module.module.embedding.word_embeddings.weight",
        text_embedding,
    ) == [("tied.embedding.text_embedding.weight", text_embedding)]
    assert convert_higgs_to_hf(
        args,
        "module.module.codec_embeddings.weight",
        codec_embedding,
    ) == [("tied.embedding.modality_embeddings.0.embedding.weight", codec_embedding)]


def test_higgs_converter_round_trips_grouped_qkv_and_fc1(convert_higgs_to_hf):
    args = _converter_args()
    q = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    k = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
    v = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 200
    fused_qkv = torch.cat(
        (
            q.view(2, 2, 1, 4),
            k.view(2, 1, 1, 4),
            v.view(2, 1, 1, 4),
        ),
        dim=1,
    ).reshape(8, 4)

    converted_qkv = convert_higgs_to_hf(
        args,
        "module.module.decoder.layers.3.self_attention.linear_qkv.weight",
        fused_qkv,
    )
    assert [name for name, _ in converted_qkv] == [
        "body.layers.3.self_attn.q_proj.weight",
        "body.layers.3.self_attn.k_proj.weight",
        "body.layers.3.self_attn.v_proj.weight",
    ]
    assert torch.equal(converted_qkv[0][1], q)
    assert torch.equal(converted_qkv[1][1], k)
    assert torch.equal(converted_qkv[2][1], v)

    gate = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    up = gate + 100
    converted_fc1 = convert_higgs_to_hf(
        args,
        "module.module.decoder.layers.3.mlp.linear_fc1.weight",
        torch.cat((gate, up), dim=0),
    )
    assert [name for name, _ in converted_fc1] == [
        "body.layers.3.mlp.gate_proj.weight",
        "body.layers.3.mlp.up_proj.weight",
    ]
    assert torch.equal(converted_fc1[0][1], gate)
    assert torch.equal(converted_fc1[1][1], up)


def test_higgs_converter_handles_both_norm_layouts(convert_higgs_to_hf):
    args = _converter_args()
    norm = torch.ones(4)
    mappings = {
        "module.module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight": (
            "body.layers.1.input_layernorm.weight"
        ),
        "module.module.decoder.layers.1.input_layernorm.weight": "body.layers.1.input_layernorm.weight",
        "module.module.decoder.layers.1.mlp.linear_fc1.layer_norm_weight": (
            "body.layers.1.post_attention_layernorm.weight"
        ),
        "module.module.decoder.layers.1.pre_mlp_layernorm.weight": ("body.layers.1.post_attention_layernorm.weight"),
    }
    for source_name, expected_name in mappings.items():
        assert convert_higgs_to_hf(args, source_name, norm) == [(expected_name, norm)]


def test_higgs_converter_covers_exact_399_weight_surface(convert_higgs_to_hf):
    args = Namespace(
        hidden_size=higgs_checkpoint.HIGGS_HIDDEN_SIZE,
        kv_channels=higgs_checkpoint.HIGGS_HEAD_DIM,
        num_attention_heads=higgs_checkpoint.HIGGS_NUM_ATTENTION_HEADS,
        num_query_groups=higgs_checkpoint.HIGGS_NUM_QUERY_GROUPS,
        higgs_num_codebooks=higgs_checkpoint.HIGGS_NUM_CODEBOOKS,
        higgs_codebook_vocab_size=higgs_checkpoint.HIGGS_CODEBOOK_VOCAB_SIZE,
    )
    expected_names = set(higgs_checkpoint.canonical_higgs_policy_shapes())
    for layout in ("transformer_engine", "local"):
        exported_names = []
        for name, shape in higgs_checkpoint._target_parameter_shapes(layout).items():
            parameter = torch.empty(shape, dtype=torch.bfloat16, device="meta")
            exported_names.extend(
                output_name for output_name, _ in convert_higgs_to_hf(args, f"module.module.{name}", parameter)
            )
        assert len(exported_names) == 399
        assert len(set(exported_names)) == 399
        assert set(exported_names) == expected_names


def test_higgs_converter_rejects_noncanonical_head_and_codec_shape(convert_higgs_to_hf):
    args = _converter_args()
    with pytest.raises(ValueError, match="tied Higgs text head"):
        convert_higgs_to_hf(args, "module.module.output_layer.weight", torch.empty(7, 4))
    with pytest.raises(ValueError, match="codec embedding shape mismatch"):
        convert_higgs_to_hf(args, "module.module.codec_embeddings.weight", torch.empty(5, 4))


def test_converter_dispatch_uses_structured_policy_family_before_model_name():
    package_name = "_miles_higgs_dispatch_test"
    base = Path(__file__).resolve().parents[4] / "miles" / "backends" / "megatron_utils" / "megatron_to_hf"
    package = types.ModuleType(package_name)
    package.__path__ = [str(base)]
    sys.modules[package_name] = package

    exports = {
        "deepseekv3": ("convert_deepseekv3_to_hf",),
        "deepseekv4": ("convert_deepseekv4_to_hf",),
        "glm4": ("convert_glm4_to_hf",),
        "glm4moe": ("convert_glm4moe_to_hf",),
        "kimi_vl": ("convert_kimi_k25_to_hf", "convert_kimivl_to_hf"),
        "llama": ("convert_llama_to_hf",),
        "mimo": ("convert_mimo_to_hf",),
        "qwen2": ("convert_qwen2_to_hf",),
        "qwen3_5": ("convert_qwen3_5_to_hf",),
        "qwen3_next": ("convert_qwen3_next_to_hf",),
        "qwen3moe": ("convert_qwen3moe_to_hf",),
    }
    for module_name, function_names in exports.items():
        module = types.ModuleType(f"{package_name}.{module_name}")
        for function_name in function_names:
            setattr(
                module, function_name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wrong dispatch"))
            )
        sys.modules[module.__name__] = module

    calls = []
    higgs_module = types.ModuleType(f"{package_name}.higgs_tts")
    higgs_module.convert_higgs_to_hf = lambda args, name, param: calls.append((name, param)) or [("higgs", param)]
    sys.modules[higgs_module.__name__] = higgs_module
    processors = types.ModuleType(f"{package_name}.processors")
    processors.remove_padding = lambda name, param, vocab_size: param
    processors.quantize_params = lambda args, name, tensors, config: tensors
    sys.modules[processors.__name__] = processors

    spec = importlib.util.spec_from_file_location(
        package_name,
        base / "__init__.py",
        submodule_search_locations=[str(base)],
    )
    dispatch_module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = dispatch_module
    spec.loader.exec_module(dispatch_module)

    parameter = torch.ones(1)
    assert dispatch_module._convert_to_hf_core(
        Namespace(structured_policy_model_family="higgs_tts"),
        "higgs-audio-v3-tts-4b",
        "codec_embeddings.weight",
        parameter,
    ) == [("higgs", parameter)]
    assert calls == [("codec_embeddings.weight", parameter)]

    assert dispatch_module._convert_to_hf_core(
        Namespace(),
        "MilesHiggsMultimodalQwen3Config",
        "codec_embeddings.weight",
        parameter,
    ) == [("higgs", parameter)]
    assert calls == [
        ("codec_embeddings.weight", parameter),
        ("codec_embeddings.weight", parameter),
    ]


def _write_manifest(tmp_path: Path, tensors: dict[str, torch.Tensor], weight_map: dict[str, str]) -> None:
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def test_manifest_validation_is_exact_and_checks_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        higgs_checkpoint,
        "_CANONICAL_HIGGS_POLICY_SHAPES",
        {"policy.weight": (2, 3)},
    )
    weight_map = {"policy.weight": "model.safetensors"}
    _write_manifest(tmp_path, {"policy.weight": torch.ones(2, 3, dtype=torch.bfloat16)}, weight_map)
    assert higgs_checkpoint.validate_higgs_checkpoint_manifest(tmp_path) == weight_map

    _write_manifest(tmp_path, {"policy.weight": torch.ones(2, 4, dtype=torch.bfloat16)}, weight_map)
    with pytest.raises(ValueError, match="checkpoint shape mismatch"):
        higgs_checkpoint.validate_higgs_checkpoint_manifest(tmp_path)

    unexpected_map = {**weight_map, "policy.extra": "model.safetensors"}
    _write_manifest(
        tmp_path,
        {
            "policy.weight": torch.ones(2, 3, dtype=torch.bfloat16),
            "policy.extra": torch.ones(1, dtype=torch.bfloat16),
        },
        unexpected_map,
    )
    with pytest.raises(ValueError, match="unexpected=.*policy.extra"):
        higgs_checkpoint.validate_higgs_checkpoint_manifest(tmp_path)


def test_target_parameter_validation_accepts_only_complete_bf16_layouts():
    for layout in ("transformer_engine", "local"):
        shapes = higgs_checkpoint._target_parameter_shapes(layout)
        parameters = {name: torch.empty(shape, dtype=torch.bfloat16, device="meta") for name, shape in shapes.items()}
        assert higgs_checkpoint.validate_higgs_target_parameters(parameters) == layout

        missing = dict(parameters)
        missing.pop(next(iter(missing)))
        with pytest.raises(ValueError, match="parameter manifest"):
            higgs_checkpoint.validate_higgs_target_parameters(missing)

        wrong_dtype = dict(parameters)
        name = next(iter(wrong_dtype))
        wrong_dtype[name] = torch.empty(shapes[name], dtype=torch.float32, device="meta")
        with pytest.raises(ValueError, match="dtype mismatch"):
            higgs_checkpoint.validate_higgs_target_parameters(wrong_dtype)


def test_direct_loader_copies_and_fuses_complete_policy(tmp_path, monkeypatch):
    dimensions = {
        "HIGGS_NUM_LAYERS": 1,
        "HIGGS_HIDDEN_SIZE": 4,
        "HIGGS_TEXT_VOCAB_SIZE": 7,
        "HIGGS_NUM_ATTENTION_HEADS": 4,
        "HIGGS_NUM_QUERY_GROUPS": 2,
        "HIGGS_HEAD_DIM": 1,
        "HIGGS_FFN_HIDDEN_SIZE": 5,
        "HIGGS_NUM_CODEBOOKS": 2,
        "HIGGS_CODEBOOK_VOCAB_SIZE": 3,
        "HIGGS_CODEC_ROWS": 6,
    }
    for name, value in dimensions.items():
        monkeypatch.setattr(higgs_checkpoint, name, value)
    policy_shapes = higgs_checkpoint._build_policy_shapes()
    monkeypatch.setattr(higgs_checkpoint, "_CANONICAL_HIGGS_POLICY_SHAPES", policy_shapes)

    def values(shape, offset=0):
        return torch.arange(offset, offset + torch.tensor(shape).prod().item(), dtype=torch.bfloat16).reshape(shape)

    source_tensors = {name: values(shape) for name, shape in policy_shapes.items()}
    source_tensors["body.layers.0.self_attn.q_proj.weight"] = values((4, 4), 10)
    source_tensors["body.layers.0.self_attn.k_proj.weight"] = values((2, 4), 100)
    source_tensors["body.layers.0.self_attn.v_proj.weight"] = values((2, 4), 200)
    source_tensors["body.layers.0.mlp.gate_proj.weight"] = values((5, 4), 300)
    source_tensors["body.layers.0.mlp.up_proj.weight"] = values((5, 4), 400)
    weight_map = {name: "model.safetensors" for name in policy_shapes}
    _write_manifest(tmp_path, source_tensors, weight_map)

    class FakeHiggsModel:
        num_codebooks = 2
        codebook_vocab_size = 3

        def __init__(self):
            self.parameters = {
                name: torch.nn.Parameter(torch.full(shape, -1, dtype=torch.bfloat16))
                for name, shape in higgs_checkpoint._target_parameter_shapes("local").items()
            }

        def named_parameters(self):
            return self.parameters.items()

    model = FakeHiggsModel()
    loaded = higgs_checkpoint.load_higgs_policy_checkpoint(model, tmp_path)
    assert loaded == set(policy_shapes)
    assert torch.equal(
        model.parameters["embedding.word_embeddings.weight"],
        source_tensors["tied.embedding.text_embedding.weight"],
    )

    q = source_tensors["body.layers.0.self_attn.q_proj.weight"].view(2, 2, 1, 4)
    k = source_tensors["body.layers.0.self_attn.k_proj.weight"].view(2, 1, 1, 4)
    v = source_tensors["body.layers.0.self_attn.v_proj.weight"].view(2, 1, 1, 4)
    expected_qkv = torch.cat((q, k, v), dim=1).reshape(8, 4)
    assert torch.equal(
        model.parameters["decoder.layers.0.self_attention.linear_qkv.weight"],
        expected_qkv,
    )
    assert torch.equal(
        model.parameters["decoder.layers.0.mlp.linear_fc1.weight"],
        torch.cat(
            (
                source_tensors["body.layers.0.mlp.gate_proj.weight"],
                source_tensors["body.layers.0.mlp.up_proj.weight"],
            ),
            dim=0,
        ),
    )


def test_repo_id_resolution_uses_snapshot_download(tmp_path, monkeypatch):
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *, repo_id: calls.append(repo_id) or str(snapshot),
    )

    assert higgs_checkpoint.resolve_higgs_checkpoint_path("bosonai/higgs-audio-v3-tts-4b") == snapshot
    assert calls == ["bosonai/higgs-audio-v3-tts-4b"]


def test_latest_cached_checkpoint_has_exact_policy_manifest():
    if not _CACHED_HIGGS_REVISION.is_dir():
        pytest.skip("latest Higgs checkpoint is not cached")
    manifest = higgs_checkpoint.validate_higgs_checkpoint_manifest(_CACHED_HIGGS_REVISION)
    assert len(manifest) == 399
    assert "tied.embedding.text_embedding.weight" in manifest
    assert "tied.embedding.modality_embeddings.0.embedding.weight" in manifest
    assert not any(name.startswith("tied.head.") for name in manifest)
