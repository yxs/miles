"""Strict raw-checkpoint loading for the initial single-device Higgs policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from types import MappingProxyType

import torch
from safetensors import safe_open

HIGGS_NUM_LAYERS = 36
HIGGS_HIDDEN_SIZE = 2560
HIGGS_TEXT_VOCAB_SIZE = 151936
HIGGS_NUM_ATTENTION_HEADS = 32
HIGGS_NUM_QUERY_GROUPS = 8
HIGGS_HEAD_DIM = 128
HIGGS_FFN_HIDDEN_SIZE = 9728
HIGGS_NUM_CODEBOOKS = 8
HIGGS_CODEBOOK_VOCAB_SIZE = 1026
HIGGS_CODEC_ROWS = HIGGS_NUM_CODEBOOKS * HIGGS_CODEBOOK_VOCAB_SIZE

_FROZEN_CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."
_INDEX_NAME = "model.safetensors.index.json"
_EXPECTED_SAFETENSORS_DTYPE = "BF16"


def _build_policy_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "body.norm.weight": (HIGGS_HIDDEN_SIZE,),
        "tied.embedding.text_embedding.weight": (
            HIGGS_TEXT_VOCAB_SIZE,
            HIGGS_HIDDEN_SIZE,
        ),
        "tied.embedding.modality_embeddings.0.embedding.weight": (
            HIGGS_CODEC_ROWS,
            HIGGS_HIDDEN_SIZE,
        ),
    }
    q_rows = HIGGS_NUM_ATTENTION_HEADS * HIGGS_HEAD_DIM
    kv_rows = HIGGS_NUM_QUERY_GROUPS * HIGGS_HEAD_DIM
    for layer in range(HIGGS_NUM_LAYERS):
        prefix = f"body.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (HIGGS_HIDDEN_SIZE,),
                f"{prefix}.post_attention_layernorm.weight": (HIGGS_HIDDEN_SIZE,),
                f"{prefix}.self_attn.q_norm.weight": (HIGGS_HEAD_DIM,),
                f"{prefix}.self_attn.k_norm.weight": (HIGGS_HEAD_DIM,),
                f"{prefix}.self_attn.q_proj.weight": (q_rows, HIGGS_HIDDEN_SIZE),
                f"{prefix}.self_attn.k_proj.weight": (kv_rows, HIGGS_HIDDEN_SIZE),
                f"{prefix}.self_attn.v_proj.weight": (kv_rows, HIGGS_HIDDEN_SIZE),
                f"{prefix}.self_attn.o_proj.weight": (HIGGS_HIDDEN_SIZE, q_rows),
                f"{prefix}.mlp.gate_proj.weight": (
                    HIGGS_FFN_HIDDEN_SIZE,
                    HIGGS_HIDDEN_SIZE,
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    HIGGS_FFN_HIDDEN_SIZE,
                    HIGGS_HIDDEN_SIZE,
                ),
                f"{prefix}.mlp.down_proj.weight": (
                    HIGGS_HIDDEN_SIZE,
                    HIGGS_FFN_HIDDEN_SIZE,
                ),
            }
        )
    return shapes


_CANONICAL_HIGGS_POLICY_SHAPES: Mapping[str, tuple[int, ...]] = MappingProxyType(_build_policy_shapes())


def canonical_higgs_policy_shapes() -> dict[str, tuple[int, ...]]:
    """Return the 399 indexed trainable-policy names and exact v3 shapes."""

    return dict(_CANONICAL_HIGGS_POLICY_SHAPES)


def resolve_higgs_checkpoint_path(load_path: str | Path) -> Path:
    """Resolve a local checkpoint directory or download a Hugging Face repo ID."""

    if not isinstance(load_path, (str, Path)):
        raise FileNotFoundError(f"Higgs checkpoint path must be a string or Path, got {load_path!r}")
    path = Path(load_path)
    if path.is_dir():
        return path
    repo_id = str(load_path)
    if path.is_absolute() or len(path.parts) != 2:
        raise FileNotFoundError(
            f"Higgs checkpoint {repo_id!r} is not a local directory or a Hugging Face owner/repo ID"
        )
    try:
        from huggingface_hub import snapshot_download

        resolved = Path(snapshot_download(repo_id=repo_id))
    except Exception as error:
        raise FileNotFoundError(f"failed to resolve Higgs checkpoint {repo_id!r}: {error}") from error
    if not resolved.is_dir():
        raise FileNotFoundError(f"resolved Higgs checkpoint is not a directory: {resolved}")
    return resolved


def _summarize_names(names: set[str]) -> str:
    ordered = sorted(names)
    shown = ordered[:8]
    suffix = f" ... ({len(ordered)} total)" if len(ordered) > len(shown) else ""
    return f"{shown}{suffix}"


def _load_weight_map(checkpoint_dir: Path) -> dict[str, str]:
    index_path = checkpoint_dir / _INDEX_NAME
    if not index_path.is_file():
        raise ValueError(f"Higgs raw loading requires {_INDEX_NAME} in {checkpoint_dir}")
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read Higgs safetensors index {index_path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("weight_map"), dict):
        raise ValueError(f"Higgs safetensors index {index_path} has no object weight_map")
    weight_map = payload["weight_map"]
    if any(not isinstance(name, str) or not isinstance(filename, str) for name, filename in weight_map.items()):
        raise ValueError("Higgs safetensors weight_map must contain string names and filenames")
    return dict(weight_map)


def _resolve_weight_files(checkpoint_dir: Path, weight_map: Mapping[str, str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for filename in sorted(set(weight_map.values())):
        relative_path = Path(filename)
        if relative_path.parts != (filename,):
            raise ValueError(f"invalid Higgs safetensors filename {filename!r}")
        path = checkpoint_dir / relative_path
        if not path.is_file():
            raise ValueError(f"invalid or missing Higgs safetensors file {filename!r}")
        files[filename] = path
    return files


def validate_higgs_checkpoint_manifest(checkpoint_dir: str | Path) -> dict[str, str]:
    """Validate the indexed canonical policy surface without loading tensor data."""

    checkpoint_dir = Path(checkpoint_dir)
    weight_map = _load_weight_map(checkpoint_dir)
    expected = set(_CANONICAL_HIGGS_POLICY_SHAPES)
    indexed_policy = {name for name in weight_map if not name.startswith(_FROZEN_CODEC_PREFIX)}
    missing = expected - indexed_policy
    unexpected = indexed_policy - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={_summarize_names(missing)}")
        if unexpected:
            details.append(f"unexpected={_summarize_names(unexpected)}")
        raise ValueError("Higgs checkpoint policy manifest mismatch: " + "; ".join(details))

    files = _resolve_weight_files(checkpoint_dir, {name: weight_map[name] for name in expected})
    names_by_file: dict[str, list[str]] = {}
    for name in expected:
        names_by_file.setdefault(weight_map[name], []).append(name)

    for filename, names in names_by_file.items():
        with safe_open(files[filename], framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing_from_file = set(names) - available
            if missing_from_file:
                raise ValueError(
                    f"Higgs safetensors file {filename!r} is missing indexed tensors "
                    f"{_summarize_names(missing_from_file)}"
                )
            for name in names:
                tensor_slice = handle.get_slice(name)
                actual_shape = tuple(tensor_slice.get_shape())
                expected_shape = _CANONICAL_HIGGS_POLICY_SHAPES[name]
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"Higgs checkpoint shape mismatch for {name}: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )
                actual_dtype = tensor_slice.get_dtype()
                if actual_dtype != _EXPECTED_SAFETENSORS_DTYPE:
                    raise ValueError(
                        f"Higgs checkpoint dtype mismatch for {name}: "
                        f"expected {_EXPECTED_SAFETENSORS_DTYPE}, got {actual_dtype}"
                    )

    return {name: weight_map[name] for name in expected}


def _target_parameter_shapes(norm_layout: str) -> dict[str, tuple[int, ...]]:
    if norm_layout not in {"transformer_engine", "local"}:
        raise ValueError(f"unknown Higgs norm layout {norm_layout!r}")
    shapes: dict[str, tuple[int, ...]] = {
        "embedding.word_embeddings.weight": (
            HIGGS_TEXT_VOCAB_SIZE,
            HIGGS_HIDDEN_SIZE,
        ),
        "decoder.final_layernorm.weight": (HIGGS_HIDDEN_SIZE,),
        "codec_embeddings.weight": (HIGGS_CODEC_ROWS, HIGGS_HIDDEN_SIZE),
    }
    qkv_rows = (HIGGS_NUM_ATTENTION_HEADS + 2 * HIGGS_NUM_QUERY_GROUPS) * HIGGS_HEAD_DIM
    q_rows = HIGGS_NUM_ATTENTION_HEADS * HIGGS_HEAD_DIM
    for layer in range(HIGGS_NUM_LAYERS):
        prefix = f"decoder.layers.{layer}"
        if norm_layout == "transformer_engine":
            input_norm = f"{prefix}.self_attention.linear_qkv.layer_norm_weight"
            post_norm = f"{prefix}.mlp.linear_fc1.layer_norm_weight"
        else:
            input_norm = f"{prefix}.input_layernorm.weight"
            post_norm = f"{prefix}.pre_mlp_layernorm.weight"
        shapes.update(
            {
                input_norm: (HIGGS_HIDDEN_SIZE,),
                post_norm: (HIGGS_HIDDEN_SIZE,),
                f"{prefix}.self_attention.q_layernorm.weight": (HIGGS_HEAD_DIM,),
                f"{prefix}.self_attention.k_layernorm.weight": (HIGGS_HEAD_DIM,),
                f"{prefix}.self_attention.linear_qkv.weight": (
                    qkv_rows,
                    HIGGS_HIDDEN_SIZE,
                ),
                f"{prefix}.self_attention.linear_proj.weight": (
                    HIGGS_HIDDEN_SIZE,
                    q_rows,
                ),
                f"{prefix}.mlp.linear_fc1.weight": (
                    2 * HIGGS_FFN_HIDDEN_SIZE,
                    HIGGS_HIDDEN_SIZE,
                ),
                f"{prefix}.mlp.linear_fc2.weight": (
                    HIGGS_HIDDEN_SIZE,
                    HIGGS_FFN_HIDDEN_SIZE,
                ),
            }
        )
    return shapes


def validate_higgs_target_parameters(named_parameters: Mapping[str, torch.Tensor]) -> str:
    """Validate the wrapper parameter manifest and return its norm layout."""

    actual_names = set(named_parameters)
    matching_layouts = []
    for layout in ("transformer_engine", "local"):
        if actual_names == set(_target_parameter_shapes(layout)):
            matching_layouts.append(layout)
    if len(matching_layouts) != 1:
        te_names = set(_target_parameter_shapes("transformer_engine"))
        local_names = set(_target_parameter_shapes("local"))
        missing_te, unexpected_te = te_names - actual_names, actual_names - te_names
        missing_local, unexpected_local = local_names - actual_names, actual_names - local_names
        raise ValueError(
            "Higgs Megatron parameter manifest does not match a supported TP1 model; "
            f"transformer_engine missing={_summarize_names(missing_te)} "
            f"unexpected={_summarize_names(unexpected_te)}; "
            f"local missing={_summarize_names(missing_local)} "
            f"unexpected={_summarize_names(unexpected_local)}"
        )

    layout = matching_layouts[0]
    for name, expected_shape in _target_parameter_shapes(layout).items():
        parameter = named_parameters[name]
        if tuple(parameter.shape) != expected_shape:
            raise ValueError(
                f"Higgs Megatron shape mismatch for {name}: "
                f"expected {expected_shape}, got {tuple(parameter.shape)}"
            )
        if parameter.dtype != torch.bfloat16:
            raise ValueError(
                f"Higgs Megatron dtype mismatch for {name}: expected torch.bfloat16, got {parameter.dtype}"
            )
    return layout


def _copy_parameter(parameter: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if tuple(source.shape) != tuple(parameter.shape):
        raise ValueError(
            f"Higgs load shape mismatch for {name}: expected {tuple(parameter.shape)}, got {tuple(source.shape)}"
        )
    if source.dtype != torch.bfloat16:
        raise ValueError(f"Higgs load dtype mismatch for {name}: expected torch.bfloat16, got {source.dtype}")
    parameter.copy_(source)


def load_higgs_policy_checkpoint(model: torch.nn.Module, checkpoint_dir: str | Path) -> set[str]:
    """Load the canonical v3 policy weights into an unwrapped TP1 Higgs model."""

    if model.num_codebooks != HIGGS_NUM_CODEBOOKS:
        raise ValueError(f"Higgs model must have {HIGGS_NUM_CODEBOOKS} codebooks")
    if model.codebook_vocab_size != HIGGS_CODEBOOK_VOCAB_SIZE:
        raise ValueError(f"Higgs model codebook vocabulary must be {HIGGS_CODEBOOK_VOCAB_SIZE}")

    named_parameters = dict(model.named_parameters())
    norm_layout = validate_higgs_target_parameters(named_parameters)
    weight_map = validate_higgs_checkpoint_manifest(checkpoint_dir)
    files = _resolve_weight_files(Path(checkpoint_dir), weight_map)
    loaded: set[str] = set()

    with ExitStack() as stack, torch.no_grad():
        handles = {
            filename: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for filename, path in files.items()
        }

        def source(name: str) -> torch.Tensor:
            tensor = handles[weight_map[name]].get_tensor(name)
            loaded.add(name)
            return tensor

        _copy_parameter(
            named_parameters["embedding.word_embeddings.weight"],
            source("tied.embedding.text_embedding.weight"),
            "embedding.word_embeddings.weight",
        )
        _copy_parameter(
            named_parameters["codec_embeddings.weight"],
            source("tied.embedding.modality_embeddings.0.embedding.weight"),
            "codec_embeddings.weight",
        )
        _copy_parameter(
            named_parameters["decoder.final_layernorm.weight"],
            source("body.norm.weight"),
            "decoder.final_layernorm.weight",
        )

        for layer in range(HIGGS_NUM_LAYERS):
            source_prefix = f"body.layers.{layer}"
            target_prefix = f"decoder.layers.{layer}"
            if norm_layout == "transformer_engine":
                input_norm = f"{target_prefix}.self_attention.linear_qkv.layer_norm_weight"
                post_norm = f"{target_prefix}.mlp.linear_fc1.layer_norm_weight"
            else:
                input_norm = f"{target_prefix}.input_layernorm.weight"
                post_norm = f"{target_prefix}.pre_mlp_layernorm.weight"

            direct_mappings = {
                input_norm: f"{source_prefix}.input_layernorm.weight",
                post_norm: f"{source_prefix}.post_attention_layernorm.weight",
                f"{target_prefix}.self_attention.q_layernorm.weight": f"{source_prefix}.self_attn.q_norm.weight",
                f"{target_prefix}.self_attention.k_layernorm.weight": f"{source_prefix}.self_attn.k_norm.weight",
                f"{target_prefix}.self_attention.linear_proj.weight": f"{source_prefix}.self_attn.o_proj.weight",
                f"{target_prefix}.mlp.linear_fc2.weight": f"{source_prefix}.mlp.down_proj.weight",
            }
            for target_name, source_name in direct_mappings.items():
                _copy_parameter(named_parameters[target_name], source(source_name), target_name)

            q = source(f"{source_prefix}.self_attn.q_proj.weight")
            k = source(f"{source_prefix}.self_attn.k_proj.weight")
            v = source(f"{source_prefix}.self_attn.v_proj.weight")
            q = q.view(HIGGS_NUM_QUERY_GROUPS, -1, HIGGS_HEAD_DIM, HIGGS_HIDDEN_SIZE)
            k = k.view(HIGGS_NUM_QUERY_GROUPS, 1, HIGGS_HEAD_DIM, HIGGS_HIDDEN_SIZE)
            v = v.view(HIGGS_NUM_QUERY_GROUPS, 1, HIGGS_HEAD_DIM, HIGGS_HIDDEN_SIZE)
            qkv = torch.cat((q, k, v), dim=1).reshape(-1, HIGGS_HIDDEN_SIZE)
            qkv_name = f"{target_prefix}.self_attention.linear_qkv.weight"
            _copy_parameter(named_parameters[qkv_name], qkv, qkv_name)

            gate = source(f"{source_prefix}.mlp.gate_proj.weight")
            up = source(f"{source_prefix}.mlp.up_proj.weight")
            fc1 = torch.cat((gate, up), dim=0)
            fc1_name = f"{target_prefix}.mlp.linear_fc1.weight"
            _copy_parameter(named_parameters[fc1_name], fc1, fc1_name)

    expected = set(_CANONICAL_HIGGS_POLICY_SHAPES)
    if loaded != expected:
        raise RuntimeError(
            "Higgs loader did not consume the canonical policy surface: "
            f"missing={_summarize_names(expected - loaded)} "
            f"unexpected={_summarize_names(loaded - expected)}"
        )
    return loaded


__all__ = [
    "canonical_higgs_policy_shapes",
    "load_higgs_policy_checkpoint",
    "resolve_higgs_checkpoint_path",
    "validate_higgs_checkpoint_manifest",
    "validate_higgs_target_parameters",
]
