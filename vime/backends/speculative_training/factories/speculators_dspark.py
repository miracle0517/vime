from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import torch

from vime.backends.speculative_training.config import load_draft_checkpoint_config


_QWEN3_DEFAULT_ROPE_THETA = 1_000_000.0


def _checkpoint_tensor_shapes(model_path: str) -> dict[str, tuple[int, ...]]:
    """Read local checkpoint tensor metadata without allocating model tensors."""

    directory = Path(model_path).expanduser()
    if not directory.is_dir():
        return {}

    shapes: dict[str, tuple[int, ...]] = {}
    safetensor_files = sorted(directory.glob("*.safetensors"))
    if safetensor_files:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "Reading a local DSpark safetensors checkpoint requires the 'safetensors' package"
            ) from exc
        for path in safetensor_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    shapes[name] = tuple(int(value) for value in handle.get_slice(name).get_shape())
        return shapes

    index_path = directory / "pytorch_model.bin.index.json"
    bin_files: list[Path]
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {})
        bin_files = sorted({directory / str(value) for value in weight_map.values()})
    else:
        bin_files = sorted(directory.glob("pytorch_model*.bin"))
    for path in bin_files:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            continue
        for name, value in state.items():
            if torch.is_tensor(value):
                shapes[str(name)] = tuple(int(dimension) for dimension in value.shape)
        del state
    return shapes


def _find_shape(shapes: dict[str, tuple[int, ...]], name: str) -> tuple[int, ...] | None:
    matches = [shape for key, shape in shapes.items() if key == name or key.endswith(f".{name}")]
    if len(matches) > 1 and any(shape != matches[0] for shape in matches[1:]):
        raise ValueError(f"DSpark checkpoint contains ambiguous shapes for {name!r}: {matches}")
    return matches[0] if matches else None


def _find_layer_shape(shapes: dict[str, tuple[int, ...]], name: str) -> tuple[int, ...] | None:
    pattern = re.compile(rf"(?:^|\.)layers\.\d+\.{re.escape(name)}$")
    matches = [shape for key, shape in shapes.items() if pattern.search(key)]
    if len(matches) > 1 and any(shape != matches[0] for shape in matches[1:]):
        raise ValueError(f"DSpark checkpoint contains inconsistent layer shapes for {name!r}: {matches}")
    return matches[0] if matches else None


def _single_dimension(name: str, values: list[int]) -> int | None:
    unique = sorted(set(values))
    if len(unique) > 1:
        raise ValueError(f"DSpark checkpoint has inconsistent {name} dimensions: {unique}")
    return unique[0] if unique else None


def _nested_transformer_config(config_dict: dict) -> dict | None:
    """Find a Qwen config in known Speculators/vLLM compatibility wrappers."""

    pending = [config_dict.get(name) for name in ("hf_config", "draft_model_config", "eagle_config")]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if not isinstance(candidate, dict) or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        transformer = candidate.get("transformer_layer_config")
        if isinstance(transformer, dict):
            return transformer
        model_type = str(candidate.get("model_type", "")).lower()
        if model_type == "qwen3":
            return candidate
        pending.extend(candidate.get(name) for name in ("hf_config", "draft_model_config", "eagle_config"))
    return None


def _target_transformer_config(args) -> dict | None:
    """Load the Target config as a source for non-shape Qwen settings."""

    candidates = (
        getattr(args, "draft_target_embedding_path", None),
        getattr(args, "hf_checkpoint", None),
    )
    for value in dict.fromkeys(str(item) for item in candidates if item):
        path = Path(value).expanduser()
        config_dict = None
        if path.is_dir() and (path / "config.json").is_file():
            try:
                with (path / "config.json").open(encoding="utf-8") as handle:
                    config_dict = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
        elif not path.is_absolute():
            try:
                from transformers import PretrainedConfig

                config_dict, _ = PretrainedConfig.get_config_dict(value)
            except Exception:
                continue
        if not isinstance(config_dict, dict):
            continue
        text_config = config_dict.get("text_config")
        if isinstance(text_config, dict):
            config_dict = text_config
        if str(config_dict.get("model_type", "")).lower() == "qwen3":
            return config_dict
    return None


def _ensure_transformer_config(config_dict: dict, args, shapes: dict) -> None:
    """Populate transformer_layer_config for old checkpoints that omit it."""

    transformer = config_dict.get("transformer_layer_config")
    if isinstance(transformer, dict):
        transformer.setdefault("model_type", "qwen3")
        return
    if transformer is not None:
        raise TypeError("DSpark transformer_layer_config must be a JSON object")

    nested = _nested_transformer_config(config_dict)
    if nested is not None:
        config_dict["transformer_layer_config"] = deepcopy(nested)
        return
    if not shapes:
        raise ValueError(
            "DSpark checkpoint config.json omits transformer_layer_config and no local weights are available "
            "to recover it. Use a complete Speculators export or a local checkpoint directory."
        )

    # Draft layers inherit all non-layout Qwen settings from the Target. Tensor
    # metadata below replaces every dimension that can differ from the Target.
    config_dict["transformer_layer_config"] = deepcopy(_target_transformer_config(args) or {"model_type": "qwen3"})


def _normalize_qwen_rope_config(config_dict: dict, args) -> None:
    """Bridge legacy Qwen3 RoPE fields to Transformers' rope_parameters API."""

    transformer = config_dict.get("transformer_layer_config")
    if not isinstance(transformer, dict) or str(transformer.get("model_type", "qwen3")).lower() != "qwen3":
        return

    raw_parameters = transformer.get("rope_parameters")
    if raw_parameters is not None and not isinstance(raw_parameters, dict):
        raise TypeError("DSpark transformer_layer_config.rope_parameters must be a JSON object or null")

    raw_scaling = transformer.get("rope_scaling")
    if raw_scaling is not None and not isinstance(raw_scaling, dict):
        raise TypeError("DSpark transformer_layer_config.rope_scaling must be a JSON object or null")

    parameters = deepcopy(raw_parameters or raw_scaling or {})
    target_config = None
    if not parameters or parameters.get("rope_theta") is None:
        target_config = _target_transformer_config(args)
    if not parameters and isinstance(target_config, dict):
        target_parameters = target_config.get("rope_parameters")
        target_scaling = target_config.get("rope_scaling")
        if isinstance(target_parameters, dict) and target_parameters:
            parameters = deepcopy(target_parameters)
        elif isinstance(target_scaling, dict) and target_scaling:
            parameters = deepcopy(target_scaling)

    legacy_type = parameters.pop("type", None)
    parameters.setdefault("rope_type", legacy_type or "default")

    theta = parameters.get("rope_theta")
    if theta is None:
        theta = transformer.get("rope_theta")
    if theta is None and isinstance(target_config, dict):
        target_parameters = target_config.get("rope_parameters")
        target_scaling = target_config.get("rope_scaling")
        if isinstance(target_parameters, dict):
            theta = target_parameters.get("rope_theta")
        if theta is None and isinstance(target_scaling, dict):
            theta = target_scaling.get("rope_theta")
        if theta is None:
            theta = target_config.get("rope_theta")
    if theta is None:
        theta = getattr(args, "rotary_base", None) or _QWEN3_DEFAULT_ROPE_THETA

    theta = float(theta)
    if theta <= 0:
        raise ValueError(f"DSpark Qwen3 rope_theta must be positive, got {theta}")
    parameters["rope_theta"] = theta
    transformer["rope_parameters"] = parameters
    # Retain the legacy scalar for older Transformers/Speculators revisions.
    transformer["rope_theta"] = theta


def _recover_transformer_layout(config_dict: dict, shapes: dict) -> None:
    """Make stale Qwen architecture metadata agree with unambiguous tensor shapes."""

    transformer = config_dict.get("transformer_layer_config")
    if not shapes:
        return
    if transformer is None:
        transformer = {"model_type": "qwen3"}
        config_dict["transformer_layer_config"] = transformer
    elif not isinstance(transformer, dict):
        raise TypeError("DSpark transformer_layer_config must be a JSON object")

    checkpoint_layers = {
        int(match.group(1)) for key in shapes if (match := re.search(r"(?:^|\.)layers\.(\d+)\.", key)) is not None
    }
    if checkpoint_layers:
        expected_layers = set(range(max(checkpoint_layers) + 1))
        if checkpoint_layers != expected_layers:
            raise ValueError(
                f"DSpark checkpoint layer ids must be contiguous from zero, got {sorted(checkpoint_layers)}"
            )
        num_hidden_layers = len(checkpoint_layers)
        transformer["num_hidden_layers"] = num_hidden_layers
        layer_types = transformer.get("layer_types")
        if isinstance(layer_types, list) and len(layer_types) != num_hidden_layers:
            if len(set(layer_types)) > 1:
                raise ValueError(
                    "DSpark checkpoint tensor shapes show a different layer count, but config.json contains "
                    "mixed layer_types whose Draft ordering cannot be recovered safely"
                )
            transformer["layer_types"] = [layer_types[0]] * num_hidden_layers if layer_types else []
        max_window_layers = transformer.get("max_window_layers")
        if max_window_layers is not None and int(max_window_layers) > num_hidden_layers:
            transformer["max_window_layers"] = num_hidden_layers

    hidden_norm_shape = _find_shape(shapes, "hidden_norm.weight")
    norm_shape = _find_shape(shapes, "norm.weight")
    fc_shape = _find_shape(shapes, "fc.weight")
    embed_shape = _find_shape(shapes, "embed_tokens.weight")
    lm_head_shape = _find_shape(shapes, "lm_head.weight")
    verifier_lm_head_shape = _find_shape(shapes, "verifier_lm_head.weight")
    markov_w1_shape = _find_shape(shapes, "markov_head.markov_w1.weight")
    t2d_shape = _find_shape(shapes, "t2d")
    q_proj_shape = _find_layer_shape(shapes, "self_attn.q_proj.weight")
    k_proj_shape = _find_layer_shape(shapes, "self_attn.k_proj.weight")
    v_proj_shape = _find_layer_shape(shapes, "self_attn.v_proj.weight")
    o_proj_shape = _find_layer_shape(shapes, "self_attn.o_proj.weight")
    q_norm_shape = _find_layer_shape(shapes, "self_attn.q_norm.weight")
    k_norm_shape = _find_layer_shape(shapes, "self_attn.k_norm.weight")
    gate_proj_shape = _find_layer_shape(shapes, "mlp.gate_proj.weight")
    up_proj_shape = _find_layer_shape(shapes, "mlp.up_proj.weight")
    down_proj_shape = _find_layer_shape(shapes, "mlp.down_proj.weight")
    input_norm_shape = _find_layer_shape(shapes, "input_layernorm.weight")
    post_norm_shape = _find_layer_shape(shapes, "post_attention_layernorm.weight")

    hidden_size = _single_dimension(
        "hidden_size",
        [
            *([hidden_norm_shape[0]] if hidden_norm_shape and len(hidden_norm_shape) == 1 else []),
            *([norm_shape[0]] if norm_shape and len(norm_shape) == 1 else []),
            *([fc_shape[0]] if fc_shape and len(fc_shape) == 2 else []),
            *([embed_shape[1]] if embed_shape and len(embed_shape) == 2 else []),
            *([lm_head_shape[1]] if lm_head_shape and len(lm_head_shape) == 2 else []),
            *([verifier_lm_head_shape[1]] if verifier_lm_head_shape and len(verifier_lm_head_shape) == 2 else []),
            *([q_proj_shape[1]] if q_proj_shape and len(q_proj_shape) == 2 else []),
            *([k_proj_shape[1]] if k_proj_shape and len(k_proj_shape) == 2 else []),
            *([v_proj_shape[1]] if v_proj_shape and len(v_proj_shape) == 2 else []),
            *([o_proj_shape[0]] if o_proj_shape and len(o_proj_shape) == 2 else []),
            *([gate_proj_shape[1]] if gate_proj_shape and len(gate_proj_shape) == 2 else []),
            *([up_proj_shape[1]] if up_proj_shape and len(up_proj_shape) == 2 else []),
            *([down_proj_shape[0]] if down_proj_shape and len(down_proj_shape) == 2 else []),
            *([input_norm_shape[0]] if input_norm_shape and len(input_norm_shape) == 1 else []),
            *([post_norm_shape[0]] if post_norm_shape and len(post_norm_shape) == 1 else []),
        ],
    )
    if hidden_size is not None:
        transformer["hidden_size"] = hidden_size
    verifier_vocab_size = _single_dimension(
        "verifier vocabulary",
        [
            *([markov_w1_shape[0]] if markov_w1_shape and len(markov_w1_shape) == 2 else []),
            *([t2d_shape[0]] if t2d_shape and len(t2d_shape) == 1 else []),
        ],
    )
    if verifier_vocab_size is None and lm_head_shape is not None and _find_shape(shapes, "d2t") is None:
        verifier_vocab_size = lm_head_shape[0]
    if verifier_vocab_size is not None:
        transformer["vocab_size"] = verifier_vocab_size

    intermediate_size = _single_dimension(
        "intermediate_size",
        [
            *([gate_proj_shape[0]] if gate_proj_shape and len(gate_proj_shape) == 2 else []),
            *([up_proj_shape[0]] if up_proj_shape and len(up_proj_shape) == 2 else []),
            *([down_proj_shape[1]] if down_proj_shape and len(down_proj_shape) == 2 else []),
        ],
    )
    if intermediate_size is not None:
        transformer["intermediate_size"] = intermediate_size

    head_dim = _single_dimension(
        "head_dim",
        [
            *([q_norm_shape[0]] if q_norm_shape and len(q_norm_shape) == 1 else []),
            *([k_norm_shape[0]] if k_norm_shape and len(k_norm_shape) == 1 else []),
        ],
    )
    if head_dim is not None:
        transformer["head_dim"] = head_dim
        if q_proj_shape is not None:
            if q_proj_shape[0] % head_dim:
                raise ValueError(
                    f"DSpark q_proj output width {q_proj_shape[0]} is not divisible by head_dim={head_dim}"
                )
            transformer["num_attention_heads"] = q_proj_shape[0] // head_dim
        kv_widths = [shape[0] for shape in (k_proj_shape, v_proj_shape) if shape is not None]
        kv_width = _single_dimension("key/value projection", kv_widths)
        if kv_width is not None:
            if kv_width % head_dim:
                raise ValueError(f"DSpark key/value output width {kv_width} is not divisible by head_dim={head_dim}")
            transformer["num_key_value_heads"] = kv_width // head_dim

    configured_layer_ids = config_dict.get("aux_hidden_state_layer_ids")
    layer_ids = tuple(int(value) for value in configured_layer_ids) if configured_layer_ids is not None else ()
    if fc_shape is not None and len(fc_shape) == 2 and layer_ids:
        if fc_shape[1] % len(layer_ids):
            raise ValueError(
                f"DSpark fc.weight input width {fc_shape[1]} is not divisible by "
                f"the {len(layer_ids)} configured auxiliary hidden states"
            )
        config_dict["target_hidden_size"] = fc_shape[1] // len(layer_ids)


def _recover_missing_layout(config_dict: dict, requested_layer_ids: tuple[int, ...], shapes: dict) -> None:
    """Fill old checkpoint metadata only where tensor shapes make it unambiguous."""

    transformer = config_dict.get("transformer_layer_config")
    hidden_size = int(transformer.get("hidden_size", 0) or 0) if isinstance(transformer, dict) else 0
    configured_layer_ids = config_dict.get("aux_hidden_state_layer_ids")
    if configured_layer_ids is None and hidden_size > 0:
        # Current Speculators constructs fc from Draft hidden_size and cannot
        # consume a different Target width. Treat stale target metadata like
        # the other recoverable Qwen architecture fields.
        config_dict["target_hidden_size"] = hidden_size
    target_hidden_size = int(config_dict.get("target_hidden_size", hidden_size) or hidden_size)
    fc_shape = _find_shape(shapes, "fc.weight")
    inferred_aux_count = None
    if fc_shape is not None and target_hidden_size > 0 and len(fc_shape) == 2:
        if fc_shape[1] % target_hidden_size:
            raise ValueError(
                f"DSpark fc.weight input width {fc_shape[1]} is not divisible by "
                f"target_hidden_size={target_hidden_size}"
            )
        inferred_aux_count = fc_shape[1] // target_hidden_size

    if configured_layer_ids is None:
        if inferred_aux_count is not None and inferred_aux_count != len(requested_layer_ids):
            raise ValueError(
                "DSpark checkpoint omits aux_hidden_state_layer_ids, but fc.weight shows that it expects "
                f"{inferred_aux_count} auxiliary hidden states while VIME resolved "
                f"--draft-feature-layer-ids={requested_layer_ids}. Pass exactly {inferred_aux_count} Target "
                "layer ids in the original checkpoint order. The identities cannot be recovered from weights alone."
            )
        config_dict["aux_hidden_state_layer_ids"] = list(requested_layer_ids)

    markov_w1_shape = _find_shape(shapes, "markov_head.markov_w1.weight")
    markov_w2_shape = _find_shape(shapes, "markov_head.markov_w2.weight")
    lm_head_shape = _find_shape(shapes, "lm_head.weight")
    d2t_shape = _find_shape(shapes, "d2t")
    inferred_draft_vocab_sizes = {
        int(shape[0]) for shape in (lm_head_shape, markov_w2_shape, d2t_shape) if shape is not None and len(shape) >= 1
    }
    if len(inferred_draft_vocab_sizes) > 1:
        raise ValueError(
            "DSpark checkpoint has inconsistent draft vocabulary dimensions across "
            f"lm_head/markov_w2/d2t: {sorted(inferred_draft_vocab_sizes)}"
        )
    if inferred_draft_vocab_sizes:
        config_dict["draft_vocab_size"] = inferred_draft_vocab_sizes.pop()

    inferred_markov_rank = None
    if markov_w1_shape is not None and len(markov_w1_shape) == 2:
        inferred_markov_rank = markov_w1_shape[1]
    if markov_w2_shape is not None and len(markov_w2_shape) == 2:
        if inferred_markov_rank is not None and inferred_markov_rank != markov_w2_shape[1]:
            raise ValueError(
                "DSpark Markov checkpoint is internally inconsistent: "
                f"markov_w1 rank={inferred_markov_rank}, markov_w2 rank={markov_w2_shape[1]}"
            )
        inferred_markov_rank = markov_w2_shape[1]
    if inferred_markov_rank is not None:
        config_dict["markov_rank"] = inferred_markov_rank
    elif shapes:
        config_dict["markov_rank"] = 0

    confidence_shape = _find_shape(shapes, "confidence_head.proj.weight")
    if shapes:
        config_dict["enable_confidence_head"] = confidence_shape is not None
    if shapes and confidence_shape is None:
        config_dict["confidence_head_with_markov"] = False
    elif confidence_shape is not None and hidden_size > 0:
        rank = int(config_dict.get("markov_rank", 0) or 0)
        if confidence_shape == (1, hidden_size + rank):
            config_dict["confidence_head_with_markov"] = True
        elif confidence_shape == (1, hidden_size):
            config_dict["confidence_head_with_markov"] = False
        else:
            raise ValueError(
                f"DSpark confidence_head.proj.weight has shape {confidence_shape}; expected "
                f"(1, {hidden_size}) or (1, {hidden_size + rank}) from checkpoint hidden_size/markov_rank"
            )


def _apply_recovered_layout(config, config_dict: dict) -> None:
    """Reapply recovered fields after runtime-specific config deserialization."""

    transformer = getattr(config, "transformer_layer_config", None)
    transformer_dict = config_dict.get("transformer_layer_config")
    if transformer is not None and isinstance(transformer_dict, dict):
        for name, value in transformer_dict.items():
            setattr(transformer, name, deepcopy(value))

    for name in (
        "aux_hidden_state_layer_ids",
        "draft_vocab_size",
        "target_hidden_size",
        "markov_rank",
        "enable_confidence_head",
        "confidence_head_with_markov",
        "block_size",
        "mask_token_id",
        "sample_from_anchor",
        "sliding_window_non_causal",
        "markov_head_type",
    ):
        if name in config_dict:
            setattr(config, name, deepcopy(config_dict[name]))


def _validate_checkpoint_layout(config, shapes: dict[str, tuple[int, ...]]) -> None:
    if not shapes:
        return
    transformer = config.transformer_layer_config
    hidden_size = int(transformer.hidden_size)
    target_hidden_size = int(getattr(config, "target_hidden_size", None) or hidden_size)
    verifier_vocab_size = int(transformer.vocab_size)
    draft_vocab_size = int(config.draft_vocab_size)
    layer_ids = tuple(int(value) for value in config.aux_hidden_state_layer_ids)
    markov_rank = int(config.markov_rank)
    expected = {
        "fc.weight": (hidden_size, len(layer_ids) * target_hidden_size),
        "hidden_norm.weight": (hidden_size,),
        "norm.weight": (hidden_size,),
    }
    if markov_rank > 0:
        expected["markov_head.markov_w1.weight"] = (verifier_vocab_size, markov_rank)
        expected["markov_head.markov_w2.weight"] = (draft_vocab_size, markov_rank)
    num_layers = int(transformer.num_hidden_layers)
    num_attention_heads = int(transformer.num_attention_heads)
    num_key_value_heads = int(transformer.num_key_value_heads)
    head_dim = int(getattr(transformer, "head_dim", None) or hidden_size // num_attention_heads)
    intermediate_size = int(transformer.intermediate_size)
    for layer_index in range(num_layers):
        prefix = f"layers.{layer_index}"
        expected.update(
            {
                f"{prefix}.self_attn.q_proj.weight": (num_attention_heads * head_dim, hidden_size),
                f"{prefix}.self_attn.k_proj.weight": (num_key_value_heads * head_dim, hidden_size),
                f"{prefix}.self_attn.v_proj.weight": (num_key_value_heads * head_dim, hidden_size),
                f"{prefix}.self_attn.o_proj.weight": (hidden_size, num_attention_heads * head_dim),
                f"{prefix}.self_attn.q_norm.weight": (head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (head_dim,),
                f"{prefix}.mlp.gate_proj.weight": (intermediate_size, hidden_size),
                f"{prefix}.mlp.up_proj.weight": (intermediate_size, hidden_size),
                f"{prefix}.mlp.down_proj.weight": (hidden_size, intermediate_size),
                f"{prefix}.input_layernorm.weight": (hidden_size,),
                f"{prefix}.post_attention_layernorm.weight": (hidden_size,),
            }
        )
    if draft_vocab_size != verifier_vocab_size:
        expected["t2d"] = (verifier_vocab_size,)
        expected["d2t"] = (draft_vocab_size,)
    if bool(config.enable_confidence_head):
        confidence_width = hidden_size + markov_rank if bool(config.confidence_head_with_markov) else hidden_size
        expected["confidence_head.proj.weight"] = (1, confidence_width)
        expected["confidence_head.proj.bias"] = (1,)
    mismatches = []
    checkpoint_layers = {
        int(match.group(1)) for key in shapes if (match := re.search(r"(?:^|\.)layers\.(\d+)\.", key)) is not None
    }
    expected_layers = set(range(num_layers))
    if checkpoint_layers != expected_layers:
        mismatches.append(
            f"layers.*: checkpoint layer ids={sorted(checkpoint_layers)}, "
            f"config layer ids={sorted(expected_layers)}"
        )
    # Speculators may intentionally omit lm_head and reload/share it from the
    # verifier. Validate it when present, but do not require it in the Draft.
    lm_head_shape = _find_shape(shapes, "lm_head.weight")
    if lm_head_shape is not None and lm_head_shape != (draft_vocab_size, hidden_size):
        mismatches.append(f"lm_head.weight: checkpoint={lm_head_shape}, config={(draft_vocab_size, hidden_size)}")
    optional_expected = {
        "embed_tokens.weight": (verifier_vocab_size, hidden_size),
        "verifier_norm.weight": (hidden_size,),
        "verifier_lm_head.weight": (draft_vocab_size, hidden_size),
    }
    for name, expected_shape in optional_expected.items():
        actual_shape = _find_shape(shapes, name)
        if actual_shape is not None and actual_shape != expected_shape:
            mismatches.append(f"{name}: checkpoint={actual_shape}, config={expected_shape}")
    for name, expected_shape in expected.items():
        actual_shape = _find_shape(shapes, name)
        if actual_shape is None:
            mismatches.append(f"{name}: missing (config expects {expected_shape})")
        elif actual_shape != expected_shape:
            mismatches.append(f"{name}: checkpoint={actual_shape}, config={expected_shape}")
    if mismatches:
        displayed = mismatches[:20]
        mismatch_summary = "; ".join(displayed)
        if len(mismatches) > len(displayed):
            mismatch_summary += f"; ... and {len(mismatches) - len(displayed)} more"
        raise ValueError(
            f"DSpark checkpoint tensor layout still has {len(mismatches)} mismatch(es) after safe metadata "
            f"recovery: {mismatch_summary}. Use config.json and weight shards from the same export, or install "
            "the Speculators version that created this checkpoint."
        )


def _build_config(args, config_type):
    """Build from the checkpoint config without changing its tensor layout."""

    # vLLM can serve checkpoints whose config omits Speculators' discriminator,
    # so normalizing that metadata is safe.  Auxiliary layer ids are different:
    # Speculators uses their count when constructing the input projection.  Do
    # not replace them with command-line values before loading weights.
    config_dict = deepcopy(load_draft_checkpoint_config(args))
    config_dict["speculators_model_type"] = "dspark"
    # Speculators currently declares speculators_config as a non-optional
    # Pydantic model while also giving it a None default. A missing field uses
    # that default, but an explicit JSON null from save_pretrained() is
    # validated and rejected. Treat the serialized null as the omitted default
    # so VIME's existing DSpark exports remain reloadable.
    if config_dict.get("speculators_config") is None:
        config_dict.pop("speculators_config", None)
    requested_layer_ids = tuple(int(value) for value in args.draft_feature_layer_ids)
    shapes = _checkpoint_tensor_shapes(str(getattr(args, "draft_model_path", "")))
    _ensure_transformer_config(config_dict, args, shapes)
    _recover_transformer_layout(config_dict, shapes)
    _normalize_qwen_rope_config(config_dict, args)
    _recover_missing_layout(config_dict, requested_layer_ids, shapes)
    config = config_type.from_dict(config_dict)
    _apply_recovered_layout(config, config_dict)

    configured_values = config.aux_hidden_state_layer_ids
    if configured_values is None:
        raise ValueError("DSpark aux_hidden_state_layer_ids is still unset after checkpoint compatibility resolution")
    configured_layer_ids = tuple(int(value) for value in configured_values)
    if configured_layer_ids != requested_layer_ids:
        raise ValueError(
            "DSpark checkpoint aux_hidden_state_layer_ids do not match "
            "--draft-feature-layer-ids: "
            f"{configured_layer_ids} != {requested_layer_ids}. Remove the command-line "
            "override to use config.json, or select a checkpoint trained for those Target layers."
        )
    _validate_checkpoint_layout(config, shapes)
    return config


def _load_pretrained_model(model_type, args, config):
    try:
        return model_type.from_pretrained(args.draft_model_path, config=config)
    except RuntimeError as exc:
        if "ignore_mismatched_sizes" not in str(exc):
            raise
        transformer_config = config.transformer_layer_config
        raise RuntimeError(
            f"DSpark checkpoint weights at {args.draft_model_path!r} do not match config.json. "
            "Do not bypass this with ignore_mismatched_sizes=True because that would randomly "
            "initialize part of the online-training model. Check that config.json and all weight "
            "shards come from the same checkpoint, and install the Speculators version used to "
            "create it. Resolved architecture: "
            f"model_type={getattr(transformer_config, 'model_type', None)!r}, "
            f"hidden_size={getattr(transformer_config, 'hidden_size', None)!r}, "
            f"vocab_size={getattr(transformer_config, 'vocab_size', None)!r}, "
            f"draft_vocab_size={getattr(config, 'draft_vocab_size', None)!r}, "
            f"num_aux_hidden_states={len(config.aux_hidden_state_layer_ids or [])}, "
            f"markov_rank={getattr(config, 'markov_rank', None)!r}, "
            f"confidence_head={getattr(config, 'enable_confidence_head', None)!r}. "
            "The original Transformers loading report immediately above this exception contains "
            "the mismatched parameter names and shapes."
        ) from exc


def _validate_dspark_config(args, config) -> None:
    transformer_config = config.transformer_layer_config
    model_type = str(getattr(transformer_config, "model_type", "")).lower()
    if model_type != "qwen3":
        raise ValueError(
            "VIME currently supports only dense Qwen3 DSpark checkpoints, got "
            f"transformer_layer_config.model_type={model_type!r}"
        )
    rope_parameters = getattr(transformer_config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        raise ValueError("DSpark Qwen3 transformer config must define rope_parameters as an object")
    rope_type = rope_parameters.get("rope_type")
    rope_theta = rope_parameters.get("rope_theta")
    if not rope_type or rope_theta is None or float(rope_theta) <= 0:
        raise ValueError("DSpark Qwen3 rope_parameters must contain a non-empty rope_type and positive rope_theta")
    if str(config.markov_head_type).lower() != "vanilla":
        raise ValueError("Qwen DSpark serving currently supports only markov_head_type='vanilla'")
    markov_rank = int(config.markov_rank)
    if markov_rank < 0:
        raise ValueError(f"DSpark markov_rank must be non-negative, got {markov_rank}")
    draft_hidden_size = int(transformer_config.hidden_size)
    target_hidden_size = int(getattr(config, "target_hidden_size", None) or draft_hidden_size)
    configured_target_hidden_size = int(getattr(args, "hidden_size", 0) or 0)
    if configured_target_hidden_size > 0 and target_hidden_size != configured_target_hidden_size:
        raise ValueError(
            "DSpark checkpoint expects Target hidden_size="
            f"{target_hidden_size}, but the configured Megatron Target uses hidden_size="
            f"{configured_target_hidden_size}. Use a DSpark checkpoint trained for this Target model."
        )
    if target_hidden_size != draft_hidden_size:
        raise ValueError(
            "The installed Speculators DSpark training model cannot load a checkpoint whose "
            f"target_hidden_size={target_hidden_size} differs from Draft hidden_size={draft_hidden_size}; "
            "its fc layer is constructed from Draft hidden_size. Use a same-width Speculators checkpoint."
        )
    if bool(config.enable_confidence_head) and not bool(config.confidence_head_with_markov):
        raise ValueError("Qwen DSpark confidence head must use confidence_head_with_markov=true")
    if bool(config.enable_confidence_head) and bool(config.confidence_head_with_markov) and markov_rank == 0:
        raise ValueError("DSpark confidence_head_with_markov=true requires markov_rank > 0")
    if getattr(config, "mask_token_id", None) is None:
        raise ValueError(
            "DSpark checkpoint config.json does not define mask_token_id. It is required by "
            "Speculators training and cannot be inferred safely from the weights. Use the complete "
            "config from the checkpoint export."
        )
    mask_token_id = int(config.mask_token_id)
    verifier_vocab_size = int(transformer_config.vocab_size)
    if mask_token_id < 0 or mask_token_id >= verifier_vocab_size:
        raise ValueError(
            f"DSpark mask_token_id={mask_token_id} is outside verifier vocabulary [0, {verifier_vocab_size})"
        )

    spec_config = getattr(args, "vllm_speculative_config", None)
    if isinstance(spec_config, str):
        spec_config = json.loads(spec_config)
    if isinstance(spec_config, dict) and spec_config.get("num_speculative_tokens") is not None:
        requested_tokens = int(spec_config["num_speculative_tokens"])
        if requested_tokens <= 0:
            raise ValueError("vLLM num_speculative_tokens must be positive for DSpark")
        available_tokens = int(config.block_size) if bool(config.sample_from_anchor) else int(config.block_size) - 1
        if requested_tokens > available_tokens:
            raise ValueError(
                "vLLM num_speculative_tokens exceeds the DSpark checkpoint capacity: "
                f"{requested_tokens} > {available_tokens} "
                f"(block_size={config.block_size}, sample_from_anchor={config.sample_from_anchor})"
            )


def preflight_dspark_checkpoint(args):
    """Validate a local DSpark checkpoint before Ray actors reserve devices."""

    try:
        from speculators.models.dspark.config import DSparkSpeculatorConfig
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "speculators"
        raise ImportError(
            "Qwen DSpark online training dependencies are incomplete before Ray startup: "
            f"cannot import {missing!r}. Install speculators and its hs_connectors package "
            "from the same source revision."
        ) from exc
    config = _build_config(args, DSparkSpeculatorConfig)
    _validate_dspark_config(args, config)
    return config


def build_model(args, device: torch.device) -> torch.nn.Module:
    """Load a Qwen3-family DSpark checkpoint with Speculators' training model."""

    try:
        from speculators.models.dspark.config import DSparkSpeculatorConfig
        from speculators.models.dspark.core import DSparkDraftModel
    except ImportError as exc:
        raise ImportError(
            "Qwen DSpark online training requires the 'speculators' package. "
            "Install the version matching the DSpark checkpoint and vLLM runtime."
        ) from exc

    config = _build_config(args, DSparkSpeculatorConfig)
    _validate_dspark_config(args, config)
    transformer_config = config.transformer_layer_config

    if device.type == "npu":
        # FlexAttention is CUDA-oriented. The bounded online windows and anchor
        # count make Speculators' eager mask practical on Ascend.
        transformer_config._attn_implementation = "eager"

    target_path = getattr(args, "draft_target_embedding_path", None) or getattr(args, "hf_checkpoint", None)
    verifier = getattr(getattr(config, "speculators_config", None), "verifier", None)
    if verifier is not None and target_path:
        # Do not let a local Draft checkpoint silently fetch the original
        # verifier from the Hub; VIME's configured Target is the version source.
        verifier.name_or_path = str(target_path)

    model = _load_pretrained_model(DSparkDraftModel, args, config)
    if tuple(int(value) for value in model.target_layer_ids) != tuple(args.draft_feature_layer_ids):
        raise ValueError(
            "DSpark checkpoint auxiliary layer ids do not match feature collection: "
            f"{tuple(model.target_layer_ids)} != {tuple(args.draft_feature_layer_ids)}"
        )
    if int(model.block_size) != int(args.draft_dspark_block_size):
        raise ValueError(
            "DSpark checkpoint block_size does not match --draft-dspark-block-size: "
            f"{model.block_size} != {args.draft_dspark_block_size}"
        )

    # VIME captures the tensor entering Megatron's output layer, which is
    # already final-normalized. Speculators normally receives pre-norm hidden
    # states from its data-generation service and applies verifier_norm itself.
    model.verifier_norm = torch.nn.Identity()
    # These heads are synchronized from the current Target model and are not
    # optimization targets. Some Speculators/Transformers combinations restore
    # requires_grad=True while loading a checkpoint, so enforce the invariant
    # before ExternalDraftTrainer constructs its optimizer.
    for head_name in ("lm_head", "verifier_lm_head"):
        head = getattr(model, head_name, None)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad_(False)
    model = model.to(device=device, dtype=torch.bfloat16)
    confidence_head = getattr(model, "confidence_head", None)
    if confidence_head is not None:
        # vLLM Ascend constructs this head with FP32 parameters and consumes
        # FP32 confidence logits for dynamic verification. Keep the online
        # training replica and the published tensors in the same precision.
        confidence_head.to(device=device, dtype=torch.float32)
    return model
