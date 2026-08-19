from __future__ import annotations

import json
from copy import deepcopy

import torch

from vime.backends.speculative_training.backends.dspark import has_valid_draft_vocab_mapping
from vime.backends.speculative_training.config import load_draft_checkpoint_config


def _build_config(args, config_type):
    """Load the supported DSpark checkpoint schema without guessing tensor layout.

    VIME accepts canonical Speculators DSpark configs and the hybrid config
    emitted by :func:`make_dspark_vllm_compatible_config`.  Older exports may
    contain an explicit ``speculators_config: null``; current Pydantic treats
    that differently from an omitted field even though the Speculators default
    is ``None``, so normalize only that known serialization defect.
    """

    config_dict = deepcopy(load_draft_checkpoint_config(args))
    checkpoint_type = str(config_dict.get("speculators_model_type", "")).lower()
    if checkpoint_type != "dspark":
        raise ValueError(
            "Qwen DSpark online training requires a canonical Speculators checkpoint with "
            "speculators_model_type='dspark'; re-export this checkpoint instead of relying on runtime inference"
        )

    transformer_config = config_dict.get("transformer_layer_config")
    if not isinstance(transformer_config, dict):
        raise ValueError(
            "DSpark checkpoint config.json must contain transformer_layer_config. "
            "VIME does not reconstruct model architecture from checkpoint tensor shapes."
        )
    model_type = str(transformer_config.get("model_type", "")).lower()
    if model_type != "qwen3":
        raise ValueError(
            "VIME currently supports only dense Qwen3 DSpark checkpoints, got "
            f"transformer_layer_config.model_type={model_type!r}"
        )

    if config_dict.get("speculators_config") is None:
        config_dict.pop("speculators_config", None)

    configured_layer_ids = config_dict.get("aux_hidden_state_layer_ids")
    if configured_layer_ids is None:
        raise ValueError(
            "DSpark checkpoint config.json must define aux_hidden_state_layer_ids; VIME does not infer Target "
            "layer identity from command-line values or tensor shapes"
        )

    config = config_type.from_dict(config_dict)
    expected_layer_ids = tuple(int(value) for value in args.draft_feature_layer_ids)
    actual_layer_ids = tuple(int(value) for value in config.aux_hidden_state_layer_ids or ())
    if actual_layer_ids != expected_layer_ids:
        raise ValueError(
            "DSpark checkpoint aux_hidden_state_layer_ids do not match --draft-feature-layer-ids: "
            f"{actual_layer_ids} != {expected_layer_ids}. Remove the command-line override to use config.json, "
            "or select a checkpoint trained for those Target layers."
        )
    return config


def _load_pretrained_model(model_type, args, config):
    try:
        loaded = model_type.from_pretrained(
            args.draft_model_path,
            config=config,
            output_loading_info=True,
        )
    except RuntimeError as exc:
        if "size mismatch" not in str(exc).lower() and "ignore_mismatched_sizes" not in str(exc):
            raise
        transformer_config = config.transformer_layer_config
        raise RuntimeError(
            f"DSpark checkpoint weights at {args.draft_model_path!r} do not match config.json. "
            "VIME deliberately does not repair or partially initialize a mismatched online-training model. "
            "Use config.json and weight shards from the same export, and install the pinned Speculators revision. "
            "Resolved architecture: "
            f"model_type={getattr(transformer_config, 'model_type', None)!r}, "
            f"hidden_size={getattr(transformer_config, 'hidden_size', None)!r}, "
            f"vocab_size={getattr(transformer_config, 'vocab_size', None)!r}, "
            f"draft_vocab_size={getattr(config, 'draft_vocab_size', None)!r}, "
            f"num_aux_hidden_states={len(config.aux_hidden_state_layer_ids or [])}, "
            f"markov_rank={getattr(config, 'markov_rank', None)!r}, "
            f"confidence_head={getattr(config, 'enable_confidence_head', None)!r}."
        ) from exc

    if not isinstance(loaded, (tuple, list)) or len(loaded) != 2 or not isinstance(loaded[1], dict):
        raise RuntimeError(
            "The installed Speculators revision did not return Hugging Face loading diagnostics. "
            "Install the revision pinned by docker/Dockerfile.npu; VIME will not load a Draft checkpoint "
            "whose tensor completeness cannot be verified."
        )
    model, loading_info = loaded
    problems = {
        name: values
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if (values := loading_info.get(name))
    }
    if problems:
        details = "; ".join(f"{name}={values!r}" for name, values in problems.items())
        raise RuntimeError(
            f"DSpark checkpoint weights at {args.draft_model_path!r} are incomplete or incompatible with "
            f"config.json: {details}. VIME accepts only the missing tensors explicitly ignored by the pinned "
            "Speculators model (Target-owned embedding/verifier tensors); all Draft backbone, LM Head, Markov, "
            "and enabled confidence tensors must be present. Re-export the checkpoint from its matching config."
        )

    mapping_ready = True
    if bool(getattr(model, "use_draft_vocab", False)):
        mapping_ready = has_valid_draft_vocab_mapping(model)
        if not mapping_ready:
            raise RuntimeError(
                "Reduced-vocabulary DSpark checkpoint is missing a complete t2d/d2t mapping. The pinned "
                "Speculators loader ignores these keys in missing-key diagnostics, so VIME validates the "
                "loaded buffers explicitly instead of accepting their all-zero initialization. Store the "
                "mapping in the checkpoint used by both VIME and the rollout engine."
            )
    # output_loading_info makes Speculators' wrapper receive a tuple, so its
    # normal verifier hook is skipped. Preserve that hook only when it cannot
    # run ahead of an external reduced-vocabulary mapping. VIME initializes the
    # Target embedding explicitly and synchronizes both LM Heads before train.
    load_verifier_weights = getattr(model, "load_verifier_weights", None)
    if mapping_ready and callable(load_verifier_weights):
        load_verifier_weights()
    return model


def _validate_dspark_config(args, config) -> None:
    transformer_config = config.transformer_layer_config
    model_type = str(getattr(transformer_config, "model_type", "")).lower()
    if model_type != "qwen3":
        raise ValueError(
            "VIME currently supports only dense Qwen3 DSpark checkpoints, got "
            f"transformer_layer_config.model_type={model_type!r}"
        )

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
            "The pinned Speculators DSpark training model requires Target and Draft hidden sizes to match, got "
            f"target_hidden_size={target_hidden_size} and draft_hidden_size={draft_hidden_size}."
        )

    if bool(config.enable_confidence_head) and not bool(config.confidence_head_with_markov):
        raise ValueError("Qwen DSpark confidence head must use confidence_head_with_markov=true")
    if bool(config.enable_confidence_head) and markov_rank == 0:
        raise ValueError("DSpark confidence_head_with_markov=true requires markov_rank > 0")

    mask_token_id = getattr(config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("DSpark checkpoint config.json must define mask_token_id")
    verifier_vocab_size = int(transformer_config.vocab_size)
    mask_token_id = int(mask_token_id)
    if mask_token_id < 0 or mask_token_id >= verifier_vocab_size:
        raise ValueError(
            f"DSpark mask_token_id={mask_token_id} is outside verifier vocabulary [0, {verifier_vocab_size})"
        )

    block_size = int(config.block_size)
    if block_size < 2:
        raise ValueError(f"DSpark block_size must be at least 2, got {block_size}")

    spec_config = getattr(args, "vllm_speculative_config", None)
    if isinstance(spec_config, str):
        spec_config = json.loads(spec_config)
    if isinstance(spec_config, dict) and spec_config.get("num_speculative_tokens") is not None:
        requested_tokens = int(spec_config["num_speculative_tokens"])
        if requested_tokens <= 0:
            raise ValueError("vLLM num_speculative_tokens must be positive for DSpark")
        available_tokens = block_size if bool(config.sample_from_anchor) else block_size - 1
        if requested_tokens > available_tokens:
            raise ValueError(
                "vLLM num_speculative_tokens exceeds the DSpark checkpoint capacity: "
                f"{requested_tokens} > {available_tokens} "
                f"(block_size={block_size}, sample_from_anchor={config.sample_from_anchor})"
            )


def preflight_dspark_checkpoint(args):
    """Validate DSpark metadata on the driver before Ray reserves devices."""

    try:
        from speculators.models.dspark.config import DSparkSpeculatorConfig
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "speculators"
        raise ImportError(
            "Qwen DSpark online training dependencies are incomplete before Ray startup: "
            f"cannot import {missing!r}. Install the pinned Speculators and hs_connectors revision."
        ) from exc
    config = _build_config(args, DSparkSpeculatorConfig)
    _validate_dspark_config(args, config)
    return config


def build_model(args, device: torch.device) -> torch.nn.Module:
    """Load a canonical Qwen3 DSpark checkpoint with Speculators' training model."""

    try:
        from speculators.models.dspark.config import DSparkSpeculatorConfig
        from speculators.models.dspark.core import DSparkDraftModel
    except ImportError as exc:
        raise ImportError(
            "Qwen DSpark online training requires the pinned 'speculators' package and hs_connectors workspace."
        ) from exc

    config = _build_config(args, DSparkSpeculatorConfig)
    _validate_dspark_config(args, config)
    transformer_config = config.transformer_layer_config

    if device.type == "npu":
        transformer_config._attn_implementation = "eager"

    target_path = getattr(args, "draft_target_embedding_path", None) or getattr(args, "hf_checkpoint", None)
    verifier = getattr(getattr(config, "speculators_config", None), "verifier", None)
    if verifier is not None and target_path:
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

    # Megatron exports the already final-normalized tensor entering its output layer.
    model.verifier_norm = torch.nn.Identity()
    for head_name in ("lm_head", "verifier_lm_head"):
        head = getattr(model, head_name, None)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad_(False)
    model = model.to(device=device, dtype=torch.bfloat16)
    confidence_head = getattr(model, "confidence_head", None)
    if confidence_head is not None:
        confidence_head.to(device=device, dtype=torch.float32)
    return model
