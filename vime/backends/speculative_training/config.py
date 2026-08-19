from __future__ import annotations

import json
import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path, PurePosixPath
from uuid import uuid4


logger = logging.getLogger(__name__)

_DRAFT_CONFIG_CACHE_ATTR = "_vime_draft_checkpoint_config"
_DSPARK_QWEN_LAYOUT_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
)
_DSPARK_SERVING_FIELDS = (
    "block_size",
    "confidence_head_with_markov",
    "draft_vocab_size",
    "enable_confidence_head",
    "markov_head_type",
    "markov_rank",
    "mask_token_id",
    "num_anchors",
    "num_target_layers",
    "sample_from_anchor",
    "sliding_window_non_causal",
    "target_hidden_size",
)


def external_draft_enabled(args: Namespace) -> bool:
    return bool(getattr(args, "enable_external_draft_training", False))


def parse_int_list(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.startswith("["):
            value = json.loads(value)
        else:
            value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, Sequence):
        raise TypeError(f"Expected a list of integers, got {type(value).__name__}")
    result = [int(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"Draft feature layer ids must be unique, got {result}")
    return result


def make_dspark_vllm_compatible_config(
    config: dict,
    *,
    transformer_config: object | None = None,
) -> dict:
    """Build one config.json that both vLLM and Speculators can consume.

    Speculators serializes Qwen fields under ``transformer_layer_config`` and
    intentionally omits the Hugging Face ``model_type`` class variable. vLLM's
    normal DSpark loader instead resolves a top-level Qwen3 config. Keep the
    nested representation for continued training while mirroring the complete
    transformer config at the top level for serving.
    """

    if not isinstance(config, dict):
        raise TypeError("DSpark config must be a dictionary")

    value = transformer_config
    if value is None:
        value = config.get("transformer_layer_config")
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise ValueError(
            "DSpark Speculators config must contain transformer_layer_config before it can be served by vLLM"
        )

    nested = deepcopy(value)
    model_type = str(nested.get("model_type", "")).lower()
    if model_type != "qwen3":
        raise ValueError(
            "VIME can export only dense Qwen3 DSpark models to vLLM, got "
            f"transformer_layer_config.model_type={model_type!r}"
        )
    missing = [name for name in _DSPARK_QWEN_LAYOUT_FIELDS if nested.get(name) is None]
    if missing:
        raise ValueError(
            "DSpark transformer_layer_config is incomplete for vLLM; missing "
            f"{missing}. Refusing to let Qwen3 defaults silently change the Draft tensor layout."
        )

    result = deepcopy(config)
    serving_values = {name: deepcopy(result[name]) for name in _DSPARK_SERVING_FIELDS if name in result}
    result.update(nested)
    result.update(serving_values)
    result["transformer_layer_config"] = nested
    result["speculators_model_type"] = "dspark"
    result["model_type"] = "qwen3"
    result["architectures"] = ["Qwen3DSparkModel"]
    sample_from_anchor = result.get("sample_from_anchor")
    if not isinstance(sample_from_anchor, bool):
        raise ValueError(
            "DSpark config must define boolean sample_from_anchor so vLLM proposal alignment is unambiguous"
        )
    # Speculators names the training convention from the proposal side;
    # vLLM names the same choice from the verifier's bonus-anchor side.
    result["dspark_bonus_anchor"] = not sample_from_anchor

    layer_ids = parse_int_list(config.get("aux_hidden_state_layer_ids"))
    if layer_ids is not None:
        # Speculators numbers captured hidden states directly. Dense DSpark's
        # target_layer_ids use the preceding decoder-layer index (i - 1).
        result["aux_hidden_state_layer_ids"] = layer_ids
        result["eagle_aux_hidden_state_layer_ids"] = layer_ids
        result["target_layer_ids"] = [layer_id - 1 for layer_id in layer_ids]

        dflash_config = result.get("dflash_config")
        if dflash_config is None:
            dflash_config = {}
        if not isinstance(dflash_config, dict):
            raise TypeError("DSpark dflash_config must be a dictionary when present")
        dflash_config = deepcopy(dflash_config)
        dflash_config["target_layer_ids"] = list(result["target_layer_ids"])
        dflash_config["mask_token_id"] = result.get("mask_token_id")
        # Speculators makes full attention non-causal and optionally makes
        # sliding-window attention non-causal too. vLLM's omitted default
        # already represents the former mixed behavior; only the latter needs
        # an explicit all-layer override.
        if bool(result.get("sliding_window_non_causal", False)):
            dflash_config["causal"] = False
        else:
            dflash_config.pop("causal", None)
        result["dflash_config"] = dflash_config

    nested_speculators_config = result.get("speculators_config")
    if nested_speculators_config is not None and not isinstance(nested_speculators_config, dict):
        raise TypeError("DSpark speculators_config must be a dictionary or null")
    if "speculators_config" in result and nested_speculators_config is None:
        # An omitted value uses Speculators' default. An explicit JSON null is
        # rejected by current Pydantic releases even though that default is None.
        del result["speculators_config"]
    return result


def ensure_local_dspark_vllm_config(
    args: Namespace,
    config: dict,
    *,
    resolved_config: object | None = None,
) -> dict:
    """Normalize a local Speculators checkpoint for vLLM before startup.

    The typed Speculators config has already passed preflight. This conversion
    is deliberately additive: the discriminator and nested transformer config
    remain in place, so the same two-file directory stays reloadable for Draft
    training. No architecture values are reconstructed from tensor shapes or
    from an unrelated Target config.
    """

    model_path = Path(str(getattr(args, "draft_model_path", ""))).expanduser()
    config_path = model_path / "config.json"
    if not model_path.is_dir() or not config_path.is_file():
        return config

    raw_transformer = config.get("transformer_layer_config")
    if not isinstance(raw_transformer, dict):
        raise ValueError(
            f"DSpark checkpoint {model_path} must contain a raw transformer_layer_config object before "
            "it can be normalized for vLLM"
        )
    missing_layout = [name for name in _DSPARK_QWEN_LAYOUT_FIELDS if raw_transformer.get(name) is None]
    if missing_layout:
        raise ValueError(
            f"DSpark checkpoint {model_path} raw transformer_layer_config is missing {missing_layout}; "
            "typed defaults cannot be persisted as guessed tensor layout"
        )
    if not isinstance(config.get("sample_from_anchor"), bool):
        raise ValueError(
            f"DSpark checkpoint {model_path} must explicitly define boolean sample_from_anchor before "
            "serving alignment can be normalized"
        )

    source_config = deepcopy(config)
    if resolved_config is not None and hasattr(resolved_config, "to_dict"):
        resolved_dict = resolved_config.to_dict()
        if isinstance(resolved_dict, dict):
            for name, value in resolved_dict.items():
                if name != "transformer_layer_config" and name not in source_config:
                    source_config[name] = deepcopy(value)
    transformer = raw_transformer

    try:
        normalized = make_dspark_vllm_compatible_config(source_config, transformer_config=transformer)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DSpark checkpoint {model_path} cannot be converted to a vLLM Qwen3 config without guessing its "
            "tensor layout or proposal alignment. Restore config.json from the matching original checkpoint "
            "or export it again with the updated VIME code."
        ) from exc

    if normalized == config:
        return config

    temporary_path = config_path.with_name(f".{config_path.name}.vime-{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(normalized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, config_path)
    except OSError as exc:
        raise ValueError(
            f"VIME needs to upgrade legacy DSpark config {config_path} for vLLM, but the checkpoint directory "
            "is not writable. Copy it to a writable directory or replace config.json with the matching original "
            "DSpark configuration."
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)

    config.clear()
    config.update(normalized)
    setattr(args, _DRAFT_CONFIG_CACHE_ATTR, config)
    logger.warning(
        "Upgraded legacy DSpark config in %s to the combined vLLM/Speculators schema",
        config_path,
    )
    return config


def _looks_like_local_path(value: str, path: Path) -> bool:
    return path.exists() or path.is_absolute() or PurePosixPath(value).is_absolute() or value.startswith(("./", "../"))


def load_draft_checkpoint_config(args: Namespace) -> dict:
    cached = getattr(args, _DRAFT_CONFIG_CACHE_ATTR, None)
    if isinstance(cached, dict):
        return cached

    model_path = getattr(args, "draft_model_path", None)
    if not model_path:
        return {}
    model_id = str(model_path)
    local_path = Path(model_id).expanduser()
    if _looks_like_local_path(model_id, local_path):
        if not local_path.exists():
            raise ValueError(
                f"DSpark checkpoint path {model_id!r} does not exist on the VIME driver. "
                "Mount the checkpoint at the same path used by the rollout workers, or use a Hugging Face model ID."
            )
        if not local_path.is_dir():
            raise ValueError(f"DSpark checkpoint path {model_id!r} must be a directory")
        config_path = local_path / "config.json"
        if not config_path.is_file():
            raise ValueError(f"DSpark checkpoint directory {model_id!r} does not contain config.json")
        try:
            with config_path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read DSpark checkpoint config {config_path}") from exc
    else:
        try:
            from transformers import PretrainedConfig

            value, _ = PretrainedConfig.get_config_dict(model_id)
        except Exception as exc:
            raise ValueError(
                f"Unable to load config.json for DSpark checkpoint {model_id!r}. "
                "If this is a local checkpoint, pass an existing path visible to the VIME driver."
            ) from exc
    if not isinstance(value, dict):
        raise TypeError(f"Draft config for {model_id!r} must contain a JSON object")
    setattr(args, _DRAFT_CONFIG_CACHE_ATTR, value)
    return value


def _draft_config_value(config: dict, keys: Sequence[str]) -> object | None:
    """Read a DSpark field across Speculators and vLLM checkpoint schemas."""

    for key in keys:
        if config.get(key) is not None:
            return config[key]
    for container_key in ("hf_config", "draft_model_config", "eagle_config"):
        nested = config.get(container_key)
        if isinstance(nested, dict):
            for key in keys:
                if nested.get(key) is not None:
                    return nested[key]
    return None


def resolve_feature_layer_ids(args: Namespace) -> list[int]:
    explicit = parse_int_list(getattr(args, "draft_feature_layer_ids", None))
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    algorithm = str(getattr(args, "draft_algorithm", "eagle3")).lower()
    if explicit is None:
        if algorithm == "dspark":
            draft_config = load_draft_checkpoint_config(args)
            explicit = parse_int_list(draft_config.get("aux_hidden_state_layer_ids"))
            if explicit is None:
                raise ValueError(
                    f"DSpark checkpoint {args.draft_model_path!r} config.json does not define "
                    "aux_hidden_state_layer_ids. VIME does not infer checkpoint layer identity from "
                    "vLLM target_layer_ids, command-line values, or Target depth."
                )
        elif num_layers < 5:
            raise ValueError(
                "--draft-feature-layer-ids is required when the Target layer count cannot "
                "safely derive the EAGLE3 default [2, num_layers//2, num_layers-3]."
            )
        else:
            explicit = [2, num_layers // 2, num_layers - 3]
    normalized = []
    for layer_id in explicit:
        if layer_id < 0:
            if num_layers <= 0:
                raise ValueError("Negative Draft layer ids require --num-layers")
            layer_id += num_layers
        if layer_id < 0 or (num_layers > 0 and layer_id >= num_layers):
            raise ValueError(f"Draft feature layer id {layer_id} is outside Target depth {num_layers}")
        normalized.append(layer_id)
    return normalized


def resolve_dspark_block_size(args: Namespace) -> int:
    value = getattr(args, "draft_dspark_block_size", None)
    if value is None:
        value = _draft_config_value(load_draft_checkpoint_config(args), ("block_size",))
    if value is None:
        raise ValueError(
            f"DSpark checkpoint {args.draft_model_path!r} config.json does not define block_size; "
            "pass --draft-dspark-block-size explicitly"
        )
    value = int(value)
    if value < 2:
        raise ValueError("DSpark block size must be at least 2")
    return value


def _local_checkpoint_has_weights(model_path: object) -> bool:
    path = Path(str(model_path)).expanduser()
    if not path.is_dir():
        return False
    patterns = ("*.safetensors", "pytorch_model*.bin")
    return any(any(path.glob(pattern)) for pattern in patterns)


def should_run_draft_interval(rollout_id: int, interval: int | None) -> bool:
    if interval is None or int(interval) <= 0:
        return False
    return (int(rollout_id) + 1) % int(interval) == 0


def _speculative_config(args: Namespace) -> dict:
    value = getattr(args, "vllm_speculative_config", None)
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def validate_external_draft_args(args: Namespace) -> None:
    if getattr(args, "draft_save_hf", None) and not external_draft_enabled(args):
        raise ValueError("--draft-save-hf requires explicit --enable-external-draft-training")
    if not external_draft_enabled(args):
        return

    algorithm = str(getattr(args, "draft_algorithm", "eagle3")).lower()
    if algorithm not in {"eagle3", "dspark"}:
        raise ValueError("External Draft training supports --draft-algorithm=eagle3 or dspark")
    if not getattr(args, "draft_model_path", None):
        raise ValueError("--enable-external-draft-training requires --draft-model-path")
    if not (getattr(args, "draft_target_embedding_path", None) or getattr(args, "hf_checkpoint", None)):
        raise ValueError("External Draft training requires --hf-checkpoint or --draft-target-embedding-path")
    if not str(getattr(args, "draft_target_embedding_key", "") or "").strip():
        raise ValueError("--draft-target-embedding-key must be non-empty")
    if str(getattr(args, "train_backend", "megatron")) != "megatron":
        raise ValueError("External Draft feature collection currently requires --train-backend=megatron")
    if bool(getattr(args, "debug_rollout_only", False)):
        raise ValueError("External Draft training is unavailable with --debug-rollout-only")
    if algorithm == "dspark" and bool(getattr(args, "debug_train_only", False)):
        raise ValueError("External Draft online publication is unavailable with --debug-train-only")
    if bool(getattr(args, "colocate", False)):
        raise ValueError("The external Draft MVP requires a disaggregated rollout; --colocate is not supported")
    if bool(getattr(args, "release_train", False)):
        raise ValueError("The external Draft MVP does not support --release-train")
    if bool(getattr(args, "keep_old_actor", False)):
        raise ValueError(
            "The external Draft MVP does not support --keep-old-actor because hidden states and the "
            "supervising LM Head must come from the same model copy"
        )
    if bool(getattr(args, "enable_mtp_training", False)):
        raise ValueError("External Draft training and inline MTP training cannot be enabled together")
    if bool(getattr(args, "use_routing_replay", False)) or bool(getattr(args, "use_rollout_routing_replay", False)):
        raise ValueError("The external Draft MVP does not yet support MoE routing replay during feature capture")
    if str(getattr(args, "update_weight_mode", "full")) != "full":
        raise ValueError("External Draft publication requires --update-weight-mode=full")
    if str(getattr(args, "update_weight_transport", "nccl")) != "nccl":
        raise ValueError("External Draft publication currently requires --update-weight-transport=nccl")
    worker_extension = getattr(args, "vllm_worker_extension_cls", None)
    required_worker_extension = (
        "vime.backends.megatron_utils.update_weight.update_weight_from_tensor.vLLMWorkerExtension"
    )
    if algorithm == "dspark" and worker_extension not in (None, required_worker_extension):
        raise ValueError(
            "External Draft publication requires VIME's vLLMWorkerExtension so the paired vLLM-Ascend "
            "NPUWorker can enter a Draft weight-update session; a custom --vllm-worker-extension-cls "
            "would replace that compatibility hook"
        )
    if int(getattr(args, "pipeline_model_parallel_size", 1) or 1) != 1:
        raise ValueError("The external Draft MVP currently requires pipeline model parallel size 1")
    if int(getattr(args, "context_parallel_size", 1) or 1) != 1:
        raise ValueError("The external Draft MVP currently requires context parallel size 1")
    if int(getattr(args, "virtual_pipeline_model_parallel_size", 1) or 1) != 1:
        raise ValueError("The external Draft MVP currently requires virtual pipeline parallel size 1")

    spec_config = _speculative_config(args)
    if not spec_config:
        raise ValueError("External Draft training requires --vllm-speculative-config")
    method = str(spec_config.get("method", "")).strip().lower()
    expected_methods = {"eagle", "eagle3"} if algorithm == "eagle3" else {"dspark"}
    if method not in expected_methods:
        expected = "'eagle' or 'eagle3'" if algorithm == "eagle3" else "'dspark'"
        raise ValueError(f"External {algorithm} training requires vLLM speculative method {expected}")
    configured_model = spec_config.get("model")
    if algorithm == "dspark" and not configured_model:
        raise ValueError(
            "External Draft training requires vLLM speculative config model to identify --draft-model-path"
        )
    if configured_model and str(configured_model) != str(args.draft_model_path):
        raise ValueError(
            "vLLM speculative model and --draft-model-path must identify the same checkpoint: "
            f"{configured_model!r} != {args.draft_model_path!r}"
        )
    requested_speculative_tokens = spec_config.get("num_speculative_tokens")
    if algorithm == "dspark":
        if requested_speculative_tokens is None:
            raise ValueError("Qwen DSpark online training requires an explicit positive vLLM num_speculative_tokens")
        try:
            requested_speculative_tokens = int(requested_speculative_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("vLLM num_speculative_tokens must be an integer for Qwen DSpark") from exc
        if requested_speculative_tokens <= 0:
            raise ValueError("vLLM num_speculative_tokens must be positive for Qwen DSpark")
    acceptance_method = str(spec_config.get("acceptance_method", "")).strip().lower()
    if acceptance_method in {"typical_acceptance_sampler", "typical", "topk"}:
        raise ValueError("External Draft RL rollout requires a lossless speculative acceptance method")
    rejection_sample_method = str(spec_config.get("rejection_sample_method", "standard")).strip().lower()
    if algorithm == "dspark" and rejection_sample_method != "standard":
        raise ValueError(
            "External Draft RL rollout requires rejection_sample_method='standard'; synthetic acceptance does "
            "not preserve the Target sampling distribution"
        )
    if algorithm == "dspark" and getattr(args, "draft_vocab_mapping_path", None):
        raise ValueError(
            "Qwen DSpark does not support --draft-vocab-mapping-path as a trainer-only override. The rollout "
            "engine loads its mapping before the first hot update, so reduced-vocabulary t2d/d2t must be "
            "stored in --draft-model-path."
        )

    layer_ids = resolve_feature_layer_ids(args)
    args.draft_feature_layer_ids = layer_ids
    if algorithm == "dspark":
        draft_config = load_draft_checkpoint_config(args)
        local_checkpoint_has_weights = _local_checkpoint_has_weights(args.draft_model_path)
        config_algorithm = str(draft_config.get("speculators_model_type", "")).lower()
        if config_algorithm != "dspark":
            raise ValueError(
                "Qwen DSpark training requires a canonical Speculators DSpark checkpoint, got "
                f"speculators_model_type={config_algorithm!r}"
            )
        if parse_int_list(draft_config.get("aux_hidden_state_layer_ids")) is None:
            raise ValueError(
                "Qwen DSpark training requires checkpoint config.json to define "
                "aux_hidden_state_layer_ids; command-line layer IDs cannot prove which Target features trained "
                "the checkpoint"
            )
        local_model_path = Path(str(args.draft_model_path)).expanduser()
        if not local_model_path.is_dir():
            layer_ids = parse_int_list(draft_config["aux_hidden_state_layer_ids"])
            try:
                expected_hybrid = make_dspark_vllm_compatible_config(draft_config)
            except (TypeError, ValueError):
                expected_hybrid = {}
            architectures = draft_config.get("architectures")
            transformer_config = draft_config.get("transformer_layer_config")
            mirrored_qwen_fields = tuple(transformer_config) if isinstance(transformer_config, dict) else ()
            direct_hybrid = (
                str(draft_config.get("model_type", "")).lower() == "qwen3"
                and isinstance(architectures, list)
                and "Qwen3DSparkModel" in architectures
                and bool(mirrored_qwen_fields)
                and all(
                    draft_config.get(name) == expected_hybrid.get(name)
                    for name in (
                        *mirrored_qwen_fields,
                        "target_layer_ids",
                        "eagle_aux_hidden_state_layer_ids",
                        "dspark_bonus_anchor",
                        "dflash_config",
                    )
                )
            )
            if not direct_hybrid:
                raise ValueError(
                    "Remote DSpark online training requires an already exported direct Qwen3 hybrid config. "
                    "The generic model_type='speculators' adapter does not preserve anchor/alignment fields "
                    "consistently across vLLM GPU and Ascend runners. Download the checkpoint to a writable "
                    "local directory and normalize/export it with this VIME revision first."
                )
        transformer_config = draft_config.get("transformer_layer_config")
        if isinstance(transformer_config, dict):
            model_type = str(transformer_config.get("model_type", "")).lower()
            if model_type and model_type != "qwen3":
                raise ValueError(
                    "VIME currently supports only dense Qwen3 DSpark checkpoints, got "
                    f"transformer_layer_config.model_type={model_type!r}"
                )
        markov_head_type = str(draft_config.get("markov_head_type", "vanilla")).lower()
        if markov_head_type != "vanilla":
            raise ValueError(
                "Qwen DSpark serving currently supports only markov_head_type='vanilla', got " f"{markov_head_type!r}"
            )
        args.draft_dspark_block_size = resolve_dspark_block_size(args)
        sample_from_anchor = draft_config.get("sample_from_anchor")
        if isinstance(sample_from_anchor, bool):
            proposal_capacity = (
                int(args.draft_dspark_block_size) if sample_from_anchor else int(args.draft_dspark_block_size) - 1
            )
            if requested_speculative_tokens > proposal_capacity:
                raise ValueError(
                    "vLLM num_speculative_tokens exceeds the DSpark checkpoint capacity: "
                    f"{requested_speculative_tokens} > {proposal_capacity} "
                    f"(block_size={args.draft_dspark_block_size}, "
                    f"sample_from_anchor={sample_from_anchor})"
                )
        for name in ("draft_dspark_max_anchors",):
            if int(getattr(args, name, 0) or 0) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        try:
            loss_config = json.loads(str(getattr(args, "draft_dspark_loss_fn", "")))
        except json.JSONDecodeError as exc:
            raise ValueError("--draft-dspark-loss-fn must be a JSON object") from exc
        if not isinstance(loss_config, dict) or not loss_config:
            raise ValueError("--draft-dspark-loss-fn must be a non-empty JSON object")
        if local_model_path.is_dir() and not local_checkpoint_has_weights:
            raise ValueError(
                f"Local DSpark checkpoint {str(local_model_path)!r} contains config.json but no "
                "*.safetensors or pytorch_model*.bin weights"
            )
        # A local checkpoint can be checked without allocating the full model.
        # Fail on the driver before Ray reserves NPU actors.
        if local_checkpoint_has_weights:
            from .factories.speculators_dspark import preflight_dspark_checkpoint

            resolved_config = preflight_dspark_checkpoint(args)
            # Legacy VIME exports may have a complete Speculators config but
            # omit the top-level Qwen fields required by vLLM. Upgrade only
            # that known schema difference after typed preflight succeeds.
            ensure_local_dspark_vllm_config(
                args,
                draft_config,
                resolved_config=resolved_config,
            )
    draft_save_hf = getattr(args, "draft_save_hf", None)
    if draft_save_hf:
        if algorithm != "dspark":
            raise ValueError("--draft-save-hf currently supports only --draft-algorithm=dspark")
        try:
            str(draft_save_hf).format(rollout_id=0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("--draft-save-hf must be a valid path template using only {rollout_id}") from exc
        export_path = Path(str(draft_save_hf)).expanduser()
        if not export_path.is_absolute():
            export_path = Path.cwd() / export_path
        args.draft_save_hf = str(export_path)
    for name in (
        "draft_collect_interval",
        "draft_train_interval",
        "draft_publish_interval",
        "draft_train_steps_per_trigger",
        "draft_batch_size_per_gpu",
        "draft_hidden_window_tokens",
    ):
        if int(getattr(args, name, 0) or 0) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    draft_save_interval = getattr(args, "draft_save_interval", None)
    if algorithm == "dspark" and draft_save_interval is not None and int(draft_save_interval) <= 0:
        raise ValueError("--draft-save-interval must be positive when provided")
    rate = float(getattr(args, "draft_collection_sample_rate", 1.0))
    if rate <= 0 or rate > 1:
        raise ValueError("--draft-collection-sample-rate must be in (0, 1]")
    if int(getattr(args, "draft_lr_warmup_steps", 0)) < 0 or int(getattr(args, "draft_lr_total_steps", 0)) < 0:
        raise ValueError("Draft LR warmup and total steps must be non-negative")
