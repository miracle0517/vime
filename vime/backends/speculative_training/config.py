from __future__ import annotations

import json
import math
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

_DRAFT_CONFIG_CACHE_ATTR = "_vime_draft_checkpoint_config"
_DRAFT_CAPTURE_LAYER_ID_KEYS = (
    "aux_hidden_state_layer_ids",
    "eagle_aux_hidden_state_layer_ids",
    "target_hidden_layer_ids",
)
_DSPARK_LOSS_DEFAULTS = {"ce": 0.1, "tv": 0.9}
_DSPARK_NATIVE_ONLY_ALIASES = (
    "dspark_num_draft_layers",
    "dspark_markov_rank",
    "dspark_markov_head_type",
    "dspark_mask_token_id",
    "dspark_draft_loss_weight",
    "dspark_intermediate_size",
)


def external_draft_enabled(args: Namespace) -> bool:
    # The compatibility flag is normalized into this canonical switch before
    # runtime components are constructed.  Keeping this predicate canonical
    # prevents an unvalidated native/compatibility Namespace from allocating
    # an external Draft worker prematurely.
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


def _looks_like_local_path(value: str, path: Path) -> bool:
    return path.exists() or path.is_absolute() or PurePosixPath(value).is_absolute() or value.startswith(("./", "../"))


def _normalize_model_reference(value: object) -> str | None:
    """Normalize a Speculators checkpoint reference.

    ``from_pretrained`` consumes a directory (or a Hub id), while many DSpark
    launch scripts point at the directory's ``model.safetensors`` file.  The
    latter is unambiguous for a local checkpoint, so accept it and use its
    parent directory.  Other references are returned unchanged and are
    resolved by Hugging Face/Speculators later.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if path.suffix.lower() != ".safetensors":
        return text
    # Only reinterpret a path that is clearly local.  A Hub id containing a
    # dot should continue through the normal transformers resolver.
    is_local = (
        path.exists() or path.is_absolute() or PurePosixPath(text).is_absolute() or text.startswith(("./", "../"))
    )
    if not is_local:
        return text
    if not path.exists():
        raise ValueError(
            f"DSpark weights file {text!r} does not exist on the VIME driver; "
            "mount the checkpoint or pass its directory"
        )
    if not path.is_file():
        raise ValueError(f"DSpark checkpoint reference {text!r} is not a file")
    parent = path.parent
    if not (parent / "config.json").is_file():
        raise ValueError(
            f"DSpark weights file {text!r} has no sibling config.json; pass the Speculators checkpoint directory"
        )
    return str(parent)


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
        # Keep direct callers (without the compatibility alias) consistent
        # with --dspark-pretrained-model handling.
        if local_path.is_file() and local_path.suffix.lower() == ".safetensors":
            normalized = _normalize_model_reference(model_id)
            assert normalized is not None
            model_id = normalized
            local_path = Path(normalized).expanduser()
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
            explicit = parse_int_list(_draft_config_value(draft_config, _DRAFT_CAPTURE_LAYER_ID_KEYS))
            if explicit is None:
                dense_target_layer_ids = parse_int_list(_draft_config_value(draft_config, ("target_layer_ids",)))
                if dense_target_layer_ids is not None:
                    # Dense vLLM/DeepSpec DSpark checkpoints store the decoder
                    # layer preceding each captured hidden state.
                    explicit = [layer_id + 1 for layer_id in dense_target_layer_ids]
            if explicit is None:
                if num_layers < 5:
                    raise ValueError(
                        f"DSpark checkpoint {args.draft_model_path!r} config.json does not define any of "
                        f"{[*_DRAFT_CAPTURE_LAYER_ID_KEYS, 'target_layer_ids']}, and the Target layer count "
                        "is unavailable; "
                        "pass --draft-feature-layer-ids explicitly"
                    )
                # Match Speculators' resolve_target_layer_ids default for
                # checkpoints created without an explicit target layer list.
                explicit = [2, num_layers // 2, num_layers - 3]
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


def should_run_draft_interval(rollout_id: int, interval: int | None) -> bool:
    if interval is None or int(interval) <= 0:
        return False
    return (int(rollout_id) + 1) % int(interval) == 0


def _speculative_config(args: Namespace) -> dict:
    value = getattr(args, "vllm_speculative_config", None)
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def _values_equal(left: object, right: object) -> bool:
    """Compare alias values after normalizing common list/float spellings."""

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        try:
            return list(left) == list(right)  # type: ignore[arg-type]
        except TypeError:
            return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return str(left) == str(right)


def _merge_dspark_scalar_alias(
    args: Namespace,
    *,
    alias_name: str,
    canonical_name: str,
    canonical_default: object,
    transform=lambda value: value,
) -> bool:
    """Merge one ``--dspark-*`` alias into the external ``draft_*`` option."""

    alias_value = getattr(args, alias_name, None)
    if alias_value is None:
        return False
    value = transform(alias_value)
    current = getattr(args, canonical_name, None)
    # The canonical parser defaults are not user intent.  Permit an alias to
    # replace those defaults, but reject two genuinely different settings.
    if current is not None and not _values_equal(current, canonical_default) and not _values_equal(current, value):
        raise ValueError(
            f"Conflicting DSpark options: --{alias_name.replace('_', '-')}={alias_value!r} and "
            f"--{canonical_name.replace('_', '-')}={current!r}"
        )
    setattr(args, canonical_name, value)
    return True


def _merge_dspark_loss_aliases(args: Namespace) -> bool:
    """Translate CE/L1 aliases to Speculators' ``ce``/``tv`` loss JSON."""

    ce = getattr(args, "dspark_ce_loss_alpha", None)
    l1 = getattr(args, "dspark_l1_loss_alpha", None)
    if ce is None and l1 is None:
        return False

    raw = getattr(args, "draft_dspark_loss_fn", None)
    if raw is None:
        loss_config = dict(_DSPARK_LOSS_DEFAULTS)
    else:
        try:
            loss_config = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--dspark-ce-loss-alpha/--dspark-l1-loss-alpha require " "--draft-dspark-loss-fn to be a JSON object"
            ) from exc
        if not isinstance(loss_config, dict):
            raise ValueError(
                "--dspark-ce-loss-alpha/--dspark-l1-loss-alpha require " "--draft-dspark-loss-fn to be a JSON object"
            )
        loss_config = dict(loss_config)

    # ``tv`` is Speculators' name for the L1/total-variation term.  Accept
    # ``l1`` in a hand-written canonical JSON config as a convenience, and make
    # omitted terms explicit before applying one-sided aliases.
    if "tv" not in loss_config and "l1" in loss_config:
        loss_config["tv"] = loss_config.pop("l1")
    for key, default in _DSPARK_LOSS_DEFAULTS.items():
        loss_config.setdefault(key, default)

    for alias_name, key, value in (
        ("dspark_ce_loss_alpha", "ce", ce),
        ("dspark_l1_loss_alpha", "tv", l1),
    ):
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{alias_name.replace('_', '-')} must be non-negative")
        current = loss_config.get(key, _DSPARK_LOSS_DEFAULTS[key])
        if (
            key in loss_config
            and not _values_equal(current, _DSPARK_LOSS_DEFAULTS[key])
            and not _values_equal(current, value)
        ):
            raise ValueError(
                f"Conflicting DSpark loss settings for {key!r}: alias={value} vs "
                f"--draft-dspark-loss-fn value={current}"
            )
        loss_config[key] = value
    args.draft_dspark_loss_fn = json.dumps(loss_config, sort_keys=True, separators=(",", ":"))
    return True


def normalize_dspark_compat_args(args: Namespace) -> bool:
    """Normalize the supported Speculators DSpark compatibility options.

    Returns ``True`` when the compatibility namespace requested DSpark.  The
    native Megatron-only architecture options are deliberately rejected here;
    Speculators loads those values from ``config.json`` and cannot safely
    override tensor shapes at trainer construction time.
    """

    native_values = {
        name: getattr(args, name, None)
        for name in _DSPARK_NATIVE_ONLY_ALIASES
        if getattr(args, name, None) is not None
    }
    if native_values:
        options = ", ".join(f"--{name.replace('_', '-')}" for name in native_values)
        raise ValueError(
            f"{options} are not runtime options for external Speculators DSpark training; they belong to native "
            "Megatron DSpark (or the checkpoint config.json). Remove these flags when using Speculators."
        )

    alias_names = (
        "dspark_enabled",
        "dspark_block_size",
        "dspark_target_layer_ids",
        "dspark_num_anchors",
        "dspark_ce_loss_alpha",
        "dspark_l1_loss_alpha",
        "dspark_confidence_head_alpha",
        "dspark_loss_decay_gamma",
        "dspark_pretrained_model",
    )
    requested = bool(getattr(args, "dspark_enabled", False)) or any(
        getattr(args, name, None) is not None for name in alias_names[1:]
    )
    if not requested:
        return False

    existing_external = bool(getattr(args, "enable_external_draft_training", False))
    existing_algorithm = str(getattr(args, "draft_algorithm", "eagle3")).lower()
    if existing_external and existing_algorithm != "dspark":
        raise ValueError(
            "The --dspark-* compatibility options cannot be combined with external EAGLE3 training; "
            "use --draft-algorithm dspark"
        )
    args.dspark_enabled = True
    args.enable_external_draft_training = True
    args.draft_algorithm = "dspark"

    # Namespaces assembled by integrations/tests may not have gone through the
    # full argparse provider.  Seed the external defaults in that case so the
    # same normalization contract applies to CLI, YAML, and programmatic use.
    if not hasattr(args, "draft_dspark_block_size"):
        args.draft_dspark_block_size = None
    if getattr(args, "draft_dspark_max_anchors", None) is None:
        args.draft_dspark_max_anchors = 64
    if getattr(args, "draft_dspark_loss_fn", None) is None:
        args.draft_dspark_loss_fn = json.dumps(_DSPARK_LOSS_DEFAULTS, separators=(",", ":"))
    if getattr(args, "draft_dspark_confidence_head_alpha", None) is None:
        args.draft_dspark_confidence_head_alpha = 1.0
    if getattr(args, "draft_dspark_decay_gamma", None) is None:
        args.draft_dspark_decay_gamma = 4.0

    _merge_dspark_scalar_alias(
        args,
        alias_name="dspark_block_size",
        canonical_name="draft_dspark_block_size",
        canonical_default=None,
        transform=int,
    )
    dense_layer_ids = getattr(args, "dspark_target_layer_ids", None)
    if dense_layer_ids is not None:
        dense_layer_ids = parse_int_list(dense_layer_ids)
        if not dense_layer_ids:
            raise ValueError("--dspark-target-layer-ids must contain at least one layer id")
        # Speculators feature capture names the hidden state after the decoder
        # layer, whereas the dense DSpark config names the preceding layer.
        capture_layer_ids = [int(layer_id) + 1 for layer_id in dense_layer_ids]
        current = getattr(args, "draft_feature_layer_ids", None)
        current_ids = parse_int_list(current)
        if current_ids is not None and current_ids != capture_layer_ids:
            raise ValueError(
                "Conflicting DSpark layer settings: --dspark-target-layer-ids maps to "
                f"{capture_layer_ids}, but --draft-feature-layer-ids is {current_ids}"
            )
        args.draft_feature_layer_ids = capture_layer_ids
    _merge_dspark_scalar_alias(
        args,
        alias_name="dspark_num_anchors",
        canonical_name="draft_dspark_max_anchors",
        canonical_default=64,
        transform=int,
    )
    _merge_dspark_loss_aliases(args)
    _merge_dspark_scalar_alias(
        args,
        alias_name="dspark_confidence_head_alpha",
        canonical_name="draft_dspark_confidence_head_alpha",
        canonical_default=1.0,
        transform=float,
    )
    _merge_dspark_scalar_alias(
        args,
        alias_name="dspark_loss_decay_gamma",
        canonical_name="draft_dspark_decay_gamma",
        canonical_default=4.0,
        transform=float,
    )
    confidence_alpha = float(getattr(args, "draft_dspark_confidence_head_alpha", 1.0))
    decay_gamma = float(getattr(args, "draft_dspark_decay_gamma", 4.0))
    if not math.isfinite(confidence_alpha) or confidence_alpha < 0:
        raise ValueError("--dspark-confidence-head-alpha must be finite and non-negative")
    if not math.isfinite(decay_gamma) or decay_gamma <= 0:
        raise ValueError("--dspark-loss-decay-gamma must be finite and positive")

    pretrained = getattr(args, "dspark_pretrained_model", None)
    if pretrained is not None:
        normalized = _normalize_model_reference(pretrained)
        current = getattr(args, "draft_model_path", None)
        current_normalized = _normalize_model_reference(current) if current is not None else None
        if current_normalized is not None and current_normalized != normalized:
            raise ValueError(
                "Conflicting DSpark checkpoint settings: "
                f"--dspark-pretrained-model={pretrained!r} vs --draft-model-path={current!r}"
            )
        if current != normalized:
            args.draft_model_path = normalized
            if hasattr(args, _DRAFT_CONFIG_CACHE_ATTR):
                delattr(args, _DRAFT_CONFIG_CACHE_ATTR)
    elif getattr(args, "draft_model_path", None) is not None:
        normalized = _normalize_model_reference(args.draft_model_path)
        if normalized != args.draft_model_path:
            args.draft_model_path = normalized
            if hasattr(args, _DRAFT_CONFIG_CACHE_ATTR):
                delattr(args, _DRAFT_CONFIG_CACHE_ATTR)

    # A checkpoint file reference is not accepted by vLLM's speculative model
    # loader either. Keep the serving-side config in sync when it is a mutable
    # dict (the usual parser result), or serialize a JSON-string config back to
    # the same representation for callers that pass the raw CLI value through.
    spec_value = getattr(args, "vllm_speculative_config", None)
    spec_config = _speculative_config(args)
    configured_model = spec_config.get("model")
    normalized_configured_model = _normalize_model_reference(configured_model) if configured_model else None
    if normalized_configured_model and normalized_configured_model != configured_model:
        if isinstance(spec_value, dict):
            spec_value["model"] = normalized_configured_model
        elif isinstance(spec_value, str):
            try:
                parsed_spec = json.loads(spec_value)
            except (TypeError, ValueError):
                parsed_spec = None
            if isinstance(parsed_spec, dict):
                parsed_spec["model"] = normalized_configured_model
                args.vllm_speculative_config = json.dumps(parsed_spec, separators=(",", ":"))
    return True


def _activate_dspark_training_for_export(args: Namespace) -> None:
    """Turn a DSpark inference export request into an explicit training setup."""

    if not getattr(args, "draft_save_hf", None):
        return
    spec_config = _speculative_config(args)
    method = str(spec_config.get("method", "")).strip().lower()
    if method != "dspark":
        raise ValueError("--draft-save-hf requires DSpark inference (--vllm-speculative-config method=dspark)")

    args.enable_external_draft_training = True
    args.draft_algorithm = "dspark"
    if not getattr(args, "draft_model_path", None):
        configured_model = spec_config.get("model")
        if not configured_model:
            raise ValueError(
                "--draft-save-hf with DSpark inference requires a model in --vllm-speculative-config "
                "or --draft-model-path"
            )
        args.draft_model_path = str(configured_model)


def validate_external_draft_args(args: Namespace) -> None:
    # Normalize the Speculators-compatible ``--dspark-*`` spelling before the
    # existing external-Draft validation/activation logic runs.  This is also
    # what turns --dspark-enabled into the legacy external-training switch.
    normalize_dspark_compat_args(args)
    _activate_dspark_training_for_export(args)
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
    normalized_configured_model = _normalize_model_reference(configured_model) if configured_model else None
    normalized_draft_model = _normalize_model_reference(getattr(args, "draft_model_path", None))
    if normalized_configured_model and normalized_draft_model != normalized_configured_model:
        raise ValueError(
            "vLLM speculative model and --draft-model-path must identify the same checkpoint: "
            f"{configured_model!r} != {args.draft_model_path!r}"
        )
    if normalized_draft_model is not None and normalized_draft_model != getattr(args, "draft_model_path", None):
        args.draft_model_path = normalized_draft_model
        if hasattr(args, _DRAFT_CONFIG_CACHE_ATTR):
            delattr(args, _DRAFT_CONFIG_CACHE_ATTR)
    acceptance_method = str(spec_config.get("acceptance_method", "")).strip().lower()
    if acceptance_method in {"typical_acceptance_sampler", "typical", "topk"}:
        raise ValueError("External Draft RL rollout requires a lossless speculative acceptance method")

    layer_ids = resolve_feature_layer_ids(args)
    args.draft_feature_layer_ids = layer_ids
    if algorithm == "dspark":
        args.draft_dspark_block_size = resolve_dspark_block_size(args)
    draft_save_hf = getattr(args, "draft_save_hf", None)
    if draft_save_hf:
        try:
            str(draft_save_hf).format(rollout_id=0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("--draft-save-hf must be a valid path template using only {rollout_id}") from exc
        export_path = Path(str(draft_save_hf)).expanduser()
        if not export_path.is_absolute():
            export_path = Path.cwd() / export_path
        args.draft_save_hf = str(export_path)
    positive_names = (
        "draft_collect_interval",
        "draft_train_interval",
        "draft_publish_interval",
        "draft_train_steps_per_trigger",
        "draft_batch_size_per_gpu",
        "draft_hidden_window_tokens",
    )
    if algorithm == "dspark":
        positive_names += ("draft_dspark_max_anchors",)
    for name in positive_names:
        if int(getattr(args, name, 0) or 0) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    rate = float(getattr(args, "draft_collection_sample_rate", 1.0))
    if rate <= 0 or rate > 1:
        raise ValueError("--draft-collection-sample-rate must be in (0, 1]")
    if int(getattr(args, "draft_lr_warmup_steps", 0)) < 0 or int(getattr(args, "draft_lr_total_steps", 0)) < 0:
        raise ValueError("Draft LR warmup and total steps must be non-negative")
