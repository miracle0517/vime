from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path


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


def _local_draft_config(args: Namespace) -> dict:
    model_path = getattr(args, "draft_model_path", None)
    if not model_path:
        return {}
    config_path = Path(str(model_path)) / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Draft config {config_path} must contain a JSON object")
    return value


def resolve_feature_layer_ids(args: Namespace) -> list[int]:
    explicit = parse_int_list(getattr(args, "draft_feature_layer_ids", None))
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    algorithm = str(getattr(args, "draft_algorithm", "eagle3")).lower()
    if explicit is None:
        if algorithm == "dspark":
            explicit = parse_int_list(_local_draft_config(args).get("aux_hidden_state_layer_ids"))
            if explicit is None:
                raise ValueError(
                    "--draft-feature-layer-ids is required for a remote DSpark checkpoint; "
                    "for a local Speculators checkpoint it is read from config.json"
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
        value = _local_draft_config(args).get("block_size")
    if value is None:
        raise ValueError(
            "--draft-dspark-block-size is required for a remote DSpark checkpoint; "
            "for a local Speculators checkpoint it is read from config.json"
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


def validate_external_draft_args(args: Namespace) -> None:
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
    if configured_model and str(configured_model) != str(args.draft_model_path):
        raise ValueError(
            "vLLM speculative model and --draft-model-path must identify the same checkpoint: "
            f"{configured_model!r} != {args.draft_model_path!r}"
        )
    acceptance_method = str(spec_config.get("acceptance_method", "")).strip().lower()
    if acceptance_method in {"typical_acceptance_sampler", "typical", "topk"}:
        raise ValueError("External Draft RL rollout requires a lossless speculative acceptance method")

    layer_ids = resolve_feature_layer_ids(args)
    args.draft_feature_layer_ids = layer_ids
    if algorithm == "dspark":
        draft_config = _local_draft_config(args)
        if draft_config and "speculators_model_type" not in draft_config:
            raise ValueError("Qwen DSpark online training requires a Speculators-format checkpoint")
        config_algorithm = str(draft_config.get("speculators_model_type", "dspark")).lower()
        if config_algorithm != "dspark":
            raise ValueError(
                "Qwen DSpark training requires a Speculators DSpark checkpoint, got "
                f"speculators_model_type={config_algorithm!r}"
            )
        transformer_config = draft_config.get("transformer_layer_config")
        if isinstance(transformer_config, dict):
            model_type = str(transformer_config.get("model_type", "")).lower()
            if model_type and not model_type.startswith("qwen3"):
                raise ValueError(
                    "VIME currently supports only Qwen3-family DSpark checkpoints, got "
                    f"transformer_layer_config.model_type={model_type!r}"
                )
        markov_head_type = str(draft_config.get("markov_head_type", "vanilla")).lower()
        if markov_head_type != "vanilla":
            raise ValueError(
                "Qwen DSpark serving currently supports only markov_head_type='vanilla', got " f"{markov_head_type!r}"
            )
        if bool(draft_config.get("enable_confidence_head", True)) and not bool(
            draft_config.get("confidence_head_with_markov", True)
        ):
            raise ValueError("Qwen DSpark confidence head must use confidence_head_with_markov=true")
        args.draft_dspark_block_size = resolve_dspark_block_size(args)
        for name in ("draft_dspark_max_anchors",):
            if int(getattr(args, name, 0) or 0) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        try:
            loss_config = json.loads(str(getattr(args, "draft_dspark_loss_fn", "")))
        except json.JSONDecodeError as exc:
            raise ValueError("--draft-dspark-loss-fn must be a JSON object") from exc
        if not isinstance(loss_config, dict) or not loss_config:
            raise ValueError("--draft-dspark-loss-fn must be a non-empty JSON object")
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
    rate = float(getattr(args, "draft_collection_sample_rate", 1.0))
    if rate <= 0 or rate > 1:
        raise ValueError("--draft-collection-sample-rate must be in (0, 1]")
    if int(getattr(args, "draft_lr_warmup_steps", 0)) < 0 or int(getattr(args, "draft_lr_total_steps", 0)) < 0:
        raise ValueError("Draft LR warmup and total steps must be non-negative")
