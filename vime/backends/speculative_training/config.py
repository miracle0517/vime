from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Sequence


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


def resolve_feature_layer_ids(args: Namespace) -> list[int]:
    explicit = parse_int_list(getattr(args, "draft_feature_layer_ids", None))
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    if explicit is None:
        if num_layers < 5:
            raise ValueError(
                "--draft-feature-layer-ids is required when the Target layer count cannot "
                "safely derive the EAGLE3 default [2, num_layers//2, num_layers-3]."
            )
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

    if str(getattr(args, "draft_algorithm", "eagle3")).lower() != "eagle3":
        raise ValueError("The first external Draft training implementation supports only --draft-algorithm=eagle3")
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
    if method not in {"eagle", "eagle3"}:
        raise ValueError("External Draft training requires vLLM speculative method 'eagle' or 'eagle3'")
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
