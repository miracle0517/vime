from __future__ import annotations

import torch


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

    config = DSparkSpeculatorConfig.from_pretrained(args.draft_model_path)
    transformer_config = config.transformer_layer_config
    model_type = str(getattr(transformer_config, "model_type", "")).lower()
    if not model_type.startswith("qwen3"):
        raise ValueError(
            "VIME currently supports only Qwen3-family DSpark checkpoints, got "
            f"transformer_layer_config.model_type={model_type!r}"
        )
    if str(config.markov_head_type).lower() != "vanilla":
        raise ValueError("Qwen DSpark serving currently supports only markov_head_type='vanilla'")
    if bool(config.enable_confidence_head) and not bool(config.confidence_head_with_markov):
        raise ValueError("Qwen DSpark confidence head must use confidence_head_with_markov=true")

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

    model = DSparkDraftModel.from_pretrained(args.draft_model_path, config=config)
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
    model = model.to(device=device, dtype=torch.bfloat16)
    confidence_head = getattr(model, "confidence_head", None)
    if confidence_head is not None:
        # vLLM Ascend constructs this head with FP32 parameters and consumes
        # FP32 confidence logits for dynamic verification. Keep the online
        # training replica and the published tensors in the same precision.
        confidence_head.to(device=device, dtype=torch.float32)
    return model
