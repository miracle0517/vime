from argparse import Namespace

import pytest

from vime.backends.speculative_training.config import (
    resolve_feature_layer_ids,
    should_run_draft_interval,
    validate_external_draft_args,
)


def _args(**overrides):
    values = {
        "enable_external_draft_training": True,
        "draft_algorithm": "eagle3",
        "draft_model_path": "/models/eagle3",
        "hf_checkpoint": "/models/target",
        "draft_target_embedding_path": None,
        "draft_target_embedding_key": "model.embed_tokens.weight",
        "draft_num_nodes": 1,
        "draft_num_gpus_per_node": 1,
        "train_backend": "megatron",
        "debug_rollout_only": False,
        "colocate": False,
        "release_train": False,
        "keep_old_actor": False,
        "enable_mtp_training": False,
        "use_routing_replay": False,
        "use_rollout_routing_replay": False,
        "update_weight_mode": "full",
        "update_weight_transport": "nccl",
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": 1,
        "vllm_speculative_config": {"method": "eagle3", "model": "/models/eagle3"},
        "num_layers": 32,
        "draft_feature_layer_ids": None,
        "draft_collect_interval": 1,
        "draft_train_interval": 1,
        "draft_publish_interval": 1,
        "draft_train_steps_per_trigger": 2,
        "draft_batch_size_per_gpu": 2,
        "draft_hidden_window_tokens": 64,
        "draft_collection_sample_rate": 1.0,
        "draft_lr_warmup_steps": 0,
        "draft_lr_total_steps": 0,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.unit
def test_default_feature_layers_and_interval_are_deterministic():
    assert resolve_feature_layer_ids(_args()) == [2, 16, 29]
    assert not should_run_draft_interval(0, 2)
    assert should_run_draft_interval(1, 2)


@pytest.mark.unit
def test_external_draft_validation_resolves_layers():
    args = _args(draft_feature_layer_ids="2,10,29")
    validate_external_draft_args(args)
    assert args.draft_feature_layer_ids == [2, 10, 29]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"train_backend": "fsdp"}, "megatron"),
        ({"pipeline_model_parallel_size": 2}, "pipeline"),
        ({"keep_old_actor": True}, "same model copy"),
        ({"vllm_speculative_config": {"method": "mtp"}}, "eagle"),
        ({"update_weight_transport": "disk"}, "nccl"),
    ],
)
def test_external_draft_validation_rejects_unsupported_mvp_modes(override, message):
    with pytest.raises(ValueError, match=message):
        validate_external_draft_args(_args(**override))
