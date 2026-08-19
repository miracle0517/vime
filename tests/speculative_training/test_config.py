import json
from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vime.backends.speculative_training.config import (
    ensure_local_dspark_vllm_config,
    make_dspark_vllm_compatible_config,
    resolve_feature_layer_ids,
    should_run_draft_interval,
    validate_external_draft_args,
)


def _legacy_saved_dspark_config():
    return {
        "speculators_model_type": "dspark",
        "speculators_config": None,
        "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
        "block_size": 7,
        "mask_token_id": 151669,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "sample_from_anchor": True,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
        },
    }


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
        "vllm_worker_extension_cls": None,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": 1,
        "vllm_speculative_config": {"method": "eagle3", "model": "/models/eagle3"},
        "num_layers": 32,
        "draft_feature_layer_ids": None,
        "draft_collect_interval": 1,
        "draft_train_interval": 1,
        "draft_publish_interval": 1,
        "draft_save_interval": None,
        "draft_train_steps_per_trigger": 2,
        "draft_batch_size_per_gpu": 2,
        "draft_hidden_window_tokens": 64,
        "draft_collection_sample_rate": 1.0,
        "draft_lr_warmup_steps": 0,
        "draft_lr_total_steps": 0,
        "draft_dspark_block_size": None,
        "draft_dspark_max_anchors": 64,
        "draft_dspark_loss_fn": '{"ce": 0.1, "tv": 0.9}',
    }
    values.update(overrides)
    return Namespace(**values)


def _dspark_spec(model, num_speculative_tokens=1):
    return {
        "method": "dspark",
        "model": str(model),
        "num_speculative_tokens": num_speculative_tokens,
    }


def _write_mock_local_dspark_checkpoint(path, monkeypatch, *, block_size=8):
    config = _legacy_saved_dspark_config()
    config["block_size"] = block_size
    config = make_dspark_vllm_compatible_config(config)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"contract-test")

    from vime.backends.speculative_training.factories import speculators_dspark

    monkeypatch.setattr(
        speculators_dspark,
        "preflight_dspark_checkpoint",
        lambda args: Namespace(),
    )
    return config


@pytest.mark.unit
def test_default_feature_layers_and_interval_are_deterministic():
    assert resolve_feature_layer_ids(_args()) == [2, 16, 29]
    assert not should_run_draft_interval(0, 2)
    assert should_run_draft_interval(1, 2)


@pytest.mark.unit
def test_dspark_hybrid_config_is_complete_for_vllm_and_speculators():
    legacy = _legacy_saved_dspark_config()

    result = make_dspark_vllm_compatible_config(legacy)

    assert result["model_type"] == "qwen3"
    assert result["architectures"] == ["Qwen3DSparkModel"]
    assert result["hidden_size"] == 2560
    assert result["num_hidden_layers"] == 5
    assert result["vocab_size"] == 151936
    assert result["target_layer_ids"] == [1, 9, 17, 25, 33]
    assert result["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]
    assert result["dspark_bonus_anchor"] is False
    assert result["dflash_config"] == {
        "mask_token_id": 151669,
        "target_layer_ids": [1, 9, 17, 25, 33],
    }
    assert result["speculators_model_type"] == "dspark"
    assert result["transformer_layer_config"] == legacy["transformer_layer_config"]
    assert "speculators_config" not in result
    assert "model_type" not in legacy


@pytest.mark.unit
def test_dspark_hybrid_config_maps_bonus_anchor_alignment():
    config = _legacy_saved_dspark_config()
    config["sample_from_anchor"] = False

    result = make_dspark_vllm_compatible_config(config)

    assert result["sample_from_anchor"] is False
    assert result["dspark_bonus_anchor"] is True


@pytest.mark.unit
def test_dspark_hybrid_config_maps_noncausal_sliding_attention():
    config = _legacy_saved_dspark_config()
    config["sliding_window_non_causal"] = True

    result = make_dspark_vllm_compatible_config(config)

    assert result["dflash_config"]["causal"] is False


@pytest.mark.unit
def test_dspark_hybrid_config_preserves_zero_based_capture_layer():
    config = _legacy_saved_dspark_config()
    config["aux_hidden_state_layer_ids"] = [0, 10, 18]

    result = make_dspark_vllm_compatible_config(config)

    assert result["target_layer_ids"] == [-1, 9, 17]
    assert result["dflash_config"]["target_layer_ids"] == [-1, 9, 17]


@pytest.mark.unit
@pytest.mark.parametrize("swap_fails", [False, True])
def test_legacy_local_dspark_export_is_upgraded_atomically(tmp_path, monkeypatch, swap_fails):
    import vime.backends.speculative_training.config as config_module

    legacy = _legacy_saved_dspark_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(legacy), encoding="utf-8")
    args = Namespace(draft_model_path=str(tmp_path))

    if swap_fails:

        def fail_replace(source, destination):
            del source, destination
            raise OSError("injected config swap failure")

        monkeypatch.setattr(config_module.os, "replace", fail_replace)
        with pytest.raises(ValueError, match="not writable"):
            ensure_local_dspark_vllm_config(args, legacy)
        assert json.loads(config_path.read_text(encoding="utf-8")) == legacy
    else:
        result = ensure_local_dspark_vllm_config(args, legacy)

        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        assert on_disk == result
        assert on_disk["model_type"] == "qwen3"
        assert on_disk["architectures"] == ["Qwen3DSparkModel"]
        assert on_disk["target_layer_ids"] == [1, 9, 17, 25, 33]
        assert on_disk["transformer_layer_config"]["hidden_size"] == 2560
    assert not list(tmp_path.glob(".config.json.vime-*.tmp"))


@pytest.mark.unit
@pytest.mark.parametrize("missing_field", ["intermediate_size", "head_dim"])
def test_local_dspark_upgrade_does_not_fill_missing_qwen_layout_defaults(tmp_path, missing_field):
    config = _legacy_saved_dspark_config()
    expected_value = config["transformer_layer_config"].pop(missing_field)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    expanded = deepcopy(config)
    expanded["transformer_layer_config"][missing_field] = expected_value
    resolved = SimpleNamespace(
        transformer_layer_config=expanded["transformer_layer_config"],
        to_dict=lambda: expanded,
    )
    with pytest.raises(ValueError, match="typed defaults cannot be persisted"):
        ensure_local_dspark_vllm_config(
            Namespace(draft_model_path=str(tmp_path)),
            config,
            resolved_config=resolved,
        )

    assert json.loads(config_path.read_text(encoding="utf-8")) == config


@pytest.mark.unit
def test_external_draft_validation_resolves_layers():
    args = _args(draft_feature_layer_ids="2,10,29")
    validate_external_draft_args(args)
    assert args.draft_feature_layer_ids == [2, 10, 29]


@pytest.mark.unit
def test_dspark_validation_uses_qwen_speculators_checkpoint_config(tmp_path, monkeypatch):
    config = _write_mock_local_dspark_checkpoint(tmp_path, monkeypatch)
    args = _args(
        draft_algorithm="dspark",
        draft_model_path=str(tmp_path),
        draft_feature_layer_ids=None,
        num_layers=36,
        vllm_speculative_config=_dspark_spec(tmp_path),
    )

    validate_external_draft_args(args)

    assert args.draft_feature_layer_ids == config["aux_hidden_state_layer_ids"]
    assert args.draft_dspark_block_size == 8


@pytest.mark.unit
def test_remote_dspark_training_requires_direct_hybrid_schema(monkeypatch):
    import vime.backends.speculative_training.config as config_module

    monkeypatch.setattr(
        config_module,
        "load_draft_checkpoint_config",
        lambda args: _legacy_saved_dspark_config(),
    )
    args = _args(
        draft_algorithm="dspark",
        draft_model_path="organization/canonical-dspark",
        num_layers=36,
        vllm_speculative_config=_dspark_spec("organization/canonical-dspark"),
    )

    with pytest.raises(ValueError, match="direct Qwen3 hybrid config"):
        validate_external_draft_args(args)


@pytest.mark.unit
def test_remote_dspark_training_accepts_direct_hybrid_schema(monkeypatch):
    import vime.backends.speculative_training.config as config_module

    hybrid = make_dspark_vllm_compatible_config(_legacy_saved_dspark_config())
    monkeypatch.setattr(config_module, "load_draft_checkpoint_config", lambda args: hybrid)
    args = _args(
        draft_algorithm="dspark",
        draft_model_path="organization/hybrid-dspark",
        num_layers=36,
        vllm_speculative_config=_dspark_spec("organization/hybrid-dspark"),
    )

    validate_external_draft_args(args)

    assert args.draft_feature_layer_ids == hybrid["aux_hidden_state_layer_ids"]


@pytest.mark.unit
def test_remote_dspark_training_rejects_stale_top_level_qwen_layout(monkeypatch):
    import vime.backends.speculative_training.config as config_module

    hybrid = make_dspark_vllm_compatible_config(_legacy_saved_dspark_config())
    hybrid["hidden_size"] += 1
    monkeypatch.setattr(config_module, "load_draft_checkpoint_config", lambda args: hybrid)
    args = _args(
        draft_algorithm="dspark",
        draft_model_path="organization/stale-hybrid-dspark",
        num_layers=36,
        vllm_speculative_config=_dspark_spec("organization/stale-hybrid-dspark"),
    )

    with pytest.raises(ValueError, match="direct Qwen3 hybrid config"):
        validate_external_draft_args(args)


@pytest.mark.unit
def test_dspark_local_config_without_weights_fails_before_ray(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(make_dspark_vllm_compatible_config(_legacy_saved_dspark_config())),
        encoding="utf-8",
    )
    args = _args(
        draft_algorithm="dspark",
        draft_model_path=str(tmp_path),
        num_layers=36,
        vllm_speculative_config=_dspark_spec(tmp_path),
    )

    with pytest.raises(ValueError, match="config.json but no .* weights"):
        validate_external_draft_args(args)


@pytest.mark.unit
def test_dspark_validation_rejects_non_qwen_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"speculators_model_type":"dspark","block_size":8,'
        '"aux_hidden_state_layer_ids":[2,14,29],'
        '"transformer_layer_config":{"model_type":"llama"}}',
        encoding="utf-8",
    )
    args = _args(
        draft_algorithm="dspark",
        draft_model_path=str(tmp_path),
        vllm_speculative_config=_dspark_spec(tmp_path),
    )

    with pytest.raises(ValueError, match="dense Qwen3"):
        validate_external_draft_args(args)


@pytest.mark.unit
def test_dspark_validation_rejects_unservable_markov_variant(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"speculators_model_type":"dspark","block_size":8,'
        '"aux_hidden_state_layer_ids":[2,14,29],"markov_head_type":"gated",'
        '"transformer_layer_config":{"model_type":"qwen3"}}',
        encoding="utf-8",
    )
    args = _args(
        draft_algorithm="dspark",
        draft_model_path=str(tmp_path),
        vllm_speculative_config=_dspark_spec(tmp_path),
    )

    with pytest.raises(ValueError, match="vanilla"):
        validate_external_draft_args(args)


@pytest.mark.unit
def test_dspark_export_does_not_enable_training_implicitly():
    with pytest.raises(ValueError, match="requires explicit --enable-external-draft-training"):
        validate_external_draft_args(
            _args(
                enable_external_draft_training=False,
                draft_algorithm="dspark",
                draft_model_path="/models/dspark",
                draft_save_hf="/exports/dspark-{rollout_id}",
                vllm_speculative_config=_dspark_spec("/models/dspark"),
            )
        )


@pytest.mark.unit
def test_dspark_training_does_not_require_hf_export(tmp_path, monkeypatch):
    model_path = tmp_path / "dspark"
    model_path.mkdir()
    _write_mock_local_dspark_checkpoint(model_path, monkeypatch)
    args = _args(
        draft_algorithm="dspark",
        draft_model_path=str(model_path),
        draft_save_hf=None,
        num_layers=36,
        vllm_speculative_config=_dspark_spec(model_path),
    )

    validate_external_draft_args(args)

    assert args.enable_external_draft_training is True
    assert args.draft_save_hf is None


@pytest.mark.unit
def test_dspark_accepts_fixed_export_directory_with_explicit_training(tmp_path, monkeypatch):
    model_path = tmp_path / "dspark_qwen3_4b_block7"
    model_path.mkdir()
    config = _write_mock_local_dspark_checkpoint(model_path, monkeypatch, block_size=7)
    export_path = tmp_path / "dspark_qwen3_4b_block7_after"
    args = _args(
        enable_external_draft_training=True,
        draft_algorithm="dspark",
        draft_model_path=str(model_path),
        draft_save_hf=str(export_path),
        num_layers=36,
        vllm_speculative_config={
            "method": "dspark",
            "model": str(model_path),
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
        },
    )

    validate_external_draft_args(args)

    assert args.draft_algorithm == "dspark"
    assert args.draft_model_path == str(model_path)
    assert args.draft_feature_layer_ids == config["aux_hidden_state_layer_ids"]
    assert args.draft_dspark_block_size == 7
    assert args.draft_save_hf == str(export_path)
    assert not export_path.exists()


@pytest.mark.unit
def test_dspark_missing_local_checkpoint_has_actionable_error(tmp_path):
    missing = tmp_path / "missing-dspark"
    args = _args(
        draft_algorithm="dspark",
        draft_model_path=str(missing),
        draft_save_hf=str(tmp_path / "after"),
        vllm_speculative_config=_dspark_spec(missing),
    )

    with pytest.raises(ValueError, match="does not exist on the VIME driver"):
        validate_external_draft_args(args)


@pytest.mark.unit
def test_dspark_export_path_is_resolved_before_ray_workers_start(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_mock_local_dspark_checkpoint(model_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    args = _args(
        draft_algorithm="dspark",
        draft_model_path=str(model_path),
        draft_save_hf="exports/dspark-{rollout_id}",
        num_layers=36,
        vllm_speculative_config=_dspark_spec(model_path),
    )

    validate_external_draft_args(args)

    assert args.draft_save_hf == str(tmp_path / "exports" / "dspark-{rollout_id}")


@pytest.mark.unit
def test_export_request_without_dspark_or_external_training_is_rejected():
    with pytest.raises(ValueError, match="requires explicit --enable-external-draft-training"):
        validate_external_draft_args(
            _args(
                enable_external_draft_training=False,
                draft_save_hf="/exports/draft-{rollout_id}",
            )
        )


@pytest.mark.unit
def test_dspark_training_export_requires_a_draft_model_path():
    with pytest.raises(ValueError, match="requires --draft-model-path"):
        validate_external_draft_args(
            _args(
                enable_external_draft_training=True,
                draft_algorithm="dspark",
                draft_model_path=None,
                draft_save_hf="/exports/draft-{rollout_id}",
                vllm_speculative_config={"method": "dspark"},
            )
        )


@pytest.mark.unit
def test_eagle3_validation_rejects_dspark_hf_export_option():
    with pytest.raises(ValueError, match="only.*dspark"):
        validate_external_draft_args(_args(draft_save_hf="/exports/draft-{rollout_id}"))


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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("debug_train_only", "debug-train-only"),
        ("vocab_mapping", "trainer-only override"),
        ("nonpositive_save_interval", "draft-save-interval"),
        ("missing_spec_model", "config model"),
        ("missing_speculative_tokens", "explicit positive.*num_speculative_tokens"),
        ("synthetic_rejection", "rejection_sample_method='standard'"),
        ("custom_worker_extension", "vLLMWorkerExtension"),
    ],
)
def test_dspark_validation_rejects_unsupported_modes_from_valid_baseline(case, message, tmp_path, monkeypatch):
    model_path = tmp_path / "dspark"
    model_path.mkdir()
    _write_mock_local_dspark_checkpoint(model_path, monkeypatch)
    overrides = {
        "draft_algorithm": "dspark",
        "draft_model_path": str(model_path),
        "num_layers": 36,
        "vllm_speculative_config": _dspark_spec(model_path),
    }
    case_overrides = {
        "debug_train_only": {"debug_train_only": True},
        "vocab_mapping": {"draft_vocab_mapping_path": "/tmp/map.pt"},
        "nonpositive_save_interval": {"draft_save_interval": 0},
        "missing_spec_model": {"vllm_speculative_config": {"method": "dspark", "num_speculative_tokens": 1}},
        "missing_speculative_tokens": {"vllm_speculative_config": {"method": "dspark", "model": str(model_path)}},
        "synthetic_rejection": {
            "vllm_speculative_config": {
                **_dspark_spec(model_path),
                "rejection_sample_method": "synthetic",
            }
        },
        "custom_worker_extension": {"vllm_worker_extension_cls": "custom.Extension"},
    }
    overrides.update(case_overrides[case])

    with pytest.raises(ValueError, match=message):
        validate_external_draft_args(_args(**overrides))


@pytest.mark.unit
@pytest.mark.parametrize(
    "override",
    [
        {"debug_train_only": True},
        {"draft_vocab_mapping_path": "/tmp/map.pt"},
        {"vllm_speculative_config": {"method": "eagle3"}},
        {
            "vllm_speculative_config": {
                "method": "eagle3",
                "model": "/models/eagle3",
                "rejection_sample_method": "synthetic",
            }
        },
        {"vllm_worker_extension_cls": "custom.Extension"},
    ],
)
def test_dspark_only_guards_do_not_reject_eagle3(override):
    validate_external_draft_args(_args(**override))
