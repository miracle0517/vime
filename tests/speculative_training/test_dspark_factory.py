from argparse import Namespace
from types import SimpleNamespace

import pytest

from vime.backends.speculative_training.factories import speculators_dspark


class _ConfigType:
    received = None

    @classmethod
    def from_dict(cls, value):
        cls.received = value
        return SimpleNamespace(aux_hidden_state_layer_ids=value["aux_hidden_state_layer_ids"])


class _DefaultingConfigType:
    @classmethod
    def from_dict(cls, value):
        """Emulate a runtime that defaults a missing nested Qwen config."""

        return SimpleNamespace(
            transformer_layer_config=SimpleNamespace(
                model_type="qwen3",
                hidden_size=4096,
                intermediate_size=22016,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                head_dim=128,
                vocab_size=151936,
                layer_types=["full_attention"] * 32,
                max_window_layers=32,
            ),
            aux_hidden_state_layer_ids=value["aux_hidden_state_layer_ids"],
            draft_vocab_size=32000,
            markov_rank=128,
            enable_confidence_head=False,
            confidence_head_with_markov=False,
            target_hidden_size=4096,
        )


def _qwen4b_dspark_shapes():
    shapes = {
        "fc.weight": (2560, 12800),
        "hidden_norm.weight": (2560,),
        "norm.weight": (2560,),
        "lm_head.weight": (151936, 2560),
        "markov_head.markov_w1.weight": (151936, 256),
        "markov_head.markov_w2.weight": (151936, 256),
        "confidence_head.proj.weight": (1, 2816),
        "confidence_head.proj.bias": (1,),
    }
    for layer_id in range(5):
        prefix = f"layers.{layer_id}"
        shapes.update(
            {
                f"{prefix}.self_attn.q_proj.weight": (4096, 2560),
                f"{prefix}.self_attn.k_proj.weight": (1024, 2560),
                f"{prefix}.self_attn.v_proj.weight": (1024, 2560),
                f"{prefix}.self_attn.o_proj.weight": (2560, 4096),
                f"{prefix}.self_attn.q_norm.weight": (128,),
                f"{prefix}.self_attn.k_norm.weight": (128,),
                f"{prefix}.mlp.gate_proj.weight": (9728, 2560),
                f"{prefix}.mlp.up_proj.weight": (9728, 2560),
                f"{prefix}.mlp.down_proj.weight": (2560, 9728),
                f"{prefix}.input_layernorm.weight": (2560,),
                f"{prefix}.post_attention_layernorm.weight": (2560,),
            }
        )
    return shapes


@pytest.mark.unit
def test_dspark_factory_preserves_checkpoint_layer_ids(monkeypatch):
    checkpoint_config = {
        "aux_hidden_state_layer_ids": [2, 14, 29],
        "speculators_model_type": "legacy-value",
        "speculators_config": None,
        "transformer_layer_config": {},
    }
    monkeypatch.setattr(
        speculators_dspark,
        "load_draft_checkpoint_config",
        lambda args: checkpoint_config,
    )
    args = Namespace(draft_feature_layer_ids=[2, 14, 29])

    speculators_dspark._build_config(args, _ConfigType)

    assert _ConfigType.received["aux_hidden_state_layer_ids"] == [2, 14, 29]
    assert _ConfigType.received["speculators_model_type"] == "dspark"
    assert "speculators_config" not in _ConfigType.received
    assert checkpoint_config["speculators_model_type"] == "legacy-value"
    assert checkpoint_config["speculators_config"] is None


@pytest.mark.unit
def test_dspark_factory_fills_missing_layer_ids_from_resolved_args(monkeypatch):
    monkeypatch.setattr(
        speculators_dspark,
        "load_draft_checkpoint_config",
        lambda args: {"aux_hidden_state_layer_ids": None, "transformer_layer_config": {}},
    )
    args = Namespace(draft_model_path="/not/local", draft_feature_layer_ids=[2, 16, 29])

    config = speculators_dspark._build_config(args, _ConfigType)

    assert config.aux_hidden_state_layer_ids == [2, 16, 29]


@pytest.mark.unit
def test_dspark_factory_uses_fc_shape_to_reject_wrong_implicit_layer_count(monkeypatch):
    monkeypatch.setattr(
        speculators_dspark,
        "_checkpoint_tensor_shapes",
        lambda path: {"fc.weight": (64, 256)},
    )
    monkeypatch.setattr(
        speculators_dspark,
        "load_draft_checkpoint_config",
        lambda args: {
            "aux_hidden_state_layer_ids": None,
            "transformer_layer_config": {"hidden_size": 64},
        },
    )
    args = Namespace(draft_model_path="/models/dspark", draft_feature_layer_ids=[2, 16, 29])

    with pytest.raises(ValueError, match="expects 4 auxiliary hidden states"):
        speculators_dspark._build_config(args, _ConfigType)


@pytest.mark.unit
def test_dspark_factory_recovers_old_config_head_layout():
    config = {
        "aux_hidden_state_layer_ids": None,
        "transformer_layer_config": {"hidden_size": 64},
    }
    shapes = {
        "fc.weight": (64, 192),
        "layers.0.self_attn.q_proj.weight": (64, 64),
        "lm_head.weight": (1000, 64),
        "markov_head.markov_w1.weight": (2000, 32),
        "markov_head.markov_w2.weight": (1000, 32),
        "confidence_head.proj.weight": (1, 96),
    }

    speculators_dspark._recover_missing_layout(config, (2, 16, 29), shapes)

    assert config["aux_hidden_state_layer_ids"] == [2, 16, 29]
    assert config["draft_vocab_size"] == 1000
    assert config["markov_rank"] == 32
    assert config["enable_confidence_head"] is True
    assert config["confidence_head_with_markov"] is True


@pytest.mark.unit
def test_dspark_factory_disables_absent_optional_heads_from_local_weights():
    config = {
        "aux_hidden_state_layer_ids": [2, 16, 29],
        "markov_rank": 256,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "transformer_layer_config": {"hidden_size": 64},
    }
    shapes = {
        "fc.weight": (64, 192),
        "lm_head.weight": (1000, 64),
    }

    speculators_dspark._recover_missing_layout(config, (2, 16, 29), shapes)

    assert config["markov_rank"] == 0
    assert config["enable_confidence_head"] is False
    assert config["confidence_head_with_markov"] is False


@pytest.mark.unit
def test_dspark_factory_accepts_disabled_markov_and_confidence_heads():
    config = SimpleNamespace(
        transformer_layer_config=SimpleNamespace(
            hidden_size=64,
            vocab_size=1000,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            intermediate_size=128,
        ),
        aux_hidden_state_layer_ids=[2, 16, 29],
        draft_vocab_size=1000,
        markov_rank=0,
        enable_confidence_head=False,
        confidence_head_with_markov=False,
    )
    shapes = {
        "fc.weight": (64, 192),
        "hidden_norm.weight": (64,),
        "norm.weight": (64,),
        "lm_head.weight": (1000, 64),
        "layers.0.self_attn.q_proj.weight": (64, 64),
        "layers.0.self_attn.k_proj.weight": (64, 64),
        "layers.0.self_attn.v_proj.weight": (64, 64),
        "layers.0.self_attn.o_proj.weight": (64, 64),
        "layers.0.self_attn.q_norm.weight": (64,),
        "layers.0.self_attn.k_norm.weight": (64,),
        "layers.0.mlp.gate_proj.weight": (128, 64),
        "layers.0.mlp.up_proj.weight": (128, 64),
        "layers.0.mlp.down_proj.weight": (64, 128),
        "layers.0.input_layernorm.weight": (64,),
        "layers.0.post_attention_layernorm.weight": (64,),
    }

    speculators_dspark._validate_checkpoint_layout(config, shapes)


@pytest.mark.unit
def test_dspark_factory_recovers_stale_transformer_layout_from_weights(monkeypatch):
    config = {
        "aux_hidden_state_layer_ids": [2, 9, 18, 27, 33],
        "draft_vocab_size": 32000,
        "markov_rank": 128,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "transformer_layer_config": {
            "hidden_size": 4096,
            "intermediate_size": 20480,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "layer_types": ["full_attention"] * 32,
            "max_window_layers": 32,
        },
    }
    shapes = _qwen4b_dspark_shapes()

    monkeypatch.setattr(speculators_dspark, "load_draft_checkpoint_config", lambda args: config)
    monkeypatch.setattr(speculators_dspark, "_checkpoint_tensor_shapes", lambda path: shapes)
    args = Namespace(draft_model_path="/models/dspark", draft_feature_layer_ids=[2, 9, 18, 27, 33])
    recovered = speculators_dspark._build_config(args, _DefaultingConfigType)

    transformer = vars(recovered.transformer_layer_config)
    assert transformer["hidden_size"] == 2560
    assert transformer["vocab_size"] == 151936
    assert transformer["intermediate_size"] == 9728
    assert transformer["num_hidden_layers"] == 5
    assert transformer["num_attention_heads"] == 32
    assert transformer["num_key_value_heads"] == 8
    assert transformer["layer_types"] == ["full_attention"] * 5
    assert transformer["max_window_layers"] == 5
    assert recovered.target_hidden_size == 2560
    assert recovered.draft_vocab_size == 151936
    assert recovered.markov_rank == 256
    assert recovered.enable_confidence_head is True
    assert recovered.confidence_head_with_markov is True
    assert config["transformer_layer_config"]["hidden_size"] == 4096


@pytest.mark.unit
def test_dspark_factory_recovers_completely_missing_transformer_config(monkeypatch):
    config = {
        "aux_hidden_state_layer_ids": [2, 9, 18, 27, 33],
        "draft_vocab_size": 32000,
        "markov_rank": 128,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    shapes = _qwen4b_dspark_shapes()
    monkeypatch.setattr(speculators_dspark, "load_draft_checkpoint_config", lambda args: config)
    monkeypatch.setattr(speculators_dspark, "_checkpoint_tensor_shapes", lambda path: shapes)
    monkeypatch.setattr(
        speculators_dspark,
        "_target_transformer_config",
        lambda args: {"model_type": "qwen3", "rope_theta": 1000000.0, "rms_norm_eps": 1e-6},
    )
    args = Namespace(draft_model_path="/models/dspark", draft_feature_layer_ids=[2, 9, 18, 27, 33])

    recovered = speculators_dspark._build_config(args, _DefaultingConfigType)

    transformer = recovered.transformer_layer_config
    assert transformer.num_hidden_layers == 5
    assert transformer.hidden_size == 2560
    assert transformer.intermediate_size == 9728
    assert transformer.num_attention_heads == 32
    assert transformer.num_key_value_heads == 8
    assert transformer.rope_theta == 1000000.0
    assert transformer.rope_parameters == {"rope_type": "default", "rope_theta": 1000000.0}
    assert transformer.rms_norm_eps == 1e-6
    assert recovered.target_hidden_size == 2560
    assert recovered.markov_rank == 256
    assert recovered.confidence_head_with_markov is True
    assert "transformer_layer_config" not in config


@pytest.mark.unit
def test_dspark_factory_normalizes_legacy_rope_scaling():
    config = {
        "transformer_layer_config": {
            "model_type": "qwen3",
            "rope_theta": 500000.0,
            "rope_scaling": {"type": "yarn", "factor": 4.0},
        }
    }

    speculators_dspark._normalize_qwen_rope_config(config, Namespace())

    assert config["transformer_layer_config"]["rope_parameters"] == {
        "rope_type": "yarn",
        "factor": 4.0,
        "rope_theta": 500000.0,
    }


@pytest.mark.unit
def test_dspark_factory_fills_new_rope_parameters_from_megatron_args():
    config = {"transformer_layer_config": {"model_type": "qwen3"}}

    speculators_dspark._normalize_qwen_rope_config(config, Namespace(rotary_base=250000.0))

    assert config["transformer_layer_config"]["rope_parameters"] == {
        "rope_type": "default",
        "rope_theta": 250000.0,
    }


@pytest.mark.unit
def test_dspark_factory_preserves_target_rope_scaling(monkeypatch):
    config = {"transformer_layer_config": {"model_type": "qwen3"}}
    monkeypatch.setattr(
        speculators_dspark,
        "_target_transformer_config",
        lambda args: {
            "model_type": "qwen3",
            "rope_theta": 750000.0,
            "rope_scaling": {"type": "linear", "factor": 2.0},
        },
    )

    speculators_dspark._normalize_qwen_rope_config(config, Namespace())

    assert config["transformer_layer_config"]["rope_parameters"] == {
        "rope_type": "linear",
        "factor": 2.0,
        "rope_theta": 750000.0,
    }


@pytest.mark.unit
def test_dspark_factory_rejects_invalid_rope_parameters():
    config = {"transformer_layer_config": {"model_type": "qwen3", "rope_parameters": []}}

    with pytest.raises(TypeError, match="rope_parameters must be a JSON object"):
        speculators_dspark._normalize_qwen_rope_config(config, Namespace())


@pytest.mark.unit
def test_dspark_factory_uses_nested_transformer_config_without_local_weights():
    config = {
        "draft_model_config": {
            "hf_config": {
                "model_type": "qwen3",
                "hidden_size": 2560,
                "num_hidden_layers": 5,
            }
        }
    }

    speculators_dspark._ensure_transformer_config(config, Namespace(), {})

    assert config["transformer_layer_config"] == {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_hidden_layers": 5,
    }


@pytest.mark.unit
def test_dspark_factory_rejects_unrecoverable_remote_config():
    with pytest.raises(ValueError, match="omits transformer_layer_config"):
        speculators_dspark._ensure_transformer_config({}, Namespace(), {})


@pytest.mark.unit
def test_dspark_factory_rejects_target_hidden_size_mismatch_before_ray():
    config = SimpleNamespace(
        transformer_layer_config=SimpleNamespace(
            model_type="qwen3",
            hidden_size=2560,
            vocab_size=151936,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        ),
        target_hidden_size=2560,
        markov_head_type="vanilla",
        markov_rank=256,
    )
    args = Namespace(hidden_size=4096)

    with pytest.raises(ValueError, match="configured Megatron Target uses hidden_size=4096"):
        speculators_dspark._validate_dspark_config(args, config)


@pytest.mark.unit
def test_dspark_factory_checks_mask_token_against_verifier_vocab():
    config = SimpleNamespace(
        transformer_layer_config=SimpleNamespace(
            model_type="qwen3",
            hidden_size=64,
            vocab_size=200,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        ),
        target_hidden_size=64,
        draft_vocab_size=100,
        markov_head_type="vanilla",
        markov_rank=0,
        enable_confidence_head=False,
        confidence_head_with_markov=False,
        mask_token_id=250,
    )
    args = Namespace(hidden_size=64)

    with pytest.raises(ValueError, match=r"outside verifier vocabulary \[0, 200\)"):
        speculators_dspark._validate_dspark_config(args, config)


@pytest.mark.unit
def test_dspark_factory_rejects_qwen3_variants_not_supported_by_backbone():
    config = SimpleNamespace(transformer_layer_config=SimpleNamespace(model_type="qwen3_next"))

    with pytest.raises(ValueError, match="only dense Qwen3"):
        speculators_dspark._validate_dspark_config(Namespace(), config)


@pytest.mark.unit
def test_dspark_factory_rejects_missing_enabled_head_weights():
    config = SimpleNamespace(
        transformer_layer_config=SimpleNamespace(
            hidden_size=64,
            vocab_size=2000,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            intermediate_size=128,
        ),
        aux_hidden_state_layer_ids=[2, 16, 29],
        draft_vocab_size=1000,
        markov_rank=32,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    shapes = {
        "fc.weight": (64, 192),
        "layers.0.self_attn.q_proj.weight": (64, 64),
        "lm_head.weight": (1000, 64),
        "markov_head.markov_w1.weight": (2000, 32),
        "markov_head.markov_w2.weight": (1000, 32),
    }

    with pytest.raises(ValueError, match="confidence_head.proj.weight: missing"):
        speculators_dspark._validate_checkpoint_layout(config, shapes)


@pytest.mark.unit
def test_dspark_factory_rejects_layer_override_before_loading_weights(monkeypatch):
    monkeypatch.setattr(
        speculators_dspark,
        "load_draft_checkpoint_config",
        lambda args: {"aux_hidden_state_layer_ids": [2, 14, 29], "transformer_layer_config": {}},
    )
    args = Namespace(draft_feature_layer_ids=[2, 10, 20, 29])

    with pytest.raises(ValueError, match="Remove the command-line override"):
        speculators_dspark._build_config(args, _ConfigType)


@pytest.mark.unit
def test_dspark_factory_explains_checkpoint_shape_mismatch():
    class _ModelType:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("You set `ignore_mismatched_sizes` to `False`")

    config = SimpleNamespace(
        transformer_layer_config=SimpleNamespace(
            model_type="qwen3",
            hidden_size=2560,
            vocab_size=151936,
        ),
        aux_hidden_state_layer_ids=[2, 14, 29],
        draft_vocab_size=151936,
        markov_rank=256,
        enable_confidence_head=True,
    )
    args = Namespace(draft_model_path="/models/dspark")

    with pytest.raises(RuntimeError, match="randomly initialize") as error:
        speculators_dspark._load_pretrained_model(_ModelType, args, config)

    assert "hidden_size=2560" in str(error.value)
    assert "markov_rank=256" in str(error.value)
