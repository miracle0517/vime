from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vime.backends.speculative_training.config import make_dspark_vllm_compatible_config
from vime.backends.speculative_training.factories import speculators_dspark


class _ConfigType:
    received = None

    @classmethod
    def from_dict(cls, value):
        cls.received = value
        return SimpleNamespace(aux_hidden_state_layer_ids=value.get("aux_hidden_state_layer_ids"))


def _checkpoint_config(**overrides):
    config = {
        "speculators_model_type": "dspark",
        "speculators_config": None,
        "aux_hidden_state_layer_ids": [2, 14, 29],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
        },
        "draft_vocab_size": 256,
        "block_size": 8,
        "target_hidden_size": 64,
        "mask_token_id": 255,
        "markov_rank": 16,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "sample_from_anchor": True,
    }
    config.update(overrides)
    return config


def _validated_config(**overrides):
    config = SimpleNamespace(
        transformer_layer_config=SimpleNamespace(model_type="qwen3", hidden_size=64, vocab_size=256),
        target_hidden_size=64,
        markov_head_type="vanilla",
        markov_rank=16,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        mask_token_id=255,
        block_size=8,
        sample_from_anchor=True,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


@pytest.mark.unit
def test_build_config_normalizes_serialized_null_without_mutating_source(monkeypatch):
    source = _checkpoint_config()
    original = deepcopy(source)
    monkeypatch.setattr(speculators_dspark, "load_draft_checkpoint_config", lambda args: source)

    config = speculators_dspark._build_config(
        Namespace(draft_feature_layer_ids=[2, 14, 29]),
        _ConfigType,
    )

    assert config.aux_hidden_state_layer_ids == [2, 14, 29]
    assert "speculators_config" not in _ConfigType.received
    assert source == original


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"transformer_layer_config": {"model_type": "qwen3"}},
            "speculators_model_type='dspark'",
        ),
        (
            {"speculators_model_type": "dspark"},
            "must contain transformer_layer_config",
        ),
        (
            {
                "speculators_model_type": "dspark",
                "transformer_layer_config": {"model_type": "qwen3_next"},
            },
            "only dense Qwen3",
        ),
    ],
)
def test_build_config_rejects_noncanonical_schema(monkeypatch, config, message):
    monkeypatch.setattr(speculators_dspark, "load_draft_checkpoint_config", lambda args: config)

    with pytest.raises(ValueError, match=message):
        speculators_dspark._build_config(Namespace(draft_feature_layer_ids=[2, 14, 29]), _ConfigType)


@pytest.mark.unit
def test_build_config_rejects_explicit_layer_ids_when_metadata_is_missing(monkeypatch):
    source = _checkpoint_config()
    source.pop("aux_hidden_state_layer_ids")
    monkeypatch.setattr(speculators_dspark, "load_draft_checkpoint_config", lambda args: source)

    with pytest.raises(ValueError, match="must define aux_hidden_state_layer_ids"):
        speculators_dspark._build_config(
            Namespace(draft_feature_layer_ids=[2, 14, 29]),
            _ConfigType,
        )


@pytest.mark.unit
def test_build_config_rejects_layer_override(monkeypatch):
    monkeypatch.setattr(
        speculators_dspark,
        "load_draft_checkpoint_config",
        lambda args: _checkpoint_config(aux_hidden_state_layer_ids=[2, 14, 29]),
    )

    with pytest.raises(ValueError, match="Remove the command-line override"):
        speculators_dspark._build_config(
            Namespace(draft_feature_layer_ids=[2, 10, 20, 29]),
            _ConfigType,
        )


@pytest.mark.unit
@pytest.mark.parametrize("hybrid", [False, True])
def test_installed_speculators_config_parses_supported_contract(monkeypatch, hybrid):
    config_module = pytest.importorskip("speculators.models.dspark.config")
    checkpoint_config = _checkpoint_config()
    if hybrid:
        checkpoint_config = make_dspark_vllm_compatible_config(checkpoint_config)
    monkeypatch.setattr(
        speculators_dspark,
        "load_draft_checkpoint_config",
        lambda args: checkpoint_config,
    )

    config = speculators_dspark._build_config(
        Namespace(draft_feature_layer_ids=[2, 14, 29]),
        config_module.DSparkSpeculatorConfig,
    )

    assert config.speculators_model_type == "dspark"
    assert config.transformer_layer_config.model_type == "qwen3"
    assert config.speculators_config is None


@pytest.mark.unit
def test_validate_dspark_config_accepts_supported_contract():
    speculators_dspark._validate_dspark_config(
        Namespace(hidden_size=64, vllm_speculative_config={"num_speculative_tokens": 4}),
        _validated_config(),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"markov_head_type": "gated"}, "only markov_head_type='vanilla'"),
        ({"markov_rank": -1}, "must be non-negative"),
        ({"block_size": 1}, "must be at least 2"),
        ({"mask_token_id": None}, "must define mask_token_id"),
        ({"mask_token_id": 256}, "outside verifier vocabulary"),
    ],
)
def test_validate_dspark_config_rejects_unsupported_values(override, message):
    with pytest.raises(ValueError, match=message):
        speculators_dspark._validate_dspark_config(Namespace(hidden_size=64), _validated_config(**override))


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        _validated_config(enable_confidence_head=True, confidence_head_with_markov=False),
        _validated_config(enable_confidence_head=True, markov_rank=0),
    ],
)
def test_validate_dspark_config_rejects_invalid_confidence_head(config):
    with pytest.raises(ValueError, match="confidence"):
        speculators_dspark._validate_dspark_config(Namespace(hidden_size=64), config)


@pytest.mark.unit
def test_validate_dspark_config_rejects_target_hidden_size_mismatch():
    with pytest.raises(ValueError, match="configured Megatron Target uses hidden_size=128"):
        speculators_dspark._validate_dspark_config(
            Namespace(hidden_size=128),
            _validated_config(),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sample_from_anchor", "requested", "available"),
    [(True, 9, 8), (False, 8, 7)],
)
def test_validate_dspark_config_rejects_tokens_beyond_checkpoint_capacity(
    sample_from_anchor,
    requested,
    available,
):
    with pytest.raises(ValueError, match=rf"{requested} > {available}"):
        speculators_dspark._validate_dspark_config(
            Namespace(hidden_size=64, vllm_speculative_config={"num_speculative_tokens": requested}),
            _validated_config(sample_from_anchor=sample_from_anchor),
        )


@pytest.mark.unit
def test_load_pretrained_model_explains_checkpoint_mismatch():
    class _ModelType:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("ignore_mismatched_sizes=False")

    config = _validated_config()
    config.aux_hidden_state_layer_ids = [2, 14, 29]
    config.draft_vocab_size = 256
    args = Namespace(draft_model_path="/models/dspark")

    with pytest.raises(RuntimeError, match="do not match config.json") as error:
        speculators_dspark._load_pretrained_model(_ModelType, args, config)

    assert "hidden_size=64" in str(error.value)
    assert "markov_rank=16" in str(error.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("loading_info", "message"),
    [
        ({"missing_keys": ["markov_head.proj.weight"]}, "missing_keys"),
        ({"unexpected_keys": ["layers.99.weight"]}, "unexpected_keys"),
        ({"mismatched_keys": [("confidence_head.weight", (4, 4), (8, 4))]}, "mismatched_keys"),
    ],
)
def test_load_pretrained_model_rejects_partial_or_extra_state(loading_info, message):
    model = torch.nn.Linear(2, 2)

    class _ModelType:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs["output_loading_info"] is True
            return model, loading_info

    with pytest.raises(RuntimeError, match=message):
        speculators_dspark._load_pretrained_model(
            _ModelType,
            Namespace(draft_model_path="/models/dspark", draft_vocab_mapping_path=None),
            _validated_config(),
        )


@pytest.mark.unit
def test_load_pretrained_model_preserves_safe_speculators_verifier_hook():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def load_verifier_weights(self):
            self.calls.append(("verifier",))

    model = _Model()

    class _ModelType:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs["output_loading_info"] is True
            return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}

    result = speculators_dspark._load_pretrained_model(
        _ModelType,
        Namespace(draft_model_path="/models/dspark"),
        _validated_config(),
    )

    assert result is model
    assert model.calls == [("verifier",)]


@pytest.mark.unit
def test_load_pretrained_model_rejects_ignored_but_missing_vocab_mapping():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_draft_vocab = True
            self.verifier_vocab_size = 4
            self.draft_vocab_size = 2
            self.register_buffer("t2d", torch.zeros(4, dtype=torch.bool))
            self.register_buffer("d2t", torch.zeros(2, dtype=torch.long))

    model = _Model()

    class _ModelType:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}

    with pytest.raises(RuntimeError, match="missing a complete t2d/d2t mapping"):
        speculators_dspark._load_pretrained_model(
            _ModelType,
            Namespace(draft_model_path="/models/dspark"),
            _validated_config(),
        )


@pytest.mark.unit
def test_external_vocab_mapping_path_cannot_bypass_checkpoint_validation():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_draft_vocab = True
            self.verifier_vocab_size = 4
            self.draft_vocab_size = 2
            self.register_buffer("t2d", torch.zeros(4, dtype=torch.bool))
            self.register_buffer("d2t", torch.zeros(2, dtype=torch.long))
            self.verifier_calls = 0

        def load_verifier_weights(self):
            self.verifier_calls += 1

    model = _Model()

    class _ModelType:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}

    with pytest.raises(RuntimeError, match="missing a complete t2d/d2t mapping"):
        speculators_dspark._load_pretrained_model(
            _ModelType,
            Namespace(draft_model_path="/models/dspark", draft_vocab_mapping_path="/mappings/dspark.pt"),
            _validated_config(),
        )

    assert model.verifier_calls == 0


@pytest.mark.unit
def test_reduced_vocab_checkpoint_accepts_nontrivial_offset_mapping():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_draft_vocab = True
            self.verifier_vocab_size = 4
            self.draft_vocab_size = 2
            self.register_buffer("t2d", torch.tensor([False, True, False, True]))
            # Target rows [1, 3] minus Draft ids [0, 1].
            self.register_buffer("d2t", torch.tensor([1, 2], dtype=torch.long))
            self.verifier_calls = 0

        def load_verifier_weights(self):
            self.verifier_calls += 1

    model = _Model()

    class _ModelType:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}

    result = speculators_dspark._load_pretrained_model(
        _ModelType,
        Namespace(draft_model_path="/models/dspark"),
        _validated_config(),
    )

    assert result is model
    assert model.verifier_calls == 1
