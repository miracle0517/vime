import json
from argparse import Namespace

import pytest
import torch

from vime.backends.speculative_training.draft_trainer import (
    ExternalDraftTrainer,
    _architecture_fingerprint,
    _load_target_embedding,
)


class _Draft(torch.nn.Module):
    def __init__(self, rows=4, hidden=3):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(rows, hidden)
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)


def _serialized_dspark_config() -> str:
    return json.dumps(
        {
            "speculators_model_type": "dspark",
            "speculators_config": None,
            "aux_hidden_state_layer_ids": [2, 4],
            "block_size": 7,
            "mask_token_id": 3,
            "markov_rank": 2,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "sample_from_anchor": True,
            "transformer_layer_config": {
                "model_type": "qwen3",
                "hidden_size": 3,
                "intermediate_size": 6,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 4,
            },
        }
    )


@pytest.mark.unit
def test_generic_target_embedding_loader_reads_pytorch_checkpoint(tmp_path):
    source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    torch.save({"model.embed_tokens.weight": source}, tmp_path / "pytorch_model.bin")
    model = _Draft()
    args = Namespace(
        draft_target_embedding_path=str(tmp_path),
        draft_target_embedding_key="model.embed_tokens.weight",
        hf_checkpoint=None,
    )

    _load_target_embedding(model, args)

    assert torch.equal(model.embed_tokens.weight, source)


@pytest.mark.unit
def test_architecture_fingerprint_changes_with_parameter_layout():
    assert _architecture_fingerprint(_Draft(rows=4)) != _architecture_fingerprint(_Draft(rows=5))


@pytest.mark.unit
def test_actor_colocated_trainer_does_not_join_actor_process_group(monkeypatch):
    import vime.backends.speculative_training.draft_trainer as module

    model = _Draft()
    monkeypatch.setattr(module, "is_npu", lambda: False)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "_load_draft_model", lambda args, device: model)
    monkeypatch.setattr(module, "_load_target_embedding", lambda model, args: None)
    monkeypatch.setattr(module.dist, "get_rank", lambda: pytest.fail("must not query Actor rank"))
    monkeypatch.setattr(module.dist, "get_world_size", lambda: pytest.fail("must not query Actor world size"))
    args = Namespace(
        draft_freeze_embeddings=True,
        draft_vocab_mapping_path=None,
        draft_learning_rate=1e-5,
        draft_weight_decay=0.0,
        draft_lr_warmup_steps=0,
        draft_lr_total_steps=1,
        draft_lr_scheduler_type="constant",
        draft_train_interval=1,
        draft_train_steps_per_trigger=1,
        draft_queue_max_samples=4,
        num_rollout=1,
        draft_checkpoint_path=None,
    )

    trainer = ExternalDraftTrainer(args, distributed=False)

    assert trainer.rank == 0
    assert trainer.world_size == 1
    assert trainer.model is model


@pytest.mark.unit
def test_dspark_publish_includes_synced_frozen_lm_head():
    model = _Draft(rows=4, hidden=3)
    model.embed_tokens.weight.requires_grad_(False)
    model.lm_head = torch.nn.Linear(3, 4, bias=False)
    model.lm_head.weight.requires_grad_(False)
    model.verifier_lm_head = torch.nn.Linear(3, 4, bias=False)
    model.verifier_norm = torch.nn.LayerNorm(3)
    model.confidence_head = torch.nn.Linear(3, 1)
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.draft_version = 2
    trainer.target_weight_version = "9"
    trainer.algorithm = "dspark"
    trainer.args = Namespace(draft_publish_dtype="bf16")
    trainer.model = model
    trainer.architecture_fingerprint = "fingerprint"

    snapshot = trainer.prepare_publish_snapshot()

    assert snapshot["algorithm"] == "dspark"
    names = {name for name, _ in snapshot["named_tensors"]}
    assert "lm_head.weight" in names
    assert "embed_tokens.weight" not in names
    assert "verifier_lm_head.weight" not in names
    assert "verifier_norm.weight" not in names
    assert "verifier_norm.bias" not in names
    assert "t2d" not in names
    tensors = dict(snapshot["named_tensors"])
    assert tensors["lm_head.weight"].dtype == torch.bfloat16
    assert tensors["confidence_head.weight"].dtype == torch.float32
    assert tensors["confidence_head.bias"].dtype == torch.float32


@pytest.mark.unit
def test_dspark_publish_filters_training_only_tensors_from_export_hook():
    class _ExportingDraft(_Draft):
        def export_for_vllm(self, *, dtype, device):
            del dtype, device
            return {
                "proj.weight": self.proj.weight,
                "verifier_lm_head.weight": torch.ones(4, 3),
                "model.verifier_norm.weight": torch.ones(3),
                "model.t2d": torch.ones(4, dtype=torch.bool),
                "model.d2t": torch.arange(3, dtype=torch.long),
            }

    model = _ExportingDraft()
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.draft_version = 2
    trainer.target_weight_version = "9"
    trainer.algorithm = "dspark"
    trainer.args = Namespace(draft_publish_dtype="bf16")
    trainer.model = model
    trainer.architecture_fingerprint = "fingerprint"

    snapshot = trainer.prepare_publish_snapshot()

    tensors = dict(snapshot["named_tensors"])
    assert set(tensors) == {"proj.weight", "model.d2t"}
    assert tensors["model.d2t"].dtype == torch.long


@pytest.mark.unit
def test_dspark_export_uses_custom_rollout_directory(tmp_path):
    model = _Draft()
    calls = []

    def save_pretrained(path, **kwargs):
        calls.append(
            (
                path,
                kwargs["safe_serialization"],
                kwargs["max_shard_size"],
                {name: (tensor.device.type, tensor.is_contiguous()) for name, tensor in kwargs["state_dict"].items()},
            )
        )
        (path / "config.json").write_text(
            _serialized_dspark_config(),
            encoding="utf-8",
        )
        torch.save(kwargs["state_dict"], path / "model.safetensors")
        (path / "generation_config.json").write_text("{}", encoding="utf-8")

    model.save_pretrained = save_pretrained
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.draft_version = 3
    trainer.architecture_fingerprint = "fingerprint"
    trainer.model = model
    trainer.args = Namespace(
        draft_model_path=str(tmp_path / "original"),
        draft_save_hf=str(tmp_path / "export-{rollout_id}"),
        save_hf=None,
    )

    result = trainer.export_speculators_model(7)

    expected = tmp_path / "export-7"
    assert result["path"] == str(expected.resolve())
    assert result["complete"] is True
    assert result["weight_files"] == ["model.safetensors"]
    assert result["weight_bytes"] == (expected / "model.safetensors").stat().st_size
    assert result["draft_version"] == 3
    assert result["rollout_id"] == 7
    assert expected.is_dir()
    assert len(calls) == 1
    save_path, safe_serialization, max_shard_size, saved_tensors = calls[0]
    assert save_path.parent == expected.parent
    assert save_path.name.startswith(f".{expected.name}.tmp-")
    assert safe_serialization is True
    assert max_shard_size == "100GB"
    assert set(saved_tensors) == set(model.state_dict())
    assert all(device == "cpu" and contiguous for device, contiguous in saved_tensors.values())
    saved_state_dict = torch.load(expected / "model.safetensors", map_location="cpu", weights_only=True)
    assert set(saved_state_dict) == set(model.state_dict())
    assert all(torch.equal(saved_state_dict[name], value.cpu()) for name, value in model.state_dict().items())
    saved_config = json.loads((expected / "config.json").read_text(encoding="utf-8"))
    assert "speculators_config" not in saved_config
    assert saved_config["model_type"] == "qwen3"
    assert saved_config["architectures"] == ["Qwen3DSparkModel"]
    assert saved_config["hidden_size"] == 3
    assert saved_config["num_hidden_layers"] == 1
    assert saved_config["vocab_size"] == 4
    assert saved_config["target_layer_ids"] == [1, 3]
    assert saved_config["eagle_aux_hidden_state_layer_ids"] == [2, 4]
    assert saved_config["transformer_layer_config"]["hidden_size"] == 3
    assert saved_config["speculators_model_type"] == "dspark"
    assert {path.name for path in expected.iterdir()} == {"config.json", "model.safetensors"}
    assert not list(tmp_path.glob(".export-7.tmp-*"))
    assert not list(tmp_path.glob(".export-7.backup-*"))


@pytest.mark.unit
def test_dspark_export_creates_fixed_output_directory(tmp_path):
    model = _Draft()

    def save_pretrained(path, **kwargs):
        (path / "config.json").write_text(
            _serialized_dspark_config(),
            encoding="utf-8",
        )
        torch.save(kwargs["state_dict"], path / "model.safetensors")
        (path / "generation_config.json").write_text("{}", encoding="utf-8")

    model.save_pretrained = save_pretrained
    output = tmp_path / "not-created-yet" / "dspark_qwen3_4b_block7_after"
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.draft_version = 1
    trainer.architecture_fingerprint = "fingerprint"
    trainer.model = model
    trainer.args = Namespace(
        draft_model_path=str(tmp_path / "dspark_qwen3_4b_block7"),
        draft_save_hf=str(output),
        save_hf=None,
    )

    result = trainer.export_hf_model(4)

    assert result["path"] == str(output.resolve())
    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {"config.json", "model.safetensors"}
    saved_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert saved_config["model_type"] == "qwen3"
    assert saved_config["architectures"] == ["Qwen3DSparkModel"]


@pytest.mark.unit
def test_dspark_export_rejects_missing_weight_files(tmp_path):
    model = _Draft()
    model.save_pretrained = lambda path, **kwargs: (path / "config.json").write_text("{}", encoding="utf-8")
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.draft_version = 1
    trainer.model = model
    trainer.args = Namespace(
        draft_model_path=str(tmp_path / "original"),
        draft_save_hf=str(tmp_path / "incomplete"),
        save_hf=None,
    )

    with pytest.raises(RuntimeError, match="did not produce a complete model directory"):
        trainer.export_speculators_model(2)


@pytest.mark.unit
def test_failed_dspark_export_preserves_previous_valid_directory(tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    marker = output / "previous-model.bin"
    marker.write_bytes(b"previous weights")
    model = _Draft()
    model.save_pretrained = lambda path, **kwargs: (path / "config.json").write_text(
        '{"speculators_model_type":"dspark"}', encoding="utf-8"
    )
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.draft_version = 1
    trainer.model = model
    trainer.args = Namespace(
        draft_model_path=str(tmp_path / "original"),
        draft_save_hf=str(output),
        save_hf=None,
    )

    with pytest.raises(RuntimeError, match="did not produce a complete model directory"):
        trainer.export_speculators_model(2)

    assert marker.read_bytes() == b"previous weights"


@pytest.mark.unit
def test_dspark_export_rejects_original_model_directory(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    model = _Draft()
    model.save_pretrained = lambda *args, **kwargs: pytest.fail("must not overwrite original model")
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.model = model
    trainer.args = Namespace(draft_model_path=str(source), draft_save_hf=str(source), save_hf=None)

    with pytest.raises(ValueError, match="must not overwrite"):
        trainer.export_speculators_model(0)


@pytest.mark.unit
def test_dspark_export_rejects_actor_hf_directory(tmp_path):
    model = _Draft()
    model.save_pretrained = lambda *args, **kwargs: pytest.fail("must not overwrite Actor model")
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.model = model
    trainer.args = Namespace(
        draft_model_path=str(tmp_path / "original"),
        draft_save_hf=str(tmp_path / "model-{rollout_id}"),
        save_hf=str(tmp_path / "model-{rollout_id}"),
    )

    with pytest.raises(ValueError, match="Actor --save-hf"):
        trainer.export_speculators_model(3)
