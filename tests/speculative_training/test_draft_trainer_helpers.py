import json
from argparse import Namespace

import pytest
import torch

from vime.backends.speculative_training.draft_trainer import (
    ExternalDraftTrainer,
    _architecture_fingerprint,
    _load_draft_model,
    _load_target_embedding,
    _resolve_draft_vocab_state,
)


class _Draft(torch.nn.Module):
    def __init__(self, rows=4, hidden=3):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(rows, hidden)
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.lm_head = torch.nn.Linear(hidden, rows, bias=False)
        self.verifier_lm_head = torch.nn.Linear(hidden, rows, bias=False)


def _prime_candidate(trainer, target_version="9"):
    with torch.no_grad():
        trainer.model.verifier_lm_head.weight.copy_(trainer.model.lm_head.weight)
    trainer.target_weight_version = str(target_version)
    trainer._capture_candidate_state()


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
def test_dspark_version_zero_artifact_keeps_initial_candidate_head():
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.algorithm = "dspark"
    trainer.draft_version = 0
    trainer.target_weight_version = None
    trainer.model = _Draft()
    with torch.no_grad():
        trainer.model.lm_head.weight.fill_(1)
        trainer.model.verifier_lm_head.weight.fill_(1)
    trainer._capture_candidate_state()

    with torch.no_grad():
        trainer.model.lm_head.weight.fill_(2)
        trainer.model.verifier_lm_head.weight.fill_(2)

    state = trainer._candidate_model_state_dict()
    assert torch.equal(state["lm_head.weight"], torch.ones_like(state["lm_head.weight"]))
    assert torch.equal(
        state["verifier_lm_head.weight"],
        torch.ones_like(state["verifier_lm_head.weight"]),
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
def test_eagle3_model_loading_ignores_dspark_only_config_fields(monkeypatch):
    import vime.backends.speculative_training.draft_trainer as module

    class _EagleDraft(_Draft):
        def forward(self, input_ids, hidden_states, loss_mask):
            del input_ids, hidden_states, loss_mask

    model = _EagleDraft()
    model.config = Namespace(
        target_hidden_size=3,
        num_aux_hidden_states=None,
        transformer_layer_config=Namespace(hidden_size=999),
        aux_hidden_state_layer_ids=[1, 99],
        eagle_aux_hidden_state_layer_ids=None,
        target_hidden_layer_ids=None,
        eagle_config=None,
    )
    monkeypatch.setattr(module, "load_function", lambda path: lambda args, device: model)
    args = Namespace(
        draft_algorithm="eagle3",
        draft_model_factory_path="tests.eagle_factory",
        draft_feature_layer_ids=[2, 4, 6],
        hidden_size=3,
        num_layers=8,
    )

    assert _load_draft_model(args, torch.device("cpu")) is model


@pytest.mark.unit
def test_draft_vocab_rows_are_recomputed_from_current_model_state():
    model = _Draft(rows=4, hidden=3)
    model.lm_head = torch.nn.Linear(3, 2, bias=False)
    model.register_buffer("t2d", torch.tensor([True, False, True, False]))

    first_rows, draft_vocab_size = _resolve_draft_vocab_state(model)
    model.t2d.copy_(torch.tensor([False, True, False, True]))
    resumed_rows, resumed_vocab_size = _resolve_draft_vocab_state(model)

    assert first_rows.tolist() == [0, 2]
    assert resumed_rows.tolist() == [1, 3]
    assert draft_vocab_size == resumed_vocab_size == 2


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
    with torch.no_grad():
        model.lm_head.weight.fill_(1)
        model.verifier_lm_head.weight.fill_(1)
    _prime_candidate(trainer)
    with torch.no_grad():
        model.lm_head.weight.fill_(2)
        model.verifier_lm_head.weight.fill_(2)
    trainer.target_weight_version = "10"

    snapshot = trainer.prepare_publish_snapshot()

    assert snapshot["algorithm"] == "dspark"
    names = {name for name, _ in snapshot["named_tensors"]}
    assert "lm_head.weight" in names
    assert "embed_tokens.weight" not in names
    assert "verifier_lm_head.weight" not in names
    assert "verifier_norm.weight" not in names
    assert "verifier_norm.bias" not in names
    tensors = dict(snapshot["named_tensors"])
    assert tensors["lm_head.weight"].dtype == torch.bfloat16
    assert torch.equal(tensors["lm_head.weight"], torch.ones_like(tensors["lm_head.weight"]))
    assert snapshot["trained_against_target_version"] == "9"
    assert tensors["confidence_head.weight"].dtype == torch.float32
    assert tensors["confidence_head.bias"].dtype == torch.float32


@pytest.mark.unit
def test_dspark_publish_filters_training_only_tensors_from_export_hook():
    class _ExportingDraft(_Draft):
        def export_for_vllm(self, *, dtype, device):
            del dtype, device
            return {
                "proj.weight": self.proj.weight,
                "lm_head.weight": self.lm_head.weight,
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
    _prime_candidate(trainer)

    snapshot = trainer.prepare_publish_snapshot()

    tensors = dict(snapshot["named_tensors"])
    assert set(tensors) == {"proj.weight", "lm_head.weight", "model.d2t"}
    assert tensors["model.d2t"].dtype == torch.long


@pytest.mark.unit
def test_eagle3_publish_keeps_legacy_snapshot_schema_and_dtype_conversion():
    class _ExportingEagle(_Draft):
        def export_for_vllm(self, *, dtype, device):
            del dtype, device
            return {"token_ids": torch.tensor([1, 2], dtype=torch.long)}

    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "eagle3"
    trainer.draft_version = 2
    trainer.target_weight_version = "9"
    trainer.args = Namespace(draft_publish_dtype="bf16")
    trainer.model = _ExportingEagle()
    trainer.architecture_fingerprint = "fingerprint"

    snapshot = trainer.prepare_publish_snapshot()

    assert "algorithm" not in snapshot
    assert dict(snapshot["named_tensors"])["token_ids"].dtype == torch.bfloat16


@pytest.mark.unit
def test_dspark_export_uses_custom_rollout_directory(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    load_file = safetensors_torch.load_file
    save_file = safetensors_torch.save_file

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
        save_file(kwargs["state_dict"], path / "model.safetensors")
        (path / "generation_config.json").write_text("{}", encoding="utf-8")

    model.save_pretrained = save_pretrained
    expected = tmp_path / "export-7"
    expected.write_bytes(b"replace an existing non-directory artifact")
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
    _prime_candidate(trainer)

    result = trainer.export_hf_model(7)

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
    saved_state_dict = load_file(expected / "model.safetensors", device="cpu")
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
    assert saved_config["dflash_config"] == {"mask_token_id": 3, "target_layer_ids": [1, 3]}
    assert saved_config["dspark_bonus_anchor"] is False
    assert saved_config["transformer_layer_config"]["hidden_size"] == 3
    assert saved_config["speculators_model_type"] == "dspark"
    assert {path.name for path in expected.iterdir()} == {"config.json", "model.safetensors"}
    assert not list(tmp_path.glob(".export-7.tmp-*"))
    assert not list(tmp_path.glob(".export-7.backup-*"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("artifact", "message"),
    [("missing", "did not produce a complete model directory"), ("invalid", "invalid model.safetensors")],
)
def test_dspark_export_rejects_invalid_weight_artifacts(tmp_path, artifact, message):
    model = _Draft()

    def save_pretrained(path, **kwargs):
        del kwargs
        (path / "config.json").write_text("{}", encoding="utf-8")
        if artifact == "invalid":
            (path / "model.safetensors").write_bytes(b"not-a-safetensors-file")

    model.save_pretrained = save_pretrained
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
    _prime_candidate(trainer)

    with pytest.raises(RuntimeError, match=message):
        trainer.export_hf_model(2)

    assert not (tmp_path / "incomplete").exists()
    assert not list(tmp_path.glob(".incomplete.tmp-*"))
    assert not list(tmp_path.glob(".incomplete.backup-*"))


@pytest.mark.unit
def test_failed_dspark_directory_swap_restores_previous_export(tmp_path, monkeypatch):
    save_file = pytest.importorskip("safetensors.torch").save_file
    import vime.backends.speculative_training.draft_trainer as module

    output = tmp_path / "export"
    output.mkdir()
    marker = output / "previous-model.bin"
    marker.write_bytes(b"previous weights")
    model = _Draft()

    def save_pretrained(path, **kwargs):
        (path / "config.json").write_text(_serialized_dspark_config(), encoding="utf-8")
        save_file(kwargs["state_dict"], path / "model.safetensors")

    model.save_pretrained = save_pretrained
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
    _prime_candidate(trainer)

    real_replace = module.os.replace

    def fail_new_directory_swap(source, destination):
        if str(source).startswith(str(output.parent / f".{output.name}.tmp-")) and destination == output:
            raise OSError("injected directory swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_new_directory_swap)

    with pytest.raises(OSError, match="injected directory swap failure"):
        trainer.export_hf_model(2)

    assert marker.read_bytes() == b"previous weights"
    assert not list(tmp_path.glob(".export.tmp-*"))
    assert not list(tmp_path.glob(".export.backup-*"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("collision", "message"),
    [("source", "original --draft-model-path"), ("actor", "Actor --save-hf")],
)
def test_dspark_export_rejects_model_directory_collisions(tmp_path, collision, message):
    output = tmp_path / "model-{rollout_id}"
    source = tmp_path / "model-3" if collision == "source" else tmp_path / "original"
    if collision == "source":
        source.mkdir()
    model = _Draft()
    model.save_pretrained = lambda *args, **kwargs: pytest.fail("must not overwrite another model")
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.model = model
    trainer.args = Namespace(
        draft_model_path=str(source),
        draft_save_hf=str(output),
        save_hf=str(output) if collision == "actor" else None,
    )

    with pytest.raises(ValueError, match=message):
        trainer.export_hf_model(3)
