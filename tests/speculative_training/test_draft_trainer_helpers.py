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
