from __future__ import annotations

import pytest
import torch

from vime.backends.speculative_training.backends import dspark
from vime.backends.speculative_training.backends.dspark import (
    collate_dspark_samples,
    compute_dspark_loss,
    dspark_trainer_kwargs,
    sync_dspark_lm_heads,
)
from vime.backends.speculative_training.feature_schema import DraftFeatureSample


def _sample(sample_id: str, offset: int = 0) -> DraftFeatureSample:
    rows = 10
    positions = torch.arange(offset, offset + rows)
    return DraftFeatureSample(
        input_ids=torch.arange(10, 10 + rows),
        loss_mask=torch.tensor([0] + [1] * (rows - 1), dtype=torch.float32),
        position_ids=positions,
        hidden_positions=positions,
        aux_hidden_states=torch.arange(offset, offset + rows * 6, dtype=torch.float32).reshape(rows, 6),
        final_hidden_states=torch.arange(offset, offset + rows * 2, dtype=torch.float32).reshape(rows, 2),
        rollout_id=1,
        target_weight_version="4",
        original_sample_id=sample_id,
        prompt_length=1,
        response_length=rows - 1,
        window_start=offset,
        window_end=offset + rows,
        aux_layer_ids=(2, 8, 13),
        algorithm="dspark",
        hidden_layout="qwen_dspark_aux_plus_last",
    )


@pytest.mark.unit
def test_dspark_collator_isolates_documents_and_masks_each_tail():
    samples = [_sample("a"), _sample("b", 20)]
    batch = collate_dspark_samples(samples, "cpu", block_size=3)

    assert tuple(batch["input_ids"].shape) == (1, 20)
    assert batch["document_ids"].tolist() == [[0] * 10 + [1] * 10]
    assert batch["loss_mask"][0, 7:10].tolist() == [0, 0, 0]
    assert batch["loss_mask"][0, 17:20].tolist() == [0, 0, 0]
    assert batch["position_ids"].tolist() == [list(range(10)) + list(range(20, 30))]
    assert torch.equal(batch["hidden_states"][0, :10], samples[0].aux_hidden_states.to(torch.bfloat16))
    assert torch.equal(batch["verifier_last_hidden_states"][0, 10:], samples[1].final_hidden_states.to(torch.bfloat16))


class _FakeDSpark(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, **batch):
        loss = self.scale.square()
        device = batch["input_ids"].device
        metrics = {
            "full_acc_sum": torch.tensor(3.0, device=device),
            "full_acc_total": torch.tensor(5.0, device=device),
            "accept_rate_sum": torch.tensor(7.0, device=device),
            "accept_rate_total": torch.tensor(11.0, device=device),
            "accept_len_sum": torch.tensor(13.0, device=device),
            "accept_len_total": torch.tensor(17.0, device=device),
            "confidence_loss_sum": torch.tensor(19.0, device=device),
            "confidence_loss_total": torch.tensor(23.0, device=device),
        }
        return None, loss, metrics


@pytest.mark.unit
def test_dspark_loss_adapts_speculators_metrics():
    batch = collate_dspark_samples([_sample("a")], "cpu", block_size=3)
    model = _FakeDSpark()

    loss, metrics = compute_dspark_loss(model, batch, {})
    loss.backward()

    assert loss.item() == 4.0
    assert metrics["loss_sum"].item() == 20.0
    assert metrics["token_count"].item() == 5.0
    assert metrics["top1_correct"].item() == 3.0
    assert metrics["accept_rate_sum"].item() == 7.0
    assert metrics["accept_rate_total"].item() == 11.0
    assert metrics["accept_len_sum"].item() == 13.0
    assert metrics["accept_len_total"].item() == 17.0
    assert metrics["confidence_loss_sum"].item() == 19.0
    assert metrics["confidence_loss_total"].item() == 23.0
    assert model.scale.grad.item() == 4.0


@pytest.mark.unit
def test_dspark_syncs_restricted_draft_and_verifier_heads():
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 3, bias=False)
    model.verifier_lm_head = torch.nn.Linear(2, 3, bias=False)
    target = torch.arange(10, dtype=torch.float32).reshape(5, 2)

    sync_dspark_lm_heads(model, target, torch.tensor([0, 2, 4]))

    expected = target[[0, 2, 4]]
    assert torch.equal(model.lm_head.weight, expected)
    assert torch.equal(model.verifier_lm_head.weight, expected)


class _TrainerKwargsModel(torch.nn.Module):
    def __init__(self, train_kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.train_kwargs = train_kwargs
        self.received = None

    def get_trainer_kwargs(self, **kwargs):
        self.received = kwargs
        return dict(self.train_kwargs, max_anchors=kwargs["max_anchors"]), {}


def _trainer_args():
    return type(
        "Args",
        (),
        {
            "draft_dspark_loss_fn": '{"ce": 0.1, "tv": 0.9}',
            "draft_dspark_decay_gamma": 4.0,
            "draft_dspark_max_anchors": 64,
            "draft_dspark_confidence_head_alpha": 1.0,
            "draft_dspark_per_position_loss_weight": "fixed-exp-decay",
            "draft_dspark_dpace_alpha": 0.5,
        },
    )()


@pytest.mark.unit
def test_dspark_npu_requests_eager_speculators_losses_when_supported(monkeypatch):
    eager_tv = lambda logits, targets: targets  # noqa: E731
    train_kwargs = {"loss_config": {"tv": (eager_tv, 0.9)}, "tv_loss_fn": eager_tv}
    model = _TrainerKwargsModel(train_kwargs)
    monkeypatch.setattr(dspark, "_model_device_type", lambda model: "npu")

    kwargs = dspark_trainer_kwargs(model, _trainer_args())

    assert kwargs["loss_config"]["tv"] == (eager_tv, 0.9)
    assert kwargs["tv_loss_fn"] is eager_tv
    assert model.received["loss_implementation"] == "eager"


@pytest.mark.unit
def test_dspark_pinned_speculators_may_omit_tv_loss_fn(monkeypatch):
    eager_tv = lambda logits, targets: targets  # noqa: E731
    model = _TrainerKwargsModel({"loss_config": {"tv": (eager_tv, 0.9)}})
    monkeypatch.setattr(dspark, "_model_device_type", lambda model: "npu")

    kwargs = dspark_trainer_kwargs(model, _trainer_args())

    assert kwargs["loss_config"]["tv"] == (eager_tv, 0.9)
    assert "tv_loss_fn" not in kwargs
    assert model.received["loss_implementation"] == "eager"


@pytest.mark.unit
def test_dspark_cuda_keeps_speculators_resolved_losses(monkeypatch):
    fused_tv = lambda logits, targets: logits  # noqa: E731
    model = _TrainerKwargsModel({"loss_config": {"tv": (fused_tv, 0.9)}, "tv_loss_fn": fused_tv})
    monkeypatch.setattr(dspark, "_model_device_type", lambda model: "cuda")

    kwargs = dspark_trainer_kwargs(model, _trainer_args())

    assert kwargs["loss_config"]["tv"] == (fused_tv, 0.9)
    assert kwargs["tv_loss_fn"] is fused_tv
    assert model.received["loss_implementation"] == "fused"
