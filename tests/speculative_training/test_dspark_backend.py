from __future__ import annotations

import pytest
import torch

from vime.backends.speculative_training.backends.dspark import (
    collate_dspark_samples,
    compute_dspark_loss,
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
        aux_hidden_states=torch.randn(rows, 6),
        final_hidden_states=torch.randn(rows, 2),
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
    batch = collate_dspark_samples([_sample("a"), _sample("b", 20)], "cpu", block_size=3)

    assert tuple(batch["input_ids"].shape) == (1, 20)
    assert batch["document_ids"].tolist() == [[0] * 10 + [1] * 10]
    assert batch["loss_mask"][0, 7:10].tolist() == [0, 0, 0]
    assert batch["loss_mask"][0, 17:20].tolist() == [0, 0, 0]
    assert batch["position_ids"].tolist() == [list(range(10)) + list(range(20, 30))]


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
            "accept_rate_sum": torch.tensor(4.0, device=device),
            "accept_rate_total": torch.tensor(5.0, device=device),
        }
        return None, loss, metrics


@pytest.mark.unit
def test_dspark_loss_adapts_speculators_metrics():
    batch = collate_dspark_samples([_sample("a")], "cpu", block_size=3)

    loss, metrics = compute_dspark_loss(_FakeDSpark(), batch, {})

    assert loss.item() == 4.0
    assert metrics["loss_sum"].item() == 20.0
    assert metrics["token_count"].item() == 5.0
    assert metrics["top1_correct"].item() == 3.0
    assert metrics["accept_rate_sum"].item() == 4.0


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
