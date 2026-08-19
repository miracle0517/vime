from argparse import Namespace

import pytest

from vime.backends.speculative_training.draft_group import ExternalDraftTrainGroup


def _args(algorithm: str = "dspark") -> Namespace:
    return Namespace(draft_algorithm=algorithm)


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _Actor:
    def __init__(self):
        self.calls = []
        self.get_external_draft_start_rollout = _RemoteMethod(self._create)
        self.collect_external_draft_features = _RemoteMethod(self._collect)
        self.train_external_draft = _RemoteMethod(self._train)
        self.prepare_external_draft_publish_snapshot = _RemoteMethod(self._snapshot)
        self.save_external_draft = _RemoteMethod(self._save)
        self.export_external_draft = _RemoteMethod(self._export)

    def _create(self):
        self.calls.append(("create",))
        return 0

    def _collect(self, feature_refs, target_head, target_version):
        self.calls.append(("collect", feature_refs, target_head, target_version))
        return {"accepted": 2, "received": 2, "queued": 2, "rejected_version_mismatch": 0}

    def _train(self, rollout_id):
        self.calls.append(("train", rollout_id))
        return {"trained": 1, "draft_version": 1, "target_weight_version": "7", "rollout_id": rollout_id}

    def _snapshot(self):
        self.calls.append(("snapshot",))
        return {
            "snapshot_ref": {"named_tensors": [("weight", 1)]},
            "draft_version": "1",
            "trained_against_target_version": "7",
        }

    def _save(self, rollout_id):
        self.calls.append(("save", rollout_id))
        return f"draft-{rollout_id}.pt"

    def _export(self, rollout_id):
        self.calls.append(("export", rollout_id))
        return {
            "complete": True,
            "path": f"draft-hf-{rollout_id}",
            "weight_files": ["model.safetensors"],
            "weight_bytes": 1024,
        }


@pytest.mark.unit
def test_draft_group_delegates_to_actor_rank_zero(monkeypatch):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module.ray, "put", lambda value: ("object-ref", value))
    monkeypatch.setattr(module.ray, "ObjectRef", type("ObjectRef", (), {}))
    rank_zero = _Actor()
    unused_rank = _Actor()
    actor_group = Namespace(_actor_handlers=[rank_zero, unused_rank])
    group = ExternalDraftTrainGroup(_args(), actor_group)

    assert group.create() == [0]
    result = group.collect_actor_results(
        [
            {
                "target_weight_version": "7",
                "draft_features_ref": "features",
                "draft_target_lm_head_ref": "head",
            }
        ]
    )
    assert result["placement"] == "actor_rank0"
    assert result["accepted"] == 2
    assert ("collect", ["features"], "head", "7") in rank_zero.calls

    assert group.train_draft(3)["trained"] == 1
    snapshot_ref, version = group.prepare_publish_snapshot()
    assert snapshot_ref[0] == "object-ref"
    assert version == "1"
    assert group.save_draft(3) == ["draft-3.pt"]
    assert group.export_draft(4) == {
        "complete": True,
        "path": "draft-hf-4",
        "weight_files": ["model.safetensors"],
        "weight_bytes": 1024,
    }
    assert {call[0] for call in rank_zero.calls} == {"create", "collect", "train", "snapshot", "save", "export"}
    assert unused_rank.calls == []


@pytest.mark.unit
def test_dspark_collect_does_not_mutate_a_frozen_publish_envelope(monkeypatch):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module.ray, "put", lambda value: ("object-ref", value))
    monkeypatch.setattr(module.ray, "ObjectRef", type("ObjectRef", (), {}))
    rank_zero = _Actor()
    group = ExternalDraftTrainGroup(_args(), Namespace(_actor_handlers=[rank_zero]))
    assert group.train_draft(1)["trained"] == 1

    group.collect_actor_results(
        [
            {
                "target_weight_version": "8",
                "draft_features_ref": None,
                "draft_target_lm_head_ref": "head-8",
            }
        ]
    )

    snapshot_ref, version = group.prepare_publish_snapshot()
    assert snapshot_ref[0] == "object-ref"
    assert version == "1"


@pytest.mark.unit
def test_dspark_failed_train_preserves_last_successful_publish_candidate(monkeypatch):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module.ray, "put", lambda value: ("object-ref", value))
    monkeypatch.setattr(module.ray, "ObjectRef", type("ObjectRef", (), {}))
    successful = {"trained": 1, "draft_version": 1, "target_weight_version": "7"}
    failed = {"trained": 0, "reason": "no_valid_optimizer_step"}
    results = iter((successful, failed))
    rank_zero = _Actor()
    rank_zero.train_external_draft = _RemoteMethod(lambda rollout_id: next(results))
    group = ExternalDraftTrainGroup(_args(), Namespace(_actor_handlers=[rank_zero]))

    assert group.train_draft(1) is successful
    assert group.train_draft(2) is failed
    assert group.last_train_result is successful
    snapshot_ref, version = group.prepare_publish_snapshot()
    assert snapshot_ref[0] == "object-ref"
    assert version == "1"


@pytest.mark.unit
@pytest.mark.parametrize("field,value", [("draft_version", "2"), ("trained_against_target_version", "8")])
def test_dspark_draft_group_rejects_snapshot_metadata_drift(monkeypatch, field, value):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda result: result)
    rank_zero = _Actor()
    original_snapshot = rank_zero._snapshot

    def mismatched_snapshot():
        result = original_snapshot()
        result[field] = value
        return result

    rank_zero.prepare_external_draft_publish_snapshot = _RemoteMethod(mismatched_snapshot)
    group = ExternalDraftTrainGroup(_args(), Namespace(_actor_handlers=[rank_zero]))
    group.train_draft(1)

    with pytest.raises(RuntimeError, match="metadata does not match"):
        group.prepare_publish_snapshot()


@pytest.mark.unit
def test_eagle3_uses_direct_snapshot_contract(monkeypatch):
    import vime.backends.speculative_training.draft_group as module

    object_ref_type = type("ObjectRef", (), {})
    direct_snapshot_ref = object_ref_type()
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(
        module.ray,
        "put",
        lambda value: pytest.fail("EAGLE3 direct snapshot must not be re-wrapped"),
    )
    monkeypatch.setattr(module.ray, "ObjectRef", object_ref_type)
    rank_zero = _Actor()
    rank_zero.prepare_external_draft_publish_snapshot = _RemoteMethod(lambda: direct_snapshot_ref)
    group = ExternalDraftTrainGroup(_args("eagle3"), Namespace(_actor_handlers=[rank_zero]))

    assert group.train_draft(1)["trained"] == 1
    snapshot_ref, version = group.prepare_publish_snapshot()
    assert snapshot_ref is direct_snapshot_ref
    assert version == "1"


@pytest.mark.unit
def test_eagle3_failed_train_overwrites_last_result_and_blocks_old_candidate(monkeypatch):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    successful = {"trained": 1, "draft_version": 1, "target_weight_version": "7"}
    failed = {"trained": 0, "reason": "no_valid_optimizer_step"}
    results = iter((successful, failed))
    rank_zero = _Actor()
    rank_zero.train_external_draft = _RemoteMethod(lambda rollout_id: next(results))
    group = ExternalDraftTrainGroup(_args("eagle3"), Namespace(_actor_handlers=[rank_zero]))

    assert group.train_draft(1) is successful
    assert group.train_draft(2) is failed
    assert group.last_train_result is failed
    assert group.prepare_publish_snapshot() is None


@pytest.mark.unit
def test_draft_group_keeps_legacy_disabled_save_return_contract(monkeypatch):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    rank_zero = _Actor()
    rank_zero.save_external_draft = _RemoteMethod(lambda rollout_id: None)
    actor_group = Namespace(_actor_handlers=[rank_zero])

    assert ExternalDraftTrainGroup(_args(), actor_group).save_draft(3) == [None]


@pytest.mark.unit
def test_draft_group_rejects_a_missing_requested_export(monkeypatch):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    rank_zero = _Actor()
    rank_zero.export_external_draft = _RemoteMethod(lambda rollout_id: None)
    actor_group = Namespace(_actor_handlers=[rank_zero])

    with pytest.raises(RuntimeError, match="did not return a complete artifact"):
        ExternalDraftTrainGroup(_args(), actor_group).export_draft(3)


@pytest.mark.unit
def test_export_draft_does_not_write_a_training_checkpoint(monkeypatch):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    rank_zero = _Actor()
    group = ExternalDraftTrainGroup(_args(), Namespace(_actor_handlers=[rank_zero]))

    assert group.export_draft(5)["path"] == "draft-hf-5"
    assert rank_zero.calls == [("export", 5)]
