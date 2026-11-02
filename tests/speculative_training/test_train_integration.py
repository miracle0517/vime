"""Exercise driver orchestration without starting Ray or accelerator workers."""

import ast
import logging
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vime.backends.speculative_training.config import should_run_draft_interval


def _train_driver(events, *, external_draft=True, publication_succeeds=True):
    def record(name, result=None):
        def call(*args, **kwargs):
            events.append((name, args, kwargs))
            return result

        return call

    def remote(name, result=None):
        return SimpleNamespace(remote=record(name, result))

    actor_results = [{"target_weight_version": "0", "draft_features_ref": "features"}]
    actor = SimpleNamespace(
        create=record("actor.create"),
        async_train=record("actor.train", actor_results),
        update_weights=record("actor.publish", [publication_succeeds]),
        set_external_draft_weights=record("actor.stage_draft"),
        save_model=record("actor.save"),
        clear_memory=record("actor.clear"),
    )
    rollout = SimpleNamespace(
        generate=remote("rollout.generate", "rollout-data"),
        offload=remote("rollout.offload"),
        onload_weights=remote("rollout.onload_weights"),
        onload_kv=remote("rollout.onload_kv"),
        dispose=remote("rollout.dispose"),
    )
    draft = SimpleNamespace(
        collect_actor_results=record("draft.collect", {"accepted": 1}),
        train_draft=record("draft.train", {"trained": 1, "draft_version": 1}),
        prepare_publish_snapshot=record("draft.snapshot", ("snapshot", "1")),
        save_draft=record("draft.save", ["checkpoint"]),
        mark_published=record("draft.mark"),
        release=record("draft.release"),
    )
    namespace = {
        "os": os,
        "logger": logging.getLogger(__name__),
        "ray": SimpleNamespace(get=lambda value: value),
        "configure_logger": lambda: None,
        "init_tracking": lambda args: None,
        "finish_tracking": lambda args: None,
        "create_placement_groups": lambda args: {"rollout": None},
        "create_rollout_manager": lambda *args: (rollout, 1),
        "create_training_models": lambda *args: (actor, None),
        "create_draft_model": lambda *args: draft if external_draft else None,
        "should_run_draft_interval": should_run_draft_interval,
        "should_run_periodic_action": lambda rollout_id, interval, *args: bool(
            interval and (rollout_id + 1) % interval == 0
        ),
        "compute_rollout_step": lambda args, rollout_id: rollout_id,
        "logging_utils": SimpleNamespace(log=lambda *args, **kwargs: None),
    }
    # Isolate the driver functions from imports requiring a full training image.
    source = Path(__file__).parents[2] / "train.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["train"]


def _args(**overrides):
    values = dict(
        release_train=False,
        draft_save_hf="export",
        draft_save_interval=None,
        draft_train_interval=1,
        draft_publish_interval=1,
        offload_rollout=True,
        offload_train=False,
        check_weight_update_equal=False,
        num_rollout=1,
        start_rollout_id=0,
        eval_interval=None,
        save_interval=1,
        use_critic=False,
        rollout_global_dataset=False,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.unit
def test_driver_trains_saves_and_acknowledges_draft_after_weight_publication():
    events = []
    _train_driver(events)(_args())

    names = [event[0] for event in events]
    assert names.index("actor.train") < names.index("draft.collect") < names.index("draft.train")
    assert names.index("draft.train") < names.index("draft.save") < names.index("actor.stage_draft")
    assert names.index("actor.stage_draft") < names.index("draft.mark")
    assert names[names.index("draft.mark") - 1] == "actor.publish"
    assert next(event for event in events if event[0] == "draft.save")[2] == {"export_hf": True}


@pytest.mark.unit
def test_driver_does_not_acknowledge_failed_draft_publication():
    events = []

    with pytest.raises(RuntimeError, match="publication did not complete"):
        _train_driver(events, publication_succeeds=False)(_args())

    assert "draft.mark" not in [event[0] for event in events]


@pytest.mark.unit
def test_driver_preserves_release_train_recreation_and_forced_save():
    events = []
    _train_driver(events, external_draft=False)(_args(release_train=True, num_rollout=2, save_interval=None))

    names = [event[0] for event in events]
    assert names.count("actor.create") == 2
    assert names.index("actor.create") < names.index("actor.train")
    assert "rollout.onload_weights" not in names
    saves = [event for event in events if event[0] == "actor.save"]
    assert len(saves) == 2
    assert all(event[2] == {"force_sync": True} for event in saves)
