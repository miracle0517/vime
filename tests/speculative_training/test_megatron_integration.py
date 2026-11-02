"""Exercise the Megatron integration without requiring a Megatron installation."""

import ast
import sys
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_function(relative_path, name, namespace):
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), relative_path, "exec"), namespace)
    return namespace[name]


@pytest.mark.unit
@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_actor_log_prob_forward_preserves_replay_and_collects_features(monkeypatch, collect, fail):
    calls = []

    class Collector:
        def __init__(self, args, model, **kwargs):
            calls.append(("create", kwargs))

        def pop_payloads(self):
            return ["features"]

        def close(self):
            calls.append("close")

    monkeypatch.setitem(
        sys.modules,
        "vime.backends.speculative_training.feature_collector",
        SimpleNamespace(DraftFeatureCollector=Collector),
    )

    def forward_only(*args, **kwargs):
        calls.append(("forward", kwargs))
        if fail:
            raise RuntimeError("forward failed")
        return {"log_probs": [1]}

    compute_log_prob = _load_function(
        "vime/backends/megatron_utils/actor.py",
        "compute_log_prob",
        {"timer": lambda name: nullcontext(), "forward_only": forward_only, "get_log_probs_and_entropy": object()},
    )
    actor = Namespace(
        args=Namespace(),
        model=[],
        weight_updater=Namespace(weight_version=7),
        _draft_feature_payloads=[],
        _draft_feature_collection_ran=False,
    )
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            compute_log_prob(actor, [], [1], collect_draft_features=collect, rollout_id=3)
    else:
        assert compute_log_prob(actor, [], [1], collect_draft_features=collect, rollout_id=3) == {"log_probs": [1]}

    forwards = [entry[1] for entry in calls if isinstance(entry, tuple) and entry[0] == "forward"]
    assert len(forwards) == 1
    assert forwards[0]["use_rollout_top_p_replay"] is True
    assert isinstance(forwards[0]["feature_collector"], Collector) is collect
    assert actor._draft_feature_collection_ran is collect
    assert actor._draft_feature_payloads == (["features"] if collect else [])
    assert ("close" in calls) is collect
    if collect:
        assert calls[0] == ("create", {"rollout_id": 3, "target_weight_version": "7"})


@pytest.mark.unit
@pytest.mark.parametrize("collect", [False, True])
def test_model_forward_accepts_optional_feature_collector(collect):
    events = []
    iterator = Namespace(reset=lambda: None)
    if collect:
        iterator.micro_batch_indices = [[4]]
        iterator.offset = 0
    batch = {
        "unconcat_tokens": [],
        "tokens": "tokens",
        "packed_seq_params": None,
        "total_lengths": [],
        "response_lengths": [],
        "full_loss_masks": [],
        "multimodal_train_inputs": None,
    }

    class Model:
        def eval(self):
            pass

        def train(self):
            pass

        def __call__(self, **kwargs):
            events.append("forward")
            return "logits"

    def schedule(**kwargs):
        kwargs["forward_step_func"](kwargs["data_iterator"][0], kwargs["model"][0])
        return []

    forward_only = _load_function(
        "vime/backends/megatron_utils/model.py",
        "forward_only",
        {
            "get_model_config": lambda model: Namespace(),
            "get_batch": lambda *args: batch,
            "get_forward_backward_func": lambda: schedule,
            "tqdm": lambda **kwargs: Namespace(close=lambda: None),
            "_disable_tqdm_for_non_main_rank": lambda: True,
            "_wrap_forward_step_with_microbatch_pbar": lambda function, progress: function,
            "mpu": Namespace(is_pipeline_last_stage=lambda: False),
            "partial": lambda function, **kwargs: function,
        },
    )
    args = Namespace(
        data_pad_size_multiplier=1,
        allgather_cp=False,
        custom_megatron_before_log_prob_hook_path=None,
        seq_length=2,
        micro_batch_size=1,
        use_rollout_entropy=False,
    )
    collector = (
        Namespace(
            begin_microbatch=lambda batch, indices: events.append(("begin", indices)),
            end_microbatch=lambda: events.append("end"),
            abort_microbatch=lambda: events.append("abort"),
        )
        if collect
        else None
    )
    assert forward_only(lambda: None, args, [Model()], [iterator], [1], feature_collector=collector) == {}
    assert events == ([("begin", [4]), "forward", "end"] if collect else ["forward"])
