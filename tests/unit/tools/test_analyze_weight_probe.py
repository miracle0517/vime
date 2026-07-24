import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[3] / "tools" / "analyze_weight_probe.py"
SPEC = importlib.util.spec_from_file_location("analyze_weight_probe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _probe(*, sample_hash=123, npu_format=29, shape=None):
    value = float(sample_hash)
    return {
        "shape": shape or [2, 2],
        "stride": [2, 1],
        "storage_offset": 0,
        "dtype": "torch.bfloat16",
        "device": "npu:0",
        "npu_format": npu_format,
        "num_bytes": 8,
        "sample_hash": sample_hash,
        "head": [1, 2],
        "tail": [3, 4],
        "value_stats": {
            "numel": 4,
            "finite_count": 4,
            "nan_count": 0,
            "posinf_count": 0,
            "neginf_count": 0,
            "zero_count": 0,
            "min": value,
            "max": value,
            "sum": value * 4,
            "mean": value,
            "abs_mean": abs(value),
            "l2_norm": abs(value) * 2,
        },
        "value_samples": [value, value],
    }


def _actor_line(rank, uuid, *, sample_hash=123):
    context = {"pid": 100 + rank, "dist_rank": rank, "npu_uuid": uuid}
    return (
        "[VIME_WEIGHT_PROBE] stage=actor_export version=1 chunk=2 "
        f"rank={rank} context={context!r} "
        "name=model.layers.0.mlp.experts.0.gate_proj.weight "
        f"probe={_probe(sample_hash=sample_hash)!r}"
    )


def _stage_line(
    stage,
    uuid,
    *,
    sample_hash=123,
    npu_format=29,
    name="layers.0.mlp.experts.0.gate_proj.weight",
    shape=None,
):
    context = {"pid": 200, "dist_rank": 0, "npu_uuid": uuid}
    return (
        f"[VIME_WEIGHT_PROBE] stage={stage} context={context!r} "
        f"name={name} "
        f"probe={_probe(sample_hash=sample_hash, npu_format=npu_format, shape=shape)!r}"
    )


def _write_log(tmp_path, lines):
    path = tmp_path / "run.log"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.mark.unit
def test_analyze_matches_actor_ipc_finalize_and_layout(tmp_path):
    lines = []
    for rank, uuid in enumerate(("node-0", "node-1")):
        lines.append(_stage_line(probe.WAKE_STAGE, uuid))
        lines.append(_actor_line(rank, uuid))
        lines.append(_stage_line(probe.IPC_STAGE, uuid))
        lines.append(_stage_line(probe.BEFORE_PROCESS_STAGE, uuid))
        lines.append(_stage_line(probe.AFTER_PROCESS_STAGE, uuid))
        lines.append(_stage_line(probe.FINISH_STAGE, uuid))
        lines.append(
            _stage_line(
                probe.FINISH_STAGE,
                uuid,
                name="layers.0.mlp.experts.w13_weight",
            )
        )

    report = probe.analyze([_write_log(tmp_path, lines)])

    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.checks["actor_cross_rank"].passed == 1
    assert report.checks["actor_to_ipc"].passed == 2
    assert report.checks["ipc_to_before_process"].passed == 2
    assert report.checks["before_to_after_process"].passed == 2
    assert report.checks["after_process_to_finish"].passed == 2
    assert report.checks["ipc_to_finish"].passed == 2
    assert report.checks["wake_to_finish_layout"].passed == 2


@pytest.mark.unit
def test_analyze_reports_ipc_value_mismatch(tmp_path):
    lines = [
        _stage_line(probe.WAKE_STAGE, "node-0"),
        _actor_line(0, "node-0"),
        _stage_line(probe.IPC_STAGE, "node-0", sample_hash=999),
        _stage_line(probe.BEFORE_PROCESS_STAGE, "node-0", sample_hash=999),
        _stage_line(probe.AFTER_PROCESS_STAGE, "node-0", sample_hash=999),
        _stage_line(probe.FINISH_STAGE, "node-0", sample_hash=999),
    ]

    report = probe.analyze([_write_log(tmp_path, lines)])

    assert report.error_count == 1
    assert report.checks["actor_to_ipc"].failed == 1
    assert any(issue.category == "IPC_TRANSFER" for issue in report.issues)


@pytest.mark.unit
def test_analyze_reports_expert_weight_loader_mismatch(tmp_path):
    lines = [
        _stage_line(probe.WAKE_STAGE, "node-0"),
        _actor_line(0, "node-0"),
        _stage_line(probe.IPC_STAGE, "node-0"),
        _stage_line(probe.BEFORE_PROCESS_STAGE, "node-0", sample_hash=999),
        _stage_line(probe.AFTER_PROCESS_STAGE, "node-0", sample_hash=999),
        _stage_line(probe.FINISH_STAGE, "node-0", sample_hash=999),
    ]

    report = probe.analyze([_write_log(tmp_path, lines)])

    assert report.checks["ipc_to_before_process"].failed == 1
    assert report.checks["before_to_after_process"].passed == 1
    assert report.checks["after_process_to_finish"].passed == 1
    loader_issue = next(issue for issue in report.issues if issue.category == "EXPERT_WEIGHT_LOADER")
    assert "'mean': {'expected': 123.0, 'actual': 999.0}" in loader_issue.message


@pytest.mark.unit
def test_analyze_accepts_physical_moe_transpose_and_compares_hf_views(tmp_path):
    lines = [
        _stage_line(probe.WAKE_STAGE, "node-0"),
        _actor_line(0, "node-0"),
        _stage_line(probe.IPC_STAGE, "node-0"),
        _stage_line(probe.BEFORE_PROCESS_STAGE, "node-0"),
        _stage_line(
            probe.BEFORE_PROCESS_STAGE,
            "node-0",
            name="layers.0.mlp.experts.w13_weight",
            shape=[32, 1536, 2048],
            sample_hash=111,
        ),
        _stage_line(probe.AFTER_PROCESS_STAGE, "node-0"),
        _stage_line(
            probe.AFTER_PROCESS_STAGE,
            "node-0",
            name="layers.0.mlp.experts.w13_weight",
            shape=[32, 2048, 1536],
            sample_hash=222,
        ),
        _stage_line(probe.FINISH_STAGE, "node-0"),
    ]

    report = probe.analyze([_write_log(tmp_path, lines)])

    assert not any(issue.category == "MOE_PROCESS" for issue in report.issues)
    assert report.checks["before_to_after_process"].compared == 1
    assert report.checks["before_to_after_process"].passed == 1


@pytest.mark.unit
def test_analyze_reports_finalize_and_layout_mismatch(tmp_path):
    lines = [
        _stage_line(probe.WAKE_STAGE, "node-0", npu_format=29),
        _actor_line(0, "node-0"),
        _stage_line(probe.IPC_STAGE, "node-0"),
        _stage_line(probe.BEFORE_PROCESS_STAGE, "node-0"),
        _stage_line(probe.AFTER_PROCESS_STAGE, "node-0"),
        _stage_line(probe.FINISH_STAGE, "node-0", sample_hash=456, npu_format=0),
    ]

    report = probe.analyze([_write_log(tmp_path, lines)])

    assert report.checks["ipc_to_before_process"].passed == 1
    assert report.checks["before_to_after_process"].passed == 1
    assert report.checks["after_process_to_finish"].failed == 1
    assert report.checks["ipc_to_finish"].failed == 1
    assert report.checks["wake_to_finish_layout"].failed == 1
    assert {issue.category for issue in report.issues} >= {
        "LAYERWISE_FINALIZE",
        "MOE_FINALIZE",
        "MOE_LAYOUT",
    }


@pytest.mark.unit
def test_analyze_reports_metadata_and_missing_stage(tmp_path):
    lines = [
        "[VIME_WEIGHT_PROBE] stage=actor_export_metadata_mismatch "
        "version=1 chunk=2 rank=1 rank0_count=2 actual_count=2 differences=[]",
        _actor_line(0, "node-0"),
    ]

    report = probe.analyze([_write_log(tmp_path, lines)])

    categories = {issue.category for issue in report.issues}
    assert "ACTOR_METADATA" in categories
    assert "MISSING_STAGE" in categories
    assert report.error_count >= 3
