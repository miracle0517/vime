#!/usr/bin/env python3
"""Analyze compact VIME MoE ALLTOALL probe records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MARKER = "VIME_MOE_PROBE_JSON "


def _same_reordering_stats(left: dict, right: dict) -> bool:
    keys = ("rows", "unique", "duplicate_rows", "max_duplicate_group")
    if any(left.get(key) != right.get(key) for key in keys):
        return False
    return all(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) <= 1e-5
        for key in ("norm_min", "norm_max", "norm_mean")
    )


def _count_mismatch_summary(expert_assignment: dict) -> str:
    actual = expert_assignment.get("actual_counts", [])
    expected = expert_assignment.get("expected_counts", [])
    mismatch_indices = [
        index
        for index, (actual_count, expected_count) in enumerate(zip(actual, expected))
        if actual_count != expected_count
    ]
    cumulative_as_counts = [
        expected[index] - (expected[index - 1] if index else 0) for index in range(len(expected))
    ]
    relation = "different_values"
    if len(actual) != len(expected):
        relation = "different_lengths"
    elif actual == cumulative_as_counts:
        relation = "expected_is_cumulative"
    elif sorted(actual) == sorted(expected):
        relation = "expert_order_permutation"
    return (
        f"relation={relation} actual_sum={sum(actual)} expected_sum={sum(expected)} "
        f"mismatch_indices={mismatch_indices} actual_counts={actual} expected_counts={expected}"
    )


def _load_reports(paths: list[Path]) -> tuple[list[dict], list[str]]:
    reports = []
    parse_errors = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                marker_index = line.find(MARKER)
                if marker_index < 0:
                    continue
                raw = line[marker_index + len(MARKER) :].strip()
                try:
                    report = json.loads(raw)
                except json.JSONDecodeError as error:
                    parse_errors.append(f"{path}:{line_number}: {error}")
                    continue
                report["_source"] = f"{path}:{line_number}"
                reports.append(report)
    return reports, parse_errors


def _analyze_report(report: dict) -> tuple[list[str], list[str]]:
    rank = report.get("ep_rank")
    source = report["_source"]
    prefix = f"rank={rank} source={source}"
    errors = []
    warnings = []

    mapping = report.get("mapping", {})
    if mapping.get("mapping_probe_error"):
        errors.append(f"{prefix}: permutation mapping probe failed")
    elif mapping.get("mapping_mismatch_count") != 0:
        errors.append(f"{prefix}: local permutation mapping mismatch")

    split = report.get("split", {})
    if split.get("split_probe_error"):
        errors.append(f"{prefix}: split metadata probe failed")
    else:
        if split.get("input_sum") != split.get("expected_input_sum"):
            errors.append(f"{prefix}: input_splits sum mismatch")
        if split.get("output_sum") != split.get("tokens_per_expert_sum"):
            errors.append(f"{prefix}: output_splits sum mismatch")
        if split.get("transpose_mismatch_count") != 0:
            errors.append(f"{prefix}: input/output split transpose mismatch")
        if split.get("negative_split_count") != 0:
            errors.append(f"{prefix}: negative split count")

    payload = report.get("payload", {})
    if payload.get("payload_probe_error"):
        errors.append(f"{prefix}: ALLTOALL payload probe failed")
    elif payload.get("mismatch_count") != 0:
        release_note = (
            " after the send storage was released" if payload.get("send_released_before_postprocess") else ""
        )
        errors.append(
            f"{prefix}: first ALLTOALL payload mismatch from src ranks "
            f"{payload.get('mismatch_src_ranks', [])}{release_note}"
        )
    if payload.get("send_recv_same_ptr"):
        warnings.append(f"{prefix}: send and receive buffers reuse the same data pointer")

    expert_assignment = report.get("expert_assignment", {})
    if not expert_assignment:
        errors.append(f"{prefix}: ALLTOALL expert assignment probe record is missing")
    elif expert_assignment.get("probe_error"):
        errors.append(f"{prefix}: ALLTOALL expert assignment probe failed")
    elif expert_assignment.get("invalid_expert_id_count") != 0:
        errors.append(f"{prefix}: ALLTOALL returned invalid local expert IDs")
    if expert_assignment.get("expert_count_mismatch_count", 0) not in (None, 0):
        errors.append(
            f"{prefix}: ALLTOALL local expert IDs disagree with grouped matmul expert counts; "
            f"{_count_mismatch_summary(expert_assignment)}"
        )

    stages = report.get("stages", {})
    roundtrip = stages.get("second_permute_roundtrip", {})
    if roundtrip.get("mismatch_count", 0) != 0:
        errors.append(f"{prefix}: second expert-local permutation roundtrip mismatch")

    alltoall1_output = stages.get("alltoall1_output")
    gmm_input = stages.get("gmm_input")
    if alltoall1_output and gmm_input and not _same_reordering_stats(alltoall1_output, gmm_input):
        errors.append(f"{prefix}: rows changed during second expert-local permutation")

    gmm_output = stages.get("gmm_output")
    after_second_unpermute = stages.get("after_second_unpermute")
    if gmm_output and after_second_unpermute and not _same_reordering_stats(gmm_output, after_second_unpermute):
        errors.append(f"{prefix}: rows changed during second expert-local unpermute")

    moe_input = stages.get("moe_input", {})
    moe_output = stages.get("moe_output", {})
    if moe_input and moe_output:
        if moe_output.get("duplicate_rows", 0) > moe_input.get("duplicate_rows", 0):
            warnings.append(
                f"{prefix}: duplicate rows increased across MoE "
                f"{moe_input.get('duplicate_rows')} -> {moe_output.get('duplicate_rows')}"
            )
        if moe_output.get("max_duplicate_group", 0) > moe_input.get("max_duplicate_group", 0):
            warnings.append(
                f"{prefix}: maximum duplicate group increased across MoE "
                f"{moe_input.get('max_duplicate_group')} -> {moe_output.get('max_duplicate_group')}"
            )

    unpermute = report.get("unpermute", {})
    if unpermute.get("mismatch_count") != 0:
        errors.append(f"{prefix}: final token unpermute mismatch")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    reports, parse_errors = _load_reports(args.logs)
    if not reports:
        print(f"[ERROR] no '{MARKER.strip()}' records found")
        for error in parse_errors:
            print(f"[ERROR] {error}")
        return 2

    errors = [f"JSON parse failure: {error}" for error in parse_errors]
    warnings = []
    for report in reports:
        report_errors, report_warnings = _analyze_report(report)
        errors.extend(report_errors)
        warnings.extend(report_warnings)

    print(f"probe records: {len(reports)}")
    print(f"errors: {len(errors)}, warnings: {len(warnings)}")
    for error in errors:
        print(f"[ERROR] {error}")
    for warning in warnings:
        print(f"[WARNING] {warning}")

    if errors:
        print("first suspected boundary: see the first error above")
        return 1
    if warnings:
        print("communication and permutation checks passed; inspect repetition warnings around GMM/combine")
    else:
        print("all checks passed for the probed ALLTOALL MoE invocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
