#!/usr/bin/env python3
"""Compare VIME colocated weight-update probe logs.

The probe is emitted when ``--check-weight-update-equal`` is enabled. This
script correlates Actor exports, vLLM IPC receives, and vLLM runtime MoE
weights without requiring users to manually compare long Ray logs.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MARKER = "[VIME_WEIGHT_PROBE]"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ACTOR_RE = re.compile(
    r"stage=actor_export version=(\d+) chunk=(\d+) rank=(\d+) " r"context=(\{.*\}) name=(\S+) probe=(\{.*\})$"
)
GENERIC_RE = re.compile(r"stage=(\S+) context=(\{.*\}) name=(\S+) probe=(\{.*\})$")
METADATA_MISMATCH_RE = re.compile(r"stage=actor_export_metadata_mismatch version=(\d+) chunk=(\d+) rank=(\S+)")

LOGICAL_KEYS = (
    "shape",
    "dtype",
    "num_bytes",
    "sample_hash",
    "head",
    "tail",
    "value_stats",
    "value_samples",
)
LAYOUT_KEYS = ("shape", "stride", "storage_offset", "dtype", "npu_format")
IPC_STAGE = "ipc_receive_before_load"
WAKE_STAGE = "after_weight_wake_up"
LOADER_INPUT_STAGE = "expert_loader_input"
LOADER_OUTPUT_STAGE = "expert_loader_output"
FINISH_STAGE = "after_finish_weight_update"


@dataclass(frozen=True)
class ProbeRecord:
    source: str
    line_number: int
    stage: str
    name: str
    context: dict[str, Any]
    probe: dict[str, Any]
    version: int | None = None
    chunk: int | None = None
    rank: int | None = None

    @property
    def npu_uuid(self) -> str | None:
        value = self.context.get("npu_uuid")
        return str(value) if value is not None else None

    @property
    def location(self) -> str:
        return f"{self.source}:{self.line_number}"


@dataclass
class Issue:
    severity: str
    category: str
    message: str
    locations: list[str] = field(default_factory=list)


@dataclass
class CheckStats:
    compared: int = 0
    passed: int = 0
    failed: int = 0


@dataclass
class AnalysisReport:
    files: list[str]
    records: int
    stage_counts: dict[str, int]
    checks: dict[str, CheckStats]
    issues: list[Issue]
    parse_errors: list[str]
    value_chains: list[dict[str, Any]]

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "ERROR" for issue in self.issues) + len(self.parse_errors)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "WARN" for issue in self.issues)

    def to_json(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "records": self.records,
            "stage_counts": self.stage_counts,
            "checks": {name: asdict(stats) for name, stats in self.checks.items()},
            "issues": [asdict(issue) for issue in self.issues],
            "parse_errors": self.parse_errors,
            "value_chains": self.value_chains,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


def normalize_weight_name(name: str) -> str:
    """Normalize checkpoint and runtime names to the same Qwen layer path."""
    normalized = name
    for prefix in ("module.", "model."):
        while normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


def _literal_dict(text: str, *, location: str, field_name: str) -> dict[str, Any]:
    value = ast.literal_eval(text)
    if not isinstance(value, dict):
        raise ValueError(f"{location}: {field_name} is not a dict")
    return value


def parse_probe_logs(paths: Iterable[Path]) -> tuple[list[ProbeRecord], list[Issue], list[str]]:
    records: list[ProbeRecord] = []
    issues: list[Issue] = []
    parse_errors: list[str] = []

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = ANSI_RE.sub("", raw_line).strip()
                marker_index = line.find(MARKER)
                if marker_index < 0:
                    continue
                payload = line[marker_index + len(MARKER) :].strip()
                location = f"{path}:{line_number}"

                metadata_match = METADATA_MISMATCH_RE.search(payload)
                if metadata_match:
                    issues.append(
                        Issue(
                            severity="ERROR",
                            category="ACTOR_METADATA",
                            message=(
                                "Actor ranks exported different names/shapes/order: "
                                f"version={metadata_match.group(1)} "
                                f"chunk={metadata_match.group(2)} rank={metadata_match.group(3)}"
                            ),
                            locations=[location],
                        )
                    )
                    continue

                actor_match = ACTOR_RE.search(payload)
                generic_match = GENERIC_RE.search(payload) if actor_match is None else None
                try:
                    if actor_match:
                        records.append(
                            ProbeRecord(
                                source=str(path),
                                line_number=line_number,
                                stage="actor_export",
                                version=int(actor_match.group(1)),
                                chunk=int(actor_match.group(2)),
                                rank=int(actor_match.group(3)),
                                context=_literal_dict(actor_match.group(4), location=location, field_name="context"),
                                name=actor_match.group(5),
                                probe=_literal_dict(actor_match.group(6), location=location, field_name="probe"),
                            )
                        )
                    elif generic_match:
                        records.append(
                            ProbeRecord(
                                source=str(path),
                                line_number=line_number,
                                stage=generic_match.group(1),
                                context=_literal_dict(generic_match.group(2), location=location, field_name="context"),
                                name=generic_match.group(3),
                                probe=_literal_dict(generic_match.group(4), location=location, field_name="probe"),
                            )
                        )
                    elif "probe=" in payload:
                        parse_errors.append(f"{location}: unrecognized probe line: {payload[:300]}")
                    else:
                        issues.append(
                            Issue(
                                severity="ERROR",
                                category="PROBE_RUNTIME",
                                message=payload,
                                locations=[location],
                            )
                        )
                except (SyntaxError, ValueError) as exc:
                    parse_errors.append(f"{location}: {exc}")

    return records, issues, parse_errors


def _probe_difference(
    expected: ProbeRecord,
    actual: ProbeRecord,
    keys: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    return {
        key: {"expected": expected.probe.get(key), "actual": actual.probe.get(key)}
        for key in keys
        if expected.probe.get(key) != actual.probe.get(key)
    }


def _value_difference_summary(
    expected: ProbeRecord,
    actual: ProbeRecord,
) -> dict[str, Any]:
    display_keys = (
        "numel",
        "finite_count",
        "nan_count",
        "posinf_count",
        "neginf_count",
        "zero_count",
        "min",
        "max",
        "sum",
        "mean",
        "abs_mean",
        "l2_norm",
    )
    expected_stats = expected.probe.get("value_stats")
    actual_stats = actual.probe.get("value_stats")

    changed_stats = {}
    if isinstance(expected_stats, dict) and isinstance(actual_stats, dict):
        changed_stats = {
            key: {
                "expected": expected_stats.get(key),
                "actual": actual_stats.get(key),
            }
            for key in display_keys
            if expected_stats.get(key) != actual_stats.get(key)
        }

    expected_samples = expected.probe.get("value_samples")
    actual_samples = actual.probe.get("value_samples")
    changed_samples = []
    if isinstance(expected_samples, list) and isinstance(actual_samples, list):
        for index, (expected_value, actual_value) in enumerate(zip(expected_samples, actual_samples, strict=False)):
            if expected_value != actual_value:
                changed_samples.append(
                    {
                        "index": index,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
            if len(changed_samples) == 8:
                break

    return {
        "changed_stats": changed_stats,
        "changed_samples": changed_samples,
    }


def _append_comparison(
    *,
    report_issues: list[Issue],
    stats: CheckStats,
    category: str,
    label: str,
    expected: ProbeRecord,
    actual: ProbeRecord,
    keys: tuple[str, ...],
) -> None:
    stats.compared += 1
    differences = _probe_difference(expected, actual, keys)
    if not differences:
        stats.passed += 1
        return
    stats.failed += 1
    value_summary = _value_difference_summary(expected, actual)
    report_issues.append(
        Issue(
            severity="ERROR",
            category=category,
            message=(
                f"{label}: uuid={actual.npu_uuid!r} "
                f"name={normalize_weight_name(actual.name)!r} "
                f"value_summary={value_summary} differences={differences}"
            ),
            locations=[expected.location, actual.location],
        )
    )


def _group_by_uuid_and_name(records: Iterable[ProbeRecord]) -> dict[tuple[str, str], list[ProbeRecord]]:
    grouped: dict[tuple[str, str], list[ProbeRecord]] = defaultdict(list)
    for record in records:
        if record.npu_uuid is None:
            continue
        grouped[(record.npu_uuid, normalize_weight_name(record.name))].append(record)
    return grouped


def _compare_actor_ranks(
    records: list[ProbeRecord],
    issues: list[Issue],
    stats: CheckStats,
) -> None:
    grouped: dict[tuple[int | None, int | None, str], list[ProbeRecord]] = defaultdict(list)
    for record in records:
        if record.stage == "actor_export":
            grouped[(record.version, record.chunk, normalize_weight_name(record.name))].append(record)

    for (version, chunk, name), group in grouped.items():
        baseline = group[0]
        seen_ranks: set[int | None] = set()
        for record in group:
            if record.rank in seen_ranks:
                issues.append(
                    Issue(
                        severity="WARN",
                        category="ACTOR_EXPORT",
                        message=(
                            f"Duplicate Actor probe: version={version} chunk={chunk} "
                            f"rank={record.rank} name={name!r}"
                        ),
                        locations=[record.location],
                    )
                )
            seen_ranks.add(record.rank)
            if record is baseline:
                continue
            _append_comparison(
                report_issues=issues,
                stats=stats,
                category="ACTOR_VALUE",
                label=f"Actor ranks exported different logical values (version={version}, chunk={chunk})",
                expected=baseline,
                actual=record,
                keys=LOGICAL_KEYS,
            )


def _compare_paired_stages(
    *,
    expected_records: list[ProbeRecord],
    actual_records: list[ProbeRecord],
    issues: list[Issue],
    stats: CheckStats,
    category: str,
    label: str,
    keys: tuple[str, ...],
    require_all_expected: bool,
) -> None:
    expected_groups = _group_by_uuid_and_name(expected_records)
    actual_groups = _group_by_uuid_and_name(actual_records)
    if require_all_expected:
        keys_to_compare = set(expected_groups)
    else:
        # Runtime stages also contain fused w13_weight/w2_weight probes. They
        # deliberately have no matching HF checkpoint name and must not be
        # reported as missing.
        keys_to_compare = set(expected_groups) & set(actual_groups)

    for key in sorted(keys_to_compare):
        expected = expected_groups.get(key, [])
        actual = actual_groups.get(key, [])
        pair_count = min(len(expected), len(actual))
        for index in range(pair_count):
            _append_comparison(
                report_issues=issues,
                stats=stats,
                category=category,
                label=f"{label} occurrence={index}",
                expected=expected[index],
                actual=actual[index],
                keys=keys,
            )
        if len(expected) != len(actual):
            issues.append(
                Issue(
                    severity="ERROR",
                    category=f"{category}_MISSING",
                    message=(
                        f"{label}: uuid={key[0]!r} name={key[1]!r} "
                        f"expected_count={len(expected)} actual_count={len(actual)}"
                    ),
                    locations=[record.location for record in (expected + actual)[:8]],
                )
            )


def _build_value_chains(records: list[ProbeRecord]) -> list[dict[str, Any]]:
    stage_order = {
        "actor_export": 0,
        IPC_STAGE: 1,
        LOADER_INPUT_STAGE: 2,
        LOADER_OUTPUT_STAGE: 3,
        FINISH_STAGE: 4,
    }
    grouped: dict[tuple[str, str], list[ProbeRecord]] = defaultdict(list)
    for record in records:
        if record.stage not in stage_order or record.npu_uuid is None:
            continue
        if not isinstance(record.probe.get("value_stats"), dict):
            continue
        grouped[(record.npu_uuid, normalize_weight_name(record.name))].append(record)

    chains = []
    for (npu_uuid, name), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda record: (stage_order[record.stage], record.line_number))
        chains.append(
            {
                "npu_uuid": npu_uuid,
                "name": name,
                "stages": [
                    {
                        "stage": record.stage,
                        "location": record.location,
                        "value_stats": record.probe.get("value_stats"),
                        "value_samples": record.probe.get("value_samples"),
                        "sample_hash": record.probe.get("sample_hash"),
                    }
                    for record in ordered
                ],
            }
        )
    return chains


def analyze(paths: list[Path]) -> AnalysisReport:
    records, issues, parse_errors = parse_probe_logs(paths)
    checks = {
        "actor_cross_rank": CheckStats(),
        "actor_to_ipc": CheckStats(),
        "ipc_to_loader_input": CheckStats(),
        "loader_input_to_output": CheckStats(),
        "loader_output_to_finish": CheckStats(),
        "ipc_to_finish": CheckStats(),
        "wake_to_finish_layout": CheckStats(),
    }

    _compare_actor_ranks(records, issues, checks["actor_cross_rank"])

    actor_records = [record for record in records if record.stage == "actor_export"]
    ipc_records = [record for record in records if record.stage == IPC_STAGE]
    loader_input_records = [record for record in records if record.stage == LOADER_INPUT_STAGE]
    loader_output_records = [record for record in records if record.stage == LOADER_OUTPUT_STAGE]
    finish_records = [record for record in records if record.stage == FINISH_STAGE]
    wake_records = [record for record in records if record.stage == WAKE_STAGE]

    _compare_paired_stages(
        expected_records=actor_records,
        actual_records=ipc_records,
        issues=issues,
        stats=checks["actor_to_ipc"],
        category="IPC_TRANSFER",
        label="Actor export -> IPC receive",
        keys=LOGICAL_KEYS,
        require_all_expected=True,
    )
    _compare_paired_stages(
        expected_records=ipc_records,
        actual_records=loader_input_records,
        issues=issues,
        stats=checks["ipc_to_loader_input"],
        category="LOADER_INPUT",
        label="IPC receive -> expert weight_loader input",
        keys=LOGICAL_KEYS,
        require_all_expected=False,
    )
    _compare_paired_stages(
        expected_records=loader_input_records,
        actual_records=loader_output_records,
        issues=issues,
        stats=checks["loader_input_to_output"],
        category="EXPERT_WEIGHT_LOADER",
        label="Expert weight_loader input -> temporary fused parameter",
        keys=LOGICAL_KEYS,
        require_all_expected=True,
    )
    _compare_paired_stages(
        expected_records=loader_output_records,
        actual_records=finish_records,
        issues=issues,
        stats=checks["loader_output_to_finish"],
        category="LAYERWISE_FINALIZE",
        label="Expert loader output -> finalized runtime HF view",
        keys=LOGICAL_KEYS,
        require_all_expected=True,
    )
    _compare_paired_stages(
        expected_records=ipc_records,
        actual_records=finish_records,
        issues=issues,
        stats=checks["ipc_to_finish"],
        category="MOE_FINALIZE",
        label="IPC receive -> finalized runtime HF view",
        keys=LOGICAL_KEYS,
        require_all_expected=False,
    )
    _compare_paired_stages(
        expected_records=wake_records,
        actual_records=finish_records,
        issues=issues,
        stats=checks["wake_to_finish_layout"],
        category="MOE_LAYOUT",
        label="Weight wake -> finish layout",
        keys=LAYOUT_KEYS,
        require_all_expected=False,
    )

    for stage in ("actor_export", IPC_STAGE, LOADER_INPUT_STAGE, LOADER_OUTPUT_STAGE, FINISH_STAGE):
        if not any(record.stage == stage for record in records):
            issues.append(
                Issue(
                    severity="ERROR",
                    category="MISSING_STAGE",
                    message=f"Required stage {stage!r} was not found",
                )
            )
    if not wake_records:
        issues.append(
            Issue(
                severity="WARN",
                category="MISSING_STAGE",
                message=f"Optional stage {WAKE_STAGE!r} was not found; wake/finalize layout was not compared",
            )
        )

    return AnalysisReport(
        files=[str(path) for path in paths],
        records=len(records),
        stage_counts=dict(Counter(record.stage for record in records)),
        checks=checks,
        issues=issues,
        parse_errors=parse_errors,
        value_chains=_build_value_chains(records),
    )


def _print_report(
    report: AnalysisReport,
    *,
    max_details: int,
    show_value_chains: int,
) -> None:
    print("VIME 共卡权重更新探针报告")
    print(f"日志文件: {len(report.files)}")
    print(f"有效探针: {report.records}")
    print(f"stage 统计: {report.stage_counts}")
    print(f"可比较的真实值链路: {len(report.value_chains)}")
    print()

    check_labels = {
        "actor_cross_rank": "Actor 跨 rank 一致性",
        "actor_to_ipc": "Actor -> IPC",
        "ipc_to_loader_input": "IPC -> expert loader 输入",
        "loader_input_to_output": "expert loader 输入 -> 临时参数",
        "loader_output_to_finish": "临时参数 -> layerwise finalize",
        "ipc_to_finish": "IPC -> 最终 MoE 权重",
        "wake_to_finish_layout": "wake -> finish 布局",
    }
    for name, stats in report.checks.items():
        status = "PASS" if stats.failed == 0 and stats.compared > 0 else "FAIL"
        if stats.compared == 0:
            status = "SKIP"
        print(
            f"[{status}] {check_labels[name]}: "
            f"compared={stats.compared} passed={stats.passed} failed={stats.failed}"
        )

    print()
    if report.parse_errors:
        print(f"[ERROR] 无法解析的探针行: {len(report.parse_errors)}")
        for message in report.parse_errors[:max_details]:
            print(f"  - {message}")

    if report.issues:
        print(f"发现问题: error={report.error_count}, warning={report.warning_count}")
        for issue in report.issues[:max_details]:
            locations = f" ({', '.join(issue.locations)})" if issue.locations else ""
            print(f"[{issue.severity}][{issue.category}] {issue.message}{locations}")
        hidden = len(report.issues) - max_details
        if hidden > 0:
            print(f"... 另有 {hidden} 条，使用 --max-details 增大显示数量或 --json 查看全部")
    elif not report.parse_errors:
        print("未发现权重传输、expert loader、layerwise finalize 或布局不一致。")

    categories = {issue.category for issue in report.issues if issue.severity == "ERROR"}
    boundary_order = (
        ("ACTOR_VALUE", "Actor 导出阶段"),
        ("IPC_TRANSFER", "Actor → IPC"),
        ("LOADER_INPUT", "IPC → expert loader 输入"),
        ("EXPERT_WEIGHT_LOADER", "expert loader 输入 → 临时 fused 参数"),
        ("LAYERWISE_FINALIZE", "临时 fused 参数 → finalize 后运行权重"),
        ("MOE_FINALIZE", "IPC → 最终运行权重"),
        ("MOE_LAYOUT", "wake → finish 物理布局"),
    )
    first_abnormal_boundary = next(
        (
            (category, label)
            for category, label in boundary_order
            if category in categories or f"{category}_MISSING" in categories
        ),
        None,
    )
    if first_abnormal_boundary is not None:
        category, label = first_abnormal_boundary
        first_issue = next(issue for issue in report.issues if issue.category in {category, f"{category}_MISSING"})
        print()
        print(f"首个异常边界: {label}")
        print(f"首个异常详情: {first_issue.message}")

    if show_value_chains > 0 and report.value_chains:
        print()
        print(f"真实权重统计链路（前 {min(show_value_chains, len(report.value_chains))} 条）:")
        for chain in report.value_chains[:show_value_chains]:
            print(f"  uuid={chain['npu_uuid']!r} name={chain['name']!r}")
            for stage in chain["stages"]:
                value_stats = stage["value_stats"]
                compact_stats = {
                    key: value_stats.get(key)
                    for key in (
                        "min",
                        "max",
                        "mean",
                        "abs_mean",
                        "l2_norm",
                        "zero_count",
                        "nan_count",
                        "posinf_count",
                        "neginf_count",
                    )
                }
                print(
                    f"    {stage['stage']}: hash={stage['sample_hash']} "
                    f"stats={compact_stats} samples={stage['value_samples']}"
                )

    diagnoses = []
    if categories & {"ACTOR_METADATA", "ACTOR_VALUE"}:
        diagnoses.append("Actor 各 rank 的 HF 导出不一致。")
    if categories & {"IPC_TRANSFER", "IPC_TRANSFER_MISSING"}:
        diagnoses.append("Actor 导出与 vLLM IPC 接收不一致。")
    if categories & {"LOADER_INPUT", "LOADER_INPUT_MISSING"}:
        diagnoses.append("IPC 正确，但 expert loader 的输入异常或 loader 未被调用。")
    if categories & {"EXPERT_WEIGHT_LOADER", "EXPERT_WEIGHT_LOADER_MISSING"}:
        diagnoses.append(
            "expert loader 输入正确，但写入临时 fused 参数后异常；重点检查 EP expert 映射和 w1/w2/w3 shard。"
        )
    if categories & {"LAYERWISE_FINALIZE", "LAYERWISE_FINALIZE_MISSING"}:
        diagnoses.append(
            "expert loader 临时参数正确，但 finalize 后运行权重异常；重点检查 "
            "process_weights_after_loading、transpose/contiguous 和 ND/NZ format cast。"
        )
    if categories & {"MOE_FINALIZE", "MOE_FINALIZE_MISSING"} and not categories & {
        "EXPERT_WEIGHT_LOADER",
        "EXPERT_WEIGHT_LOADER_MISSING",
        "LAYERWISE_FINALIZE",
        "LAYERWISE_FINALIZE_MISSING",
    }:
        diagnoses.append("IPC 与最终运行时权重不一致，需结合 loader 输入/输出 stage 判断。")
    if categories & {"MOE_LAYOUT", "MOE_LAYOUT_MISSING"}:
        diagnoses.append("wake/finalize 物理布局不一致，检查 w13/w2 transpose 和 ND/NZ format。")
    if categories & {"MISSING_STAGE", "PROBE_RUNTIME"} or report.parse_errors:
        diagnoses.append("探针日志不完整或无法解析，需要使用修改后的代码重新运行一次。")
    if diagnoses:
        print()
        print("自动定位结论:")
        for diagnosis in diagnoses:
            print(f"  - {diagnosis}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Ray/VIME log file(s)")
    parser.add_argument("--json", type=Path, dest="json_path", help="Write the complete report as JSON")
    parser.add_argument("--max-details", type=int, default=30, help="Maximum issues printed to stdout")
    parser.add_argument(
        "--show-value-chains",
        type=int,
        default=0,
        metavar="N",
        help="Print compact real-value statistics for the first N weight chains",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return exit code 1 when the report only contains warnings",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    missing = [str(path) for path in args.logs if not path.is_file()]
    if missing:
        print(f"Log file not found: {missing}", file=sys.stderr)
        return 2

    report = analyze(args.logs)
    _print_report(
        report,
        max_details=max(1, args.max_details),
        show_value_chains=max(0, args.show_value_chains),
    )
    if args.json_path is not None:
        args.json_path.write_text(
            json.dumps(report.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 报告已写入: {args.json_path}")

    if report.error_count:
        return 1
    if args.fail_on_warning and report.warning_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
