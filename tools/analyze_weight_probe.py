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

LOGICAL_KEYS = ("shape", "dtype", "num_bytes", "sample_hash", "head", "tail")
LAYOUT_KEYS = ("shape", "stride", "storage_offset", "dtype", "npu_format")
IPC_STAGE = "ipc_receive_before_load"
WAKE_STAGE = "after_weight_wake_up"
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
    report_issues.append(
        Issue(
            severity="ERROR",
            category=category,
            message=(
                f"{label}: uuid={actual.npu_uuid!r} "
                f"name={normalize_weight_name(actual.name)!r} differences={differences}"
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
    keys_to_compare = set(actual_groups)
    if require_all_expected:
        keys_to_compare |= set(expected_groups)

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


def analyze(paths: list[Path]) -> AnalysisReport:
    records, issues, parse_errors = parse_probe_logs(paths)
    checks = {
        "actor_cross_rank": CheckStats(),
        "actor_to_ipc": CheckStats(),
        "ipc_to_finish": CheckStats(),
        "wake_to_finish_layout": CheckStats(),
    }

    _compare_actor_ranks(records, issues, checks["actor_cross_rank"])

    actor_records = [record for record in records if record.stage == "actor_export"]
    ipc_records = [record for record in records if record.stage == IPC_STAGE]
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

    for stage in ("actor_export", IPC_STAGE, FINISH_STAGE):
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
    )


def _print_report(report: AnalysisReport, *, max_details: int) -> None:
    print("VIME 共卡权重更新探针报告")
    print(f"日志文件: {len(report.files)}")
    print(f"有效探针: {report.records}")
    print(f"stage 统计: {report.stage_counts}")
    print()

    check_labels = {
        "actor_cross_rank": "Actor 跨 rank 一致性",
        "actor_to_ipc": "Actor -> IPC",
        "ipc_to_finish": "IPC -> MoE finalize",
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
        print("未发现权重传输、MoE finalize 或布局不一致。")

    categories = {issue.category for issue in report.issues if issue.severity == "ERROR"}
    diagnoses = []
    if categories & {"ACTOR_METADATA", "ACTOR_VALUE"}:
        diagnoses.append("Actor 各 rank 的 HF 导出不一致，优先检查 Bridge chunk 顺序和 native IPC 按位置合并")
    if categories & {"IPC_TRANSFER", "IPC_TRANSFER_MISSING"}:
        diagnoses.append("Actor 导出与 vLLM 接收不一致，优先检查 NPU IPC handle、UUID 路由和 rebuild")
    if categories & {"MOE_FINALIZE", "MOE_FINALIZE_MISSING"}:
        diagnoses.append("IPC 接收正确但运行时权重异常，优先检查 expert weight_loader 和 layerwise finalize")
    if categories & {"MOE_LAYOUT", "MOE_LAYOUT_MISSING"}:
        diagnoses.append("wake/finalize 物理布局不一致，优先检查 w13/w2 transpose 和 ND/NZ format 恢复")
    if categories & {"MISSING_STAGE", "PROBE_RUNTIME"} or report.parse_errors:
        diagnoses.append("探针日志不完整或无法解析，需要先补齐同一次运行的四个 stage")
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
    _print_report(report, max_details=max(1, args.max_details))
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
