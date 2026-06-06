"""Wire structs for delta weight sync.

Ported from slime, which defines these in the sglang ``io_struct`` module (and
ships them via ``docker/patch/.../sglang.patch``). vime has no sglang dependency,
so the structs live here and are shared by both ends of the wire:

  - the trainer encoder (``update_weight_from_distributed_delta.py``), and
  - the receiver decoder (``delta_receiver.py``, mixed into the vLLM worker via
    ``vLLMColocateWorkerExtension``).

Three ``DeltaEncoding`` variants differ only in how the changed-position blob is
packed; ``DeltaParam`` slices the shared (positions, values) bucket per param;
``DeltaSpec`` is the per-bucket decoding manifest that travels as JSON alongside
the NCCL broadcast / disk safetensors payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DeltaEncoding(str, Enum):
    """Position encoding for delta weight updates."""

    # int32 absolute nonzero offsets.
    INDICES = "indices"
    # uint16 gap-deltas between consecutive sorted positions; uint32 per-param fallback.
    DELTAS = "deltas"
    # ``deltas`` wrapped in zstd L1.
    DELTAS_ZSTD = "deltas_zstd"


@dataclass
class DeltaParam:
    """Per-param slice into the shared (positions, values) bucket. ``pos_*`` index
    into the uint8 byte blob; ``val_*`` index into the param-dtype value tensor."""

    name: str
    dtype: str
    shape: list[int]
    pos_start: int
    pos_end: int
    pos_width: int  # 2 or 4
    val_start: int
    val_end: int


@dataclass
class DeltaSpec:
    """Decoding manifest for one delta bucket. ``checksum`` is verified on apply."""

    encoding: DeltaEncoding
    params: list[DeltaParam] = field(default_factory=list)
    checksum: int = 0
