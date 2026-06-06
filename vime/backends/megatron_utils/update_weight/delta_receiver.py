"""Receiver-side delta decode + apply (engine-agnostic pure torch).

Ported VERBATIM from slime's ``docker/patch/.../sglang.patch`` (the delta hunk
of ``model_runner.py``, lines ~1731-2070). The algorithm has zero sglang
dependency — it works on any ``torch.nn.Module`` with ``load_weights`` and
``named_parameters``/``named_buffers`` — so the only change versus the patch is
to lift the sglang ``ModelRunner`` methods into free functions that take the
model + device + chunk budget explicitly. In vime these are driven by the vLLM
worker (via ``vLLMColocateWorkerExtension``), where ``self.model_runner.model``
is the live vLLM model and ``self.device`` is the worker's CUDA device.

Wire layout (shared by nccl + disk transports): a uint8 ``__positions__`` byte
blob + a param-dtype ``__values__`` tensor + a per-bucket ``DeltaSpec`` manifest.
Each param decodes into a full-shape NaN-masked tensor (NaN = unchanged), and the
masked write is enforced by patching ``torch.Tensor.copy_``/``fill_`` for the
duration of one ``model.load_weights`` call (``_delta_apply_context``), so the
normal vLLM sharded load proceeds but only changed positions are overwritten.
"""

from __future__ import annotations

import bisect
import contextlib
import json
import math
from typing import Callable

import torch

from .delta_io import DeltaEncoding, DeltaParam, DeltaSpec


def decode_delta_one_param(
    encoding: DeltaEncoding,
    positions: torch.Tensor,
    values: torch.Tensor,
    p: DeltaParam,
    device: torch.device | str,
) -> torch.Tensor:
    """Decode one param's (positions, values) into a full-shape NaN-masked tensor.
    NaN at unchanged positions triggers the patched-copy on apply."""
    numel = math.prod(p.shape)
    param_dtype = p.dtype if isinstance(p.dtype, torch.dtype) else getattr(torch, p.dtype)
    flat = torch.full((numel,), float("nan"), dtype=param_dtype, device=device)
    val_slice = values[p.val_start : p.val_end]
    if val_slice.numel() == 0:
        return flat.view(tuple(p.shape))

    pos_bytes = positions[p.pos_start : p.pos_end]
    if encoding is DeltaEncoding.INDICES:
        width = 4  # int32 absolute indices
    elif encoding in (DeltaEncoding.DELTAS, DeltaEncoding.DELTAS_ZSTD):
        width = p.pos_width  # uint16 or uint32 gap-deltas
    else:
        raise ValueError(f"unsupported delta encoding: {encoding!r}")

    n_elems = pos_bytes.numel() // width
    b = pos_bytes.view(n_elems, width).to(torch.int64)
    if width == 2:
        unpacked = b[:, 0] | (b[:, 1] << 8)
    else:  # 4
        unpacked = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)

    if encoding is DeltaEncoding.INDICES:
        idx = unpacked
    else:
        # Sender encodes ``delta[k] = idx[k] - idx[k-1] - 1`` with idx[-1] := -1;
        # receiver inverts with ``idx = cumsum(delta + 1) - 1``.
        idx = (unpacked + 1).cumsum(dim=0) - 1
    # Sender may concat values across params of mixed dtypes (bf16 weights
    # + fp32 norms in one bucket); torch.cat promotes to the widest dtype,
    # so re-cast each slice back to the param's own dtype. The promoted
    # round-trip is exact (bf16 ⊂ fp32), no precision loss.
    flat.index_copy_(0, idx, val_slice.to(param_dtype))
    return flat.view(tuple(p.shape))


def apply_delta_payload(
    model: torch.nn.Module,
    encoding: DeltaEncoding,
    params: list[DeltaParam],
    positions: torch.Tensor,
    values: torch.Tensor,
    expected_checksum: int,
    device: torch.device | str,
    chunk_byte_cap: int,
) -> None:
    """Verify checksum, decode each param, apply via the patched-copy context.
    ``load_weights`` is called per ``chunk_byte_cap`` budget."""
    actual_checksum = delta_checksum(positions, values)
    if actual_checksum != expected_checksum:
        raise RuntimeError(
            f"delta checksum mismatch: expected={expected_checksum} got={actual_checksum}; "
            "indicates corruption between sender encode and receiver apply"
        )
    with delta_apply_context(model):
        chunk: list[tuple[str, torch.Tensor]] = []
        chunk_bytes = 0
        for p in params:
            t = decode_delta_one_param(encoding, positions, values, p, device)
            tensor_bytes = t.numel() * t.element_size()
            if chunk_bytes + tensor_bytes > chunk_byte_cap and chunk:
                model.load_weights(chunk)
                chunk = []
                chunk_bytes = 0
            chunk.append((p.name, t))
            chunk_bytes += tensor_bytes
        if chunk:
            model.load_weights(chunk)


def decode_and_apply_blob(
    model: torch.nn.Module,
    blob: bytes,
    device: torch.device | str,
    chunk_byte_cap: int,
) -> None:
    """Decode + apply one decompressed safetensors blob from the delta sender."""
    from safetensors.torch import load as st_load

    # st_load only returns tensors, so parse the header for metadata.
    hdr_len = int.from_bytes(blob[:8], "little")
    meta = json.loads(blob[8 : 8 + hdr_len]).get("__metadata__", {})
    encoding = DeltaEncoding(meta["encoding"])
    params = [DeltaParam(**p) for p in json.loads(meta["params"])]
    expected_checksum = int(meta["checksum"])

    tensors = st_load(blob)
    positions = tensors["__positions__"].to(device, non_blocking=True)
    values = tensors["__values__"].to(device, non_blocking=True)
    apply_delta_payload(
        model, encoding, params, positions, values, expected_checksum, device, chunk_byte_cap
    )


def apply_delta_files(
    model: torch.nn.Module,
    paths: list[str],
    device: torch.device | str,
    chunk_byte_cap: int,
    read_workers: int,
) -> tuple[bool, str]:
    """Read + decompress delta safetensors files in parallel, decode + apply each."""
    import concurrent.futures

    n_files = len(paths)
    workers = min(n_files, read_workers)

    def _read_and_decompress(path: str) -> bytes:
        with open(path, "rb") as fh:
            return maybe_zstd_decompress(fh.read())

    try:
        # Cap peak memory at workers × file_size by applying each batch before
        # prefetching the next.
        for i in range(0, n_files, workers):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                batch = list(pool.map(_read_and_decompress, paths[i : i + workers]))
            for blob in batch:
                decode_and_apply_blob(model, blob, device, chunk_byte_cap)
        return True, f"Applied {n_files} delta file(s)"
    except Exception as e:  # noqa: BLE001
        return False, f"Failed to apply delta update from disk: {e}."


def param_storage_index(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor | None]:
    """Build ``find_parent(dst)``: looks up the param/buffer owning ``dst``'s storage,
    or None. Used by ``delta_apply_context`` to scope its patched copy_/fill_."""
    starts: list[int] = []
    ends: list[int] = []
    owners: list[torch.Tensor] = []
    seen: set = set()
    for tensors in (model.named_parameters(), model.named_buffers()):
        for _, t in tensors:
            if t.is_meta:
                continue
            try:
                ptr = t.data_ptr()
            except RuntimeError:
                continue
            if ptr == 0 or ptr in seen:
                continue
            seen.add(ptr)
            sz = t.numel() * t.element_size()
            starts.append(ptr)
            ends.append(ptr + sz)
            owners.append(t)
    order = sorted(range(len(starts)), key=lambda i: starts[i])
    starts = [starts[i] for i in order]
    ends = [ends[i] for i in order]
    owners = [owners[i] for i in order]

    def find_parent(dst):
        try:
            ptr = dst.data_ptr()
        except RuntimeError:
            return None
        idx = bisect.bisect_right(starts, ptr) - 1
        if 0 <= idx < len(starts) and starts[idx] <= ptr < ends[idx]:
            return owners[idx]
        return None

    return find_parent


@contextlib.contextmanager
def delta_apply_context(model: torch.nn.Module):
    """Patch ``copy_`` / ``fill_`` so writes into ``model``'s param storage skip
    positions whose source is NaN. Non-param writes go through unmodified.
    ``post_load_weights`` runs in the original env so derived tensors (fp8 scales,
    MoE biases, w_kc/w_vc) overwrite as usual."""
    is_param_target = param_storage_index(model)
    original_copy_ = torch.Tensor.copy_
    original_fill_ = torch.Tensor.fill_

    def patched_copy_(self, src, *args, **kwargs):
        if is_param_target(self) is not None:
            src_aligned = (
                src.to(device=self.device, dtype=self.dtype) if src.dtype != self.dtype else src
            )
            mask = ~torch.isnan(src_aligned)
            self[mask] = src_aligned[mask]
            return self
        return original_copy_(self, src, *args, **kwargs)

    def patched_fill_(self, value):
        if is_param_target(self) is not None:
            # NaN scalar means "don't change the param" (per-element analog of
            # patched_copy_). Non-NaN scalars write through.
            try:
                if math.isnan(value):
                    return self
            except TypeError:
                pass
            return original_fill_(self, value)
        return original_fill_(self, value)

    original_post_load = getattr(model, "post_load_weights", None)
    if original_post_load is not None:

        def wrapped_post_load(*args, **kwargs):
            current_copy = torch.Tensor.copy_
            current_fill = torch.Tensor.fill_
            torch.Tensor.copy_ = original_copy_
            torch.Tensor.fill_ = original_fill_
            try:
                return original_post_load(*args, **kwargs)
            finally:
                torch.Tensor.copy_ = current_copy
                torch.Tensor.fill_ = current_fill

        model.post_load_weights = wrapped_post_load

    torch.Tensor.copy_ = patched_copy_
    torch.Tensor.fill_ = patched_fill_
    try:
        yield
    finally:
        torch.Tensor.copy_ = original_copy_
        torch.Tensor.fill_ = original_fill_
        if original_post_load is not None:
            model.post_load_weights = original_post_load


def delta_checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    """Wire-corruption check, must match the sender's computation."""
    p = int(torch.hash_tensor(positions).item()) if positions.numel() else 0
    v = int(torch.hash_tensor(values).item()) if values.numel() else 0
    return p ^ (v << 1)


def maybe_zstd_decompress(blob: bytes) -> bytes:
    """Decompress if zstd-framed (sender uses zstd when encoding=deltas_zstd)."""
    # Zstandard frame magic: 0xFD2FB528 little-endian (RFC 8478 §3.1.1).
    if blob.startswith(b"\x28\xb5\x2f\xfd"):
        import zstandard

        return zstandard.ZstdDecompressor().decompress(blob)
    return blob
