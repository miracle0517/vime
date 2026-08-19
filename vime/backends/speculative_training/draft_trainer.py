from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import socket
from argparse import Namespace
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from vime.utils.common import is_npu
from vime.utils.misc import load_function

from .backends.dspark import (
    collate_dspark_samples,
    compute_dspark_loss,
    dspark_trainer_kwargs,
    has_valid_draft_vocab_mapping,
    sync_dspark_lm_heads,
)
from .backends.eagle3 import collate_eagle3_samples, compute_eagle3_loss
from .config import make_dspark_vllm_compatible_config
from .feature_schema import DraftFeatureSample, VersionedFeatureQueue

logger = logging.getLogger(__name__)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _cpu_contiguous_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Materialize a portable state dict before a worker writes it to disk.

    In particular, safetensors cannot reliably serialize Ascend NPU storages
    directly.  Keeping the staging tensors on CPU also prevents the saved
    checkpoint from depending on the accelerator device tag.
    """

    state_dict = {}
    for name, value in model.state_dict().items():
        if not torch.is_tensor(value):
            raise TypeError(f"Draft state dict value {name!r} is not a tensor")
        state_dict[name] = value.detach().to(device="cpu").contiguous()
    if not state_dict:
        raise RuntimeError("Draft state dict is empty")
    return state_dict


def _remove_export_artifact(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _publish_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _dspark_training_only_tensor(name: str) -> bool:
    """Return whether a Speculators tensor has no vLLM DSpark counterpart."""

    parts = name.split(".")
    return "verifier_lm_head" in parts or "verifier_norm" in parts or "t2d" in parts


def _load_draft_model(args: Namespace, device: torch.device) -> torch.nn.Module:
    factory_path = getattr(args, "draft_model_factory_path", None)
    algorithm = str(getattr(args, "draft_algorithm", "eagle3")).lower()
    if not factory_path and algorithm == "dspark":
        factory_path = "vime.backends.speculative_training.factories.speculators_dspark.build_model"
    if factory_path:
        model = load_function(factory_path)(args, device)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.draft_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The external Draft model factory must return torch.nn.Module")
    model = model.to(device)
    signature = inspect.signature(model.forward)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    required = {"input_ids", "hidden_states", "loss_mask"}
    if algorithm == "dspark":
        required.update({"verifier_last_hidden_states", "document_ids"})
    missing = required - set(signature.parameters)
    if missing and not accepts_kwargs:
        algorithm_name = "EAGLE3" if algorithm == "eagle3" else algorithm
        raise TypeError(
            f"The loaded Draft model is not {algorithm_name} training compatible; its forward must accept "
            f"{sorted(required)}. Missing {sorted(missing)}. Supply --draft-model-factory-path "
            "for checkpoints without Transformers auto_map training code."
        )
    config = getattr(model, "config", None)
    configured_target_hidden = getattr(config, "target_hidden_size", None)
    transformer_config = None
    configured_layer_ids = None
    if algorithm == "dspark":
        transformer_config = getattr(config, "transformer_layer_config", None)
        model_type = str(getattr(transformer_config, "model_type", "")).lower()
        if model_type and model_type != "qwen3":
            raise ValueError(
                "VIME currently supports only dense Qwen3 DSpark checkpoints, got "
                f"transformer_layer_config.model_type={model_type!r}"
            )
        if configured_target_hidden is None and transformer_config is not None:
            configured_target_hidden = getattr(transformer_config, "hidden_size", None)
        configured_layer_ids = getattr(config, "aux_hidden_state_layer_ids", None)
    target_hidden = int(getattr(args, "hidden_size", 0) or 0)
    if configured_target_hidden is not None and target_hidden > 0 and int(configured_target_hidden) != target_hidden:
        raise ValueError(
            "Draft checkpoint target_hidden_size does not match the Megatron Target: "
            f"{configured_target_hidden} != {target_hidden}"
        )
    configured_aux_count = getattr(config, "num_aux_hidden_states", None)
    if algorithm == "dspark" and configured_aux_count is None and configured_layer_ids is not None:
        configured_aux_count = len(configured_layer_ids)
    expected_layer_ids = tuple(int(value) for value in args.draft_feature_layer_ids)
    if configured_aux_count is not None and int(configured_aux_count) != len(expected_layer_ids):
        raise ValueError(
            "Draft checkpoint num_aux_hidden_states does not match --draft-feature-layer-ids: "
            f"{configured_aux_count} != {len(expected_layer_ids)}"
        )
    for value in (
        configured_layer_ids,
        getattr(config, "eagle_aux_hidden_state_layer_ids", None),
        getattr(config, "target_hidden_layer_ids", None),
        (
            (getattr(config, "eagle_config", None) or {}).get("target_hidden_layer_ids")
            if isinstance(getattr(config, "eagle_config", None), dict)
            else None
        ),
    ):
        if value is not None:
            configured_layer_ids = tuple(int(item) for item in value)
            break
    if configured_layer_ids is not None:
        target_depth = int(getattr(args, "num_layers", 0) or 0)
        configured_layer_ids = tuple(
            item + target_depth if item < 0 and target_depth > 0 else item for item in configured_layer_ids
        )
        if configured_layer_ids != expected_layer_ids:
            raise ValueError(
                "Draft checkpoint Target layer ids do not match feature collection: "
                f"{configured_layer_ids} != {expected_layer_ids}"
            )
    return model


def _architecture_fingerprint(model: torch.nn.Module) -> str:
    config = getattr(model, "config", None)
    config_dict = config.to_dict() if config is not None and hasattr(config, "to_dict") else {}
    identity = {
        "architectures": config_dict.get("architectures"),
        "model_type": config_dict.get("model_type"),
        "hidden_size": config_dict.get("hidden_size"),
        "draft_vocab_size": config_dict.get("draft_vocab_size", config_dict.get("vocab_size")),
        "num_aux_hidden_states": config_dict.get("num_aux_hidden_states"),
        "parameters": [(name, list(parameter.shape)) for name, parameter in model.named_parameters()],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint_tensor(model_path: str, key: str) -> torch.Tensor:
    path = Path(model_path)
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=model_path))

    index_paths = sorted(path.glob("*.safetensors.index.json")) + sorted(path.glob("*.bin.index.json"))
    checkpoint_path = None
    for index_path in index_paths:
        with index_path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        if key in weight_map:
            checkpoint_path = path / weight_map[key]
            break
    if checkpoint_path is None:
        for filename in ("model.safetensors", "pytorch_model.bin"):
            candidate = path / filename
            if candidate.exists():
                checkpoint_path = candidate
                break
    if checkpoint_path is None:
        raise FileNotFoundError(f"Cannot locate Target checkpoint tensor {key!r} under {model_path!r}")
    if checkpoint_path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                raise KeyError(f"Target checkpoint {checkpoint_path} does not contain {key!r}")
            return handle.get_tensor(key)
    state = torch.load(checkpoint_path, map_location="cpu")
    if key not in state:
        raise KeyError(f"Target checkpoint {checkpoint_path} does not contain {key!r}")
    return state[key]


def _load_target_embedding(model: torch.nn.Module, args: Namespace) -> None:
    model_path = getattr(args, "draft_target_embedding_path", None) or getattr(args, "hf_checkpoint", None)
    if not model_path:
        if str(getattr(args, "draft_algorithm", "eagle3")).lower() == "dspark":
            raise ValueError(
                "External DSpark training requires a Target checkpoint for Draft embedding initialization"
            )
        raise ValueError("External EAGLE3 training requires a Target checkpoint for Draft embedding initialization")
    key = str(getattr(args, "draft_target_embedding_key", "model.embed_tokens.weight"))
    custom_loader = getattr(model, "load_embedding", None)
    if callable(custom_loader):
        custom_loader(model_path, embedding_key=key)
        return
    embedding = getattr(model, "embed_tokens", None)
    if embedding is None:
        nested_model = getattr(model, "model", None)
        embedding = getattr(nested_model, "embed_tokens", None)
    weight = getattr(embedding, "weight", None)
    if not torch.is_tensor(weight):
        if str(getattr(args, "draft_algorithm", "eagle3")).lower() == "dspark":
            raise RuntimeError("DSpark Draft model does not expose embed_tokens.weight or load_embedding()")
        raise RuntimeError("EAGLE3 Draft model does not expose embed_tokens.weight or load_embedding()")
    source = _load_checkpoint_tensor(str(model_path), key)
    if source.shape != weight.shape:
        raise ValueError(
            f"Target embedding shape {tuple(source.shape)} does not match Draft embedding {tuple(weight.shape)}"
        )
    with torch.no_grad():
        weight.copy_(source.to(device=weight.device, dtype=weight.dtype))


def _resolve_draft_vocab_state(model: torch.nn.Module) -> tuple[torch.Tensor | None, int | None]:
    output_weight = getattr(getattr(model, "lm_head", None), "weight", None)
    configured_size = getattr(getattr(model, "config", None), "draft_vocab_size", None)
    draft_vocab_size = (
        int(output_weight.size(0))
        if torch.is_tensor(output_weight)
        else (int(configured_size) if configured_size is not None else None)
    )
    t2d = getattr(model, "t2d", None)
    if not torch.is_tensor(t2d):
        return None, draft_vocab_size
    rows = torch.nonzero(t2d.detach().bool(), as_tuple=False).reshape(-1)
    if torch.is_tensor(output_weight) and output_weight.size(0) != rows.numel():
        raise ValueError(
            "Draft t2d mapping selects a different number of Target rows than the Draft LM Head: "
            f"{rows.numel()} != {output_weight.size(0)}"
        )
    return rows, draft_vocab_size


class ExternalDraftTrainer:
    def __init__(self, args: Namespace, *, distributed: bool = True) -> None:
        self.args = args
        self.algorithm = str(getattr(args, "draft_algorithm", "eagle3")).lower()
        self.device_type = "npu" if is_npu() else "cuda"
        self.current_device = torch.npu.current_device() if is_npu() else torch.cuda.current_device()
        self.device = torch.device(self.device_type, self.current_device)
        # A dedicated Draft group participates in its own torch.distributed
        # process group.  The Actor-colocated mode deliberately runs only on
        # Actor rank zero, even though the surrounding Actor process belongs to
        # a TP process group.  Treat it as a local trainer so Draft backward
        # never enters Actor TP/DDP collectives while the other Actor ranks are
        # idle.
        self.distributed = bool(distributed)
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.model = _load_draft_model(args, self.device)
        raw_model = _unwrap_model(self.model)
        self.architecture_fingerprint = _architecture_fingerprint(raw_model)
        _load_target_embedding(raw_model, args)
        if bool(getattr(args, "draft_freeze_embeddings", True)):
            embedding = getattr(raw_model, "embed_tokens", None)
            if embedding is not None and hasattr(embedding, "weight"):
                embedding.weight.requires_grad_(False)
        mapping_path = getattr(args, "draft_vocab_mapping_path", None)
        if mapping_path:
            if self.algorithm != "dspark":
                if hasattr(raw_model, "load_vocab_mapping"):
                    raw_model.load_vocab_mapping(mapping_path)
                else:
                    mapping = torch.load(mapping_path, map_location=self.device)
                    for name in ("t2d", "d2t"):
                        if name in mapping and hasattr(raw_model, name):
                            destination = getattr(raw_model, name)
                            destination.copy_(mapping[name].to(destination.device))
            else:
                mapping = torch.load(mapping_path, map_location="cpu")
                if not isinstance(mapping, dict) or not all(
                    torch.is_tensor(mapping.get(name)) for name in ("t2d", "d2t")
                ):
                    raise ValueError("--draft-vocab-mapping-path must contain tensor entries 't2d' and 'd2t'")
                mapping_loader = getattr(raw_model, "load_vocab_mappings", None)
                if callable(mapping_loader):
                    mapping_loader(mapping["t2d"], mapping["d2t"])
                else:
                    for name in ("t2d", "d2t"):
                        destination = getattr(raw_model, name, None)
                        if not torch.is_tensor(destination) or destination.shape != mapping[name].shape:
                            raise ValueError(
                                f"Draft model {name} buffer is unavailable or has shape "
                                f"{getattr(destination, 'shape', None)}, mapping provides {mapping[name].shape}"
                            )
                        destination.copy_(mapping[name].to(device=destination.device, dtype=destination.dtype))
                if not has_valid_draft_vocab_mapping(raw_model):
                    raise ValueError(
                        "--draft-vocab-mapping-path t2d/d2t must be one-dimensional exact inverses: draft index "
                        "+ d2t offset must equal the ordered Target row selected by t2d"
                    )
        self.draft_to_target_rows, self.draft_vocab_size = _resolve_draft_vocab_state(raw_model)
        self.algorithm_train_kwargs = dspark_trainer_kwargs(raw_model, args) if self.algorithm == "dspark" else {}
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("External Draft model has no trainable parameters")
        if self.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.current_device],
                output_device=self.current_device,
            )
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=float(args.draft_learning_rate),
            weight_decay=float(args.draft_weight_decay),
        )
        warmup_steps = int(getattr(args, "draft_lr_warmup_steps", 0))
        configured_total = int(getattr(args, "draft_lr_total_steps", 0))
        estimated_triggers = max(
            int(getattr(args, "num_rollout", 1) or 1) // int(args.draft_train_interval),
            1,
        )
        total_steps = configured_total or estimated_triggers * int(args.draft_train_steps_per_trigger)

        def lr_multiplier(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if str(getattr(args, "draft_lr_scheduler_type", "constant")) == "constant":
                return 1.0
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_multiplier)
        self.queue = VersionedFeatureQueue(max_samples=int(args.draft_queue_max_samples))
        self.target_lm_head_weight: torch.Tensor | None = None
        self.target_weight_version: str | None = None
        self.optimizer_steps = 0
        self.draft_version = 0
        self.last_trained_rollout = -1
        if self.algorithm == "dspark":
            self.candidate_draft_version = 0
            self.candidate_target_weight_version: str | None = None
            self._candidate_lm_head_weight: torch.Tensor | None = None
        self._load_checkpoint_if_present()
        if self.algorithm == "dspark":
            # A DSpark resume checkpoint restores the full model state,
            # including t2d/d2t, after the source checkpoint was loaded.
            # Refresh the cached row selection before synchronizing LM Heads.
            self.draft_to_target_rows, self.draft_vocab_size = _resolve_draft_vocab_state(raw_model)
            self._capture_candidate_state()

    def _capture_candidate_state(self) -> None:
        """Freeze the small Target-synchronized part of the latest complete candidate."""

        self.candidate_draft_version = int(self.draft_version)
        self.candidate_target_weight_version = self.target_weight_version
        self._candidate_lm_head_weight = None
        if self.algorithm != "dspark":
            raise RuntimeError("Immutable candidate snapshots are implemented only for DSpark")
        raw_model = _unwrap_model(self.model)
        draft_weight = getattr(getattr(raw_model, "lm_head", None), "weight", None)
        verifier_weight = getattr(getattr(raw_model, "verifier_lm_head", None), "weight", None)
        if not torch.is_tensor(draft_weight) or not torch.is_tensor(verifier_weight):
            raise RuntimeError("Speculators DSpark model must expose lm_head.weight and verifier_lm_head.weight")
        if draft_weight.shape != verifier_weight.shape:
            raise RuntimeError("DSpark lm_head and verifier_lm_head shapes must match")
        # Both frozen heads are intentionally identical. Keep one CPU copy and
        # reuse it for both state-dict entries to avoid a second full-vocabulary
        # host allocation while collect continues mutating the live heads.
        self._candidate_lm_head_weight = draft_weight.detach().to(device="cpu", copy=True).contiguous()

    def _candidate_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a full CPU state using heads from the last successful optimizer candidate."""

        if self.algorithm != "dspark":
            raise RuntimeError("Immutable candidate state materialization is implemented only for DSpark")
        state_dict = _cpu_contiguous_state_dict(_unwrap_model(self.model))
        if self._candidate_lm_head_weight is None:
            raise RuntimeError("DSpark candidate LM Head is unavailable")
        unmatched = {"lm_head.weight", "verifier_lm_head.weight"}
        for state_name in tuple(state_dict):
            for head_name in tuple(unmatched):
                if state_name == head_name or state_name.endswith(f".{head_name}"):
                    state_dict[state_name] = self._candidate_lm_head_weight.clone()
                    unmatched.discard(head_name)
                    break
        if unmatched:
            raise RuntimeError(f"DSpark candidate state is missing model entries for {sorted(unmatched)}")
        return state_dict

    def collect(self, payloads: list[dict[str, Any]], expected_version: str) -> int:
        samples = [DraftFeatureSample.from_payload(payload) for payload in payloads]
        return self.queue.add(samples, expected_version=str(expected_version))

    def sync_target_lm_head(self, weight: torch.Tensor, target_version: str) -> None:
        if not torch.is_tensor(weight) or weight.dim() != 2:
            raise ValueError("Target LM Head snapshot must be a two-dimensional tensor")
        if self.draft_to_target_rows is not None:
            if self.draft_to_target_rows.numel() == 0 or int(self.draft_to_target_rows.max().item()) >= weight.size(0):
                raise ValueError("Draft t2d mapping references rows outside the exported Target LM Head")
        elif self.draft_vocab_size is not None:
            if weight.size(0) < self.draft_vocab_size:
                raise ValueError(
                    f"Target LM Head has {weight.size(0)} rows but Draft logits use {self.draft_vocab_size} rows"
                )
            if weight.size(0) > self.draft_vocab_size:
                self.draft_to_target_rows = torch.arange(
                    self.draft_vocab_size,
                    dtype=torch.long,
                    device=self.device,
                )
        self.target_lm_head_weight = weight.detach().to(device=self.device, dtype=torch.bfloat16).contiguous()
        if self.algorithm == "dspark":
            sync_dspark_lm_heads(
                _unwrap_model(self.model),
                self.target_lm_head_weight,
                self.draft_to_target_rows,
            )
        self.target_weight_version = str(target_version)
        self.queue.clear_except(self.target_weight_version)

    def _local_available(self) -> int:
        if self.target_weight_version is None:
            return 0
        return self.queue.count(self.target_weight_version)

    def train(self, rollout_id: int) -> dict[str, float | int | str]:
        if self.target_lm_head_weight is None or self.target_weight_version is None:
            return {"trained": 0, "reason": "missing_target_head"}
        local_available = torch.tensor(self._local_available(), dtype=torch.long, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(local_available, op=dist.ReduceOp.MIN)
        if int(local_available.item()) <= 0:
            return {"trained": 0, "reason": "no_version_matched_features"}

        self.model.train()
        steps = int(self.args.draft_train_steps_per_trigger)
        batch_size = int(self.args.draft_batch_size_per_gpu)
        loss_sum = 0.0
        token_sum = 0.0
        top1_sum = 0.0
        top5_sum = 0.0
        if self.algorithm == "dspark":
            accept_rate_sum = 0.0
            accept_rate_total = 0.0
            accept_len_sum = 0.0
            accept_len_total = 0.0
            confidence_loss_sum = 0.0
            confidence_loss_total = 0.0
        grad_norm_sum = 0.0
        successful_steps = 0
        for _ in range(steps):
            samples = self.queue.take(self.target_weight_version, batch_size, repeat=True)
            if not samples:
                break
            if self.algorithm == "dspark":
                batch = collate_dspark_samples(
                    samples,
                    self.device,
                    block_size=int(self.args.draft_dspark_block_size),
                )
            else:
                batch = collate_eagle3_samples(samples, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                if self.algorithm == "dspark":
                    loss, metrics = compute_dspark_loss(self.model, batch, self.algorithm_train_kwargs)
                else:
                    loss, metrics = compute_eagle3_loss(
                        self.model,
                        batch,
                        self.target_lm_head_weight,
                        draft_to_target_ids=self.draft_to_target_rows,
                        temporal_decay=float(self.args.draft_temporal_decay),
                        ttt_length=int(self.args.draft_ttt_length),
                    )

            local_tokens = metrics["token_count"].detach().float()
            global_tokens = local_tokens.clone()
            finite = torch.tensor(
                float(bool(torch.isfinite(loss).item())),
                dtype=torch.float32,
                device=self.device,
            )
            if self.world_size > 1:
                dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if global_tokens.item() <= 0 or finite.item() <= 0:
                continue
            # DDP averages gradients across ranks. This scale makes that average
            # equal to global loss-sum / global valid-token-count, even when
            # local batches contain different numbers of active tokens.
            loss = loss * local_tokens * self.world_size / global_tokens
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                float(self.args.draft_max_grad_norm),
            )
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer_steps += 1
            successful_steps += 1
            loss_sum += float(metrics["loss_sum"].item())
            token_sum += float(metrics["token_count"].item())
            top1_sum += float(metrics["top1_correct"].item())
            top5_sum += float(metrics["top5_correct"].item())
            if self.algorithm == "dspark":
                accept_rate_sum += float(metrics.get("accept_rate_sum", loss.new_zeros(())).item())
                accept_rate_total += float(metrics.get("accept_rate_total", loss.new_zeros(())).item())
                accept_len_sum += float(metrics.get("accept_len_sum", loss.new_zeros(())).item())
                accept_len_total += float(metrics.get("accept_len_total", loss.new_zeros(())).item())
                confidence_loss_sum += float(metrics.get("confidence_loss_sum", loss.new_zeros(())).item())
                confidence_loss_total += float(metrics.get("confidence_loss_total", loss.new_zeros(())).item())
            grad_norm_sum += float(grad_norm.item())

        if self.world_size > 1:
            if self.algorithm == "dspark":
                reduced = torch.tensor(
                    [
                        loss_sum,
                        token_sum,
                        top1_sum,
                        top5_sum,
                        accept_rate_sum,
                        accept_rate_total,
                        accept_len_sum,
                        accept_len_total,
                        confidence_loss_sum,
                        confidence_loss_total,
                        grad_norm_sum,
                        float(successful_steps),
                    ],
                    dtype=torch.float64,
                    device=self.device,
                )
            else:
                reduced = torch.tensor(
                    [loss_sum, token_sum, top1_sum, top5_sum, grad_norm_sum, float(successful_steps)],
                    dtype=torch.float64,
                    device=self.device,
                )
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            if self.algorithm == "dspark":
                (
                    loss_sum,
                    token_sum,
                    top1_sum,
                    top5_sum,
                    accept_rate_sum,
                    accept_rate_total,
                    accept_len_sum,
                    accept_len_total,
                    confidence_loss_sum,
                    confidence_loss_total,
                    grad_norm_sum,
                    successful,
                ) = reduced.tolist()
            else:
                loss_sum, token_sum, top1_sum, top5_sum, grad_norm_sum, successful = reduced.tolist()
            successful_steps = int(successful / self.world_size)
        if successful_steps <= 0:
            return {"trained": 0, "reason": "no_valid_optimizer_step"}
        self.draft_version += 1
        self.last_trained_rollout = int(rollout_id)
        if self.algorithm == "dspark":
            self._capture_candidate_state()
        else:
            return {
                "trained": 1,
                "draft_version": self.draft_version,
                "target_weight_version": self.target_weight_version,
                "successful_steps": successful_steps,
                "loss": loss_sum / max(token_sum, 1.0),
                "top1_accuracy": top1_sum / max(token_sum, 1.0),
                "top5_accuracy": top5_sum / max(token_sum, 1.0),
                "valid_tokens": int(token_sum),
                "grad_norm": grad_norm_sum / max(successful_steps * self.world_size, 1),
                "optimizer_steps": self.optimizer_steps,
                "queue_samples": self.queue.count(self.target_weight_version),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
        result = {
            "trained": 1,
            "algorithm": self.algorithm,
            "draft_version": self.draft_version,
            "target_weight_version": self.target_weight_version,
            "successful_steps": successful_steps,
            "loss": loss_sum / max(token_sum, 1.0),
            "top1_accuracy": top1_sum / max(token_sum, 1.0),
            "valid_tokens": int(token_sum),
            "grad_norm": grad_norm_sum / max(successful_steps * self.world_size, 1),
            "optimizer_steps": self.optimizer_steps,
            "queue_samples": self.queue.count(self.target_weight_version),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        if accept_rate_total > 0:
            result["accept_rate"] = accept_rate_sum / accept_rate_total
        if accept_len_total > 0:
            result["expected_accept_length"] = accept_len_sum / accept_len_total
        if confidence_loss_total > 0:
            result["confidence_loss"] = confidence_loss_sum / confidence_loss_total
        return result

    def prepare_publish_snapshot(self) -> dict[str, Any] | None:
        if self.algorithm == "dspark":
            draft_version = self.candidate_draft_version
            target_weight_version = self.candidate_target_weight_version
        else:
            draft_version = self.draft_version
            target_weight_version = self.target_weight_version
        if self.rank != 0 or draft_version <= 0:
            return None
        if self.algorithm == "dspark" and target_weight_version is None:
            raise RuntimeError("The trained Draft candidate has no Target teacher version")
        dtype = _publish_dtype(str(self.args.draft_publish_dtype))
        raw_model = _unwrap_model(self.model)
        exporter = getattr(raw_model, "export_for_vllm", None)
        if callable(exporter):
            exported = exporter(dtype=dtype, device="cpu")
            named_tensors = list(exported.items()) if isinstance(exported, dict) else list(exported)
        else:
            named_tensors = [
                (name, parameter)
                for name, parameter in raw_model.named_parameters()
                if parameter.requires_grad or (self.algorithm == "dspark" and name == "lm_head.weight")
            ]
        normalized_tensors = []
        seen_names = set()
        published_candidate_head = self.algorithm != "dspark"
        for name, tensor in named_tensors:
            name = str(name)
            if self.algorithm == "dspark" and _dspark_training_only_tensor(name):
                continue
            if name in seen_names:
                raise ValueError(f"Draft publication contains duplicate parameter name {name!r}")
            if not torch.is_tensor(tensor):
                raise TypeError(f"Draft publication value {name!r} is not a tensor")
            seen_names.add(name)
            parts = name.split(".")
            if self.algorithm == "dspark" and parts[-2:] == ["lm_head", "weight"]:
                if self._candidate_lm_head_weight is None:
                    raise RuntimeError("DSpark candidate LM Head is unavailable")
                tensor = self._candidate_lm_head_weight
                published_candidate_head = True
            if self.algorithm != "dspark":
                tensor_dtype = dtype
            elif not tensor.is_floating_point():
                tensor_dtype = tensor.dtype
            elif "confidence_head" in parts:
                tensor_dtype = torch.float32
            else:
                tensor_dtype = dtype
            normalized_tensors.append((name, tensor.detach().to(device="cpu", dtype=tensor_dtype).contiguous()))
        if not normalized_tensors:
            raise RuntimeError("Draft publication snapshot is empty")
        if not published_candidate_head:
            raise RuntimeError("DSpark publication snapshot does not contain the candidate lm_head.weight")
        snapshot = {
            "named_tensors": normalized_tensors,
            "draft_version": str(draft_version),
            "trained_against_target_version": str(target_weight_version),
            "architecture_fingerprint": self.architecture_fingerprint,
        }
        if self.algorithm == "dspark":
            snapshot["algorithm"] = self.algorithm
        return snapshot

    def save_checkpoint(self, rollout_id: int) -> str | None:
        checkpoint_path = getattr(self.args, "draft_checkpoint_path", None)
        if self.rank != 0 or not checkpoint_path:
            return None
        directory = Path(checkpoint_path)
        directory.mkdir(parents=True, exist_ok=True)
        final_path = directory / "draft_latest.pt"
        temporary_path = directory / ".draft_latest.pt.tmp"
        if self.algorithm == "dspark":
            model_state_dict = self._candidate_model_state_dict()
            draft_version = self.candidate_draft_version
            target_weight_version = self.candidate_target_weight_version
        else:
            model_state_dict = _unwrap_model(self.model).state_dict()
            draft_version = self.draft_version
            target_weight_version = self.target_weight_version
        torch.save(
            {
                "model": model_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer_steps": self.optimizer_steps,
                "draft_version": draft_version,
                "target_weight_version": target_weight_version,
                "rollout_id": int(rollout_id),
                "architecture_fingerprint": self.architecture_fingerprint,
            },
            temporary_path,
        )
        if self.algorithm == "dspark":
            model_state_dict.clear()
        os.replace(temporary_path, final_path)
        return str(final_path)

    def export_hf_model(self, rollout_id: int) -> dict[str, Any] | None:
        """Export the trained DSpark model as a reloadable HuggingFace directory."""

        export_template = getattr(self.args, "draft_save_hf", None)
        if self.rank != 0 or not export_template:
            return None
        if self.algorithm != "dspark":
            raise ValueError("--draft-save-hf currently supports only --draft-algorithm=dspark")

        output_path = Path(str(export_template).format(rollout_id=int(rollout_id))).expanduser()
        source_path = Path(str(self.args.draft_model_path)).expanduser()
        if source_path.exists() and source_path.resolve() == output_path.resolve():
            raise ValueError("--draft-save-hf must not overwrite the original --draft-model-path")
        actor_export_template = getattr(self.args, "save_hf", None)
        if actor_export_template:
            actor_output_path = Path(str(actor_export_template).format(rollout_id=int(rollout_id))).expanduser()
            if actor_output_path.resolve() == output_path.resolve():
                raise ValueError("--draft-save-hf must not overwrite the Actor --save-hf directory")

        raw_model = _unwrap_model(self.model)
        save_pretrained = getattr(raw_model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise TypeError("The DSpark training model must implement save_pretrained() for --draft-save-hf")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.parent / f".{output_path.name}.tmp-{uuid4().hex}"
        backup_path = output_path.parent / f".{output_path.name}.backup-{uuid4().hex}"
        model_state_dict = self._candidate_model_state_dict()
        logger.info(
            "Saving %s Draft model in HuggingFace format to %s (%s CPU tensors)",
            self.algorithm,
            output_path,
            len(model_state_dict),
        )
        try:
            temporary_path.mkdir(parents=False, exist_ok=False)
            # The state dict is already materialized as contiguous CPU tensors,
            # so safetensors never sees an Ascend storage. Force one unsharded
            # file to match the original two-file DSpark checkpoint layout.
            save_pretrained(
                temporary_path,
                state_dict=model_state_dict,
                safe_serialization=True,
                max_shard_size="100GB",
            )
            config_path = temporary_path / "config.json"
            weight_path = temporary_path / "model.safetensors"
            if (
                not config_path.is_file()
                or config_path.stat().st_size == 0
                or not weight_path.is_file()
                or weight_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    "DSpark save_pretrained() did not produce a complete model directory with exactly two files at "
                    f"{temporary_path.resolve()}: config.json="
                    f"{config_path.is_file() and config_path.stat().st_size > 0}, "
                    f"model.safetensors={weight_path.is_file() and weight_path.stat().st_size > 0}"
                )
            try:
                from safetensors import safe_open

                with safe_open(weight_path, framework="pt", device="cpu") as handle:
                    saved_tensor_names = set(handle.keys())
                    if not saved_tensor_names:
                        raise RuntimeError("model.safetensors contains no tensors")
                    ignored_on_save = set(getattr(raw_model, "_keys_to_ignore_on_save", None) or ())
                    expected_tensor_names = set(model_state_dict) - ignored_on_save
                    if saved_tensor_names != expected_tensor_names:
                        missing = sorted(expected_tensor_names - saved_tensor_names)
                        unexpected = sorted(saved_tensor_names - expected_tensor_names)
                        raise RuntimeError(
                            "model.safetensors does not contain the exact DSpark state passed to save_pretrained: "
                            f"missing={missing}, unexpected={unexpected}"
                        )
            except Exception as exc:
                raise RuntimeError(f"DSpark export contains an invalid model.safetensors: {weight_path}") from exc
            try:
                saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"DSpark export contains an invalid config.json: {config_path}") from exc
            if not isinstance(saved_config, dict):
                raise RuntimeError(f"DSpark export config.json must contain a JSON object: {config_path}")
            model_config = getattr(raw_model, "config", None)
            transformer_config = getattr(model_config, "transformer_layer_config", None)
            try:
                saved_config = make_dspark_vllm_compatible_config(
                    saved_config,
                    transformer_config=transformer_config,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "DSpark export config.json cannot be converted to the combined vLLM/Speculators schema"
                ) from exc
            config_path.write_text(
                json.dumps(saved_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            retained_names = {"config.json", "model.safetensors"}
            for artifact in temporary_path.iterdir():
                if artifact.name in retained_names:
                    continue
                if artifact.is_dir():
                    shutil.rmtree(artifact)
                else:
                    artifact.unlink()

            if output_path.exists():
                os.replace(output_path, backup_path)
            try:
                os.replace(temporary_path, output_path)
            except BaseException:
                if backup_path.exists() and not output_path.exists():
                    os.replace(backup_path, output_path)
                raise
            if backup_path.exists():
                _remove_export_artifact(backup_path)
        finally:
            model_state_dict.clear()
            if temporary_path.exists():
                _remove_export_artifact(temporary_path)
            if backup_path.exists() and output_path.exists():
                _remove_export_artifact(backup_path)

        weight_files = ["model.safetensors"]
        result = {
            "complete": True,
            "path": str(output_path.resolve()),
            "hostname": socket.gethostname(),
            "weight_files": weight_files,
            "weight_bytes": sum((output_path / name).stat().st_size for name in weight_files),
            "draft_version": self.candidate_draft_version,
            "trained_against_target_version": self.candidate_target_weight_version,
            "rollout_id": int(rollout_id),
            "algorithm": self.algorithm,
        }
        logger.info("Successfully saved external Draft model: %s", result)
        return result

    def _load_checkpoint_if_present(self) -> None:
        checkpoint_path = getattr(self.args, "draft_checkpoint_path", None)
        if not checkpoint_path:
            return
        path = Path(checkpoint_path) / "draft_latest.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location=self.device)
        saved_fingerprint = state.get("architecture_fingerprint")
        if saved_fingerprint is not None and str(saved_fingerprint) != self.architecture_fingerprint:
            raise RuntimeError(
                "External Draft checkpoint architecture does not match the configured model: "
                f"{saved_fingerprint} != {self.architecture_fingerprint}"
            )
        _unwrap_model(self.model).load_state_dict(state["model"], strict=True)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        self.optimizer_steps = int(state.get("optimizer_steps", 0))
        self.draft_version = int(state.get("draft_version", 0))
        saved_target_version = state.get("target_weight_version")
        self.target_weight_version = None if saved_target_version is None else str(saved_target_version)
        self.last_trained_rollout = int(state.get("rollout_id", -1))
        logger.info("Restored external Draft checkpoint %s at Draft version %s", path, self.draft_version)
