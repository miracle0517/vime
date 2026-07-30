"""
Colocated vLLM weight sync using native IPC transfer engines.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from vime.utils.common import is_npu
from vime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: build CUDA IPC handles → all_gather_object(Gloo CPU, over the engine
    slot ranks) → Ray IPC to engine.  Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._model_update_groups = None
        # vLLM #39212 IPC transfer-engine init runs once per set of colocated engines.
        self._ipc_initialized = False
        # vLLM IPC handle payloads are pickled on the Ray/HTTP bridge.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "vime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        # vLLM #39212: one-time IPC transfer-engine init on each colocated engine.
        if dist.get_rank() == 0 and self.rollout_engines and not self._ipc_initialized:
            ray.get([engine.init_weight_transfer_engine.remote({"init_info": {}}) for engine in self.rollout_engines])
            self._ipc_initialized = True

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Empty under colocate today;
        kept symmetric with UpdateWeightFromDistributed so the actor can drain unconditionally.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    # ------------------------------------------------------------------
    # weight update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        # Enter the native vLLM weight-update state machine on every colocated engine.
        if rank == 0:
            ray.get([engine.start_weight_update.remote(is_checkpoint_format=True) for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            # Free chunk tensors so the caching allocator can reuse the blocks.
            del hf_named_tensors
            if is_npu():
                torch.npu.synchronize()
            else:
                torch.cuda.ipc_collect()

        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        if is_npu():
            torch.npu.synchronize()
        else:
            torch.cuda.ipc_collect()

        # Exit the native vLLM weight-update state machine.
        if rank == 0:
            ray.get([engine.finish_weight_update.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> list[object]:
        all_refs: list[object] = []

        _send_to_colocated_engine(
            hf_named_tensors,
            rollout_engines=self.rollout_engines,
            weight_version=self.weight_version,
        )

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
                packed=False,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    rollout_engines: Sequence[ActorHandle],
    weight_version: int,
) -> None:
    if not rollout_engines:
        return

    def send_to_vllm(update_info) -> None:
        request = {"update_info": asdict(update_info)}
        ray.get(
            [engine.update_weights.remote(request, weight_version=str(weight_version)) for engine in rollout_engines]
        )

    if is_npu():
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
            NPUIPCTrainerSendWeightsArgs,
            NPUIPCWeightTransferEngine,
        )

        trainer_args = NPUIPCTrainerSendWeightsArgs(send_mode=send_to_vllm, packed=False)
        NPUIPCWeightTransferEngine.trainer_send_weights(iter(hf_named_tensors), trainer_args)
    else:
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerSendWeightsArgs, IPCWeightTransferEngine

        trainer_args = IPCTrainerSendWeightsArgs(send_mode=send_to_vllm, packed=False)
        IPCWeightTransferEngine.trainer_send_weights(iter(hf_named_tensors), trainer_args)


# ---------------------------------------------------------------------------
# vLLM worker extension (loaded by ``--worker-extension-cls``)
# ---------------------------------------------------------------------------


class _VLLMHijack:
    """vLLM worker extension helpers.

    On NPU:
    - Patches NPUWorker.load_model and NPUWorker.start_weight_update to fix
      MoE weight_loader missing on EP (a vLLM bug where w13_weight/w2_weight
      params lack weight_loader attr when EP is enabled).
    - Patches ApplyRotaryEmb.__init__ to skip flash_attn import
      (mindspeed/megatron backends introduce flash_attn as a dummy module,
      but vllm_ascend does not use it).
    """

    @staticmethod
    def _patch_npu_worker() -> None:
        from vllm_ascend.worker.worker import NPUWorker

        if getattr(NPUWorker, "_npu_worker_patched", False):
            return

        _VLLMHijack._patch_one_worker(NPUWorker)
        NPUWorker._npu_worker_patched = True

    @staticmethod
    def _patch_a3_moe_alltoall_padding() -> None:
        """Fix uneven TP token splits in the Ascend A3 MoE ALLTOALL path."""
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() != AscendDeviceType.A3:
            return

        from vllm_ascend.ops.fused_moe.prepare_finalize import MoEPrepareOutput, PrepareAndFinalizeWithAll2All

        if getattr(PrepareAndFinalizeWithAll2All, "_vime_alltoall_padding_patched", False):
            return

        def _patched_prepare(
            self,
            hidden_states,
            router_logits,
            enable_shared_expert_dp=False,
            replace_allreduce=False,
            quant_type=None,
        ):
            self.replace_allreduce = replace_allreduce
            self.enable_shared_expert_dp = enable_shared_expert_dp

            padded_hidden_states_shape = hidden_states.shape
            if not (self.replace_allreduce or self.enable_shared_expert_dp):
                self.num_tokens, _ = hidden_states.shape
                pad_size = (-self.num_tokens) % self.tp_size
                if pad_size > 0:
                    hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                    router_logits = torch.nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                    padded_hidden_states_shape = hidden_states.shape

                if self.tp_size > 1:
                    hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)[self.tp_rank]
                    router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)[self.tp_rank]

            return MoEPrepareOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                mc2_mask=None,
                padded_hidden_states_shape=padded_hidden_states_shape,
                pertoken_scale=None,
            )

        def _patched_pad_and_split_input_ids(self, input_ids):
            if not (self.replace_allreduce or self.enable_shared_expert_dp):
                pad_size = (-self.num_tokens) % self.tp_size
                if pad_size > 0:
                    input_ids = torch.nn.functional.pad(input_ids, (0, pad_size))

                if self.tp_size > 1:
                    input_ids = torch.tensor_split(input_ids, self.tp_size, dim=0)[self.tp_rank]
            return input_ids

        PrepareAndFinalizeWithAll2All.prepare = _patched_prepare
        PrepareAndFinalizeWithAll2All.pad_and_split_input_ids = _patched_pad_and_split_input_ids
        PrepareAndFinalizeWithAll2All._vime_alltoall_padding_patched = True
        logger.info("Colocated Ascend A3 detected: installed MoE ALLTOALL TP padding fix")

    @staticmethod
    def _patch_a3_moe_unpermute_probe() -> None:
        """Validate the final ALLTOALL token unpermute against a CPU reference."""
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() != AscendDeviceType.A3:
            return

        from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV

        if getattr(TokenDispatcherWithAll2AllV, "_vime_unpermute_probe_patched", False):
            return

        import torch_npu

        original_dispatch_preprocess = TokenDispatcherWithAll2AllV._dispatch_preprocess
        original_dispatch_postprocess = TokenDispatcherWithAll2AllV._dispatch_postprocess
        original_combine_preprocess = TokenDispatcherWithAll2AllV._combine_preprocess
        original_combine_postprocess = TokenDispatcherWithAll2AllV._combine_postprocess
        original_grouped_matmul = torch_npu.npu_grouped_matmul

        def _row_stats(tensor):
            rows = tensor.detach().reshape(-1, tensor.shape[-1])
            sampled = torch.round(rows[:, :64].float() * 4096).to(torch.int32).cpu()
            _, counts = torch.unique(sampled, dim=0, return_counts=True)
            norms = torch.linalg.vector_norm(rows.float(), dim=1).cpu()
            return {
                "rows": rows.shape[0],
                "unique": counts.numel(),
                "duplicate_rows": rows.shape[0] - counts.numel(),
                "max_duplicate_group": int(counts.max().item()) if counts.numel() else 0,
                "norm_min": float(norms.min().item()) if norms.numel() else 0.0,
                "norm_max": float(norms.max().item()) if norms.numel() else 0.0,
                "norm_mean": float(norms.mean().item()) if norms.numel() else 0.0,
            }

        def _buffer_info(tensor):
            storage = tensor.untyped_storage()
            return {
                "data_ptr": tensor.data_ptr(),
                "storage_data_ptr": storage.data_ptr(),
                "storage_nbytes": storage.nbytes(),
                "tensor_nbytes": tensor.numel() * tensor.element_size(),
                "storage_offset": tensor.storage_offset(),
                "stride": list(tensor.stride()),
                "contiguous": tensor.is_contiguous(),
            }

        def _chunk_hashes(tensor, splits):
            tensor_cpu = tensor.detach().contiguous().to(device="cpu")
            hashes = []
            offset = 0
            for size in splits:
                size = int(size)
                chunk = tensor_cpu.narrow(0, offset, size).contiguous()
                hashes.append(hashlib.sha256(chunk.view(torch.uint8).numpy().tobytes()).hexdigest())
                offset += size
            return hashes

        def _grouped_matmul_probe(*args, **kwargs):
            output = original_grouped_matmul(*args, **kwargs)
            if not getattr(TokenDispatcherWithAll2AllV, "_vime_unpermute_probe_pending", False):
                return output

            try:
                x = kwargs.get("x", args[0] if args else None)
                weight = kwargs.get("weight", args[1] if len(args) > 1 else None)
                group_list = kwargs.get("group_list")
                group_list_type = kwargs.get("group_list_type", 0)
                x = x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x
                weight = weight[0] if isinstance(weight, (list, tuple)) and len(weight) == 1 else weight
                actual = output[0] if isinstance(output, (list, tuple)) else output
                if not (
                    isinstance(x, torch.Tensor)
                    and isinstance(weight, torch.Tensor)
                    and weight.ndim == 3
                    and isinstance(actual, torch.Tensor)
                    and group_list is not None
                ):
                    return output

                groups = torch.as_tensor(group_list).detach().to(device="cpu", dtype=torch.int64).reshape(-1)
                counts = groups if group_list_type == 1 else torch.diff(torch.nn.functional.pad(groups, (1, 0)))
                nonempty = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
                if nonempty.numel() > 4:
                    indices = torch.linspace(0, nonempty.numel() - 1, steps=4).round().to(torch.int64)
                    nonempty = nonempty.index_select(0, indices)

                starts = torch.cumsum(counts, dim=0) - counts
                samples = []
                for expert_tensor in nonempty:
                    expert = int(expert_tensor.item())
                    row = int(starts[expert].item())
                    expert_weight = weight[expert]
                    if x.shape[-1] == expert_weight.shape[-1]:
                        expert_weight = expert_weight.transpose(0, 1)
                    if x.shape[-1] != expert_weight.shape[0]:
                        continue
                    reference = torch.matmul(x[row : row + 1].float(), expert_weight.float())
                    actual_row = actual[row : row + 1].float()
                    abs_diff = (actual_row - reference).abs()
                    samples.append(
                        {
                            "expert": expert,
                            "row": row,
                            "max_abs_diff": float(abs_diff.max().item()),
                            "mean_abs_diff": float(abs_diff.mean().item()),
                            "reference_abs_max": float(reference.abs().max().item()),
                        }
                    )

                TokenDispatcherWithAll2AllV._vime_gmm_probe.append(
                    {
                        "index": len(TokenDispatcherWithAll2AllV._vime_gmm_probe),
                        "x_shape": list(x.shape),
                        "weight_shape": list(weight.shape),
                        "output_shape": list(actual.shape),
                        "weight_data_ptr": weight.data_ptr(),
                        "weight_stride": list(weight.stride()),
                        "group_list_type": int(group_list_type),
                        "samples": samples,
                    }
                )
            except Exception:
                logger.exception("VIME_MOE_GMM_PROBE failed")
                TokenDispatcherWithAll2AllV._vime_gmm_probe.append({"probe_error": True})
            return output

        def _patched_dispatch_preprocess(self, hidden_states, topk_ids):
            result = original_dispatch_preprocess(self, hidden_states, topk_ids)
            if not getattr(TokenDispatcherWithAll2AllV, "_vime_unpermute_probe_pending", False):
                return result

            topk_cpu = topk_ids.detach().to(device="cpu", dtype=torch.int64)
            expert_counts = torch.bincount(topk_cpu.reshape(-1))
            top_experts = torch.topk(expert_counts, min(5, expert_counts.numel()))
            self._vime_moe_stage_probe = {
                "moe_input": _row_stats(hidden_states),
                "routing": {
                    "unique_patterns": torch.unique(topk_cpu, dim=0).shape[0],
                    "top_experts": torch.stack((top_experts.indices, top_experts.values), dim=1).tolist(),
                },
            }
            TokenDispatcherWithAll2AllV._vime_gmm_probe = []

            try:
                input_splits_cpu = torch.as_tensor(result[3], dtype=torch.int64)
                output_splits_cpu = torch.as_tensor(result[4], dtype=torch.int64)
                input_splits_device = input_splits_cpu.to(device=topk_ids.device)
                gathered_input_splits = [torch.empty_like(input_splits_device) for _ in range(self.ep_size)]
                dist.all_gather(
                    gathered_input_splits,
                    input_splits_device,
                    group=self.ep_group,
                )
                input_split_matrix = torch.stack(gathered_input_splits).to(device="cpu")
                expected_output_splits = input_split_matrix[:, self.ep_rank]
                output_split_diff = output_splits_cpu != expected_output_splits
                tokens_per_expert_sum = int(result[2].detach().to(device="cpu", dtype=torch.int64).sum().item())
                self._vime_tokens_per_expert = result[2].detach().to(device="cpu", dtype=torch.int64)
                split_probe = {
                    "input_splits": input_splits_cpu.tolist(),
                    "output_splits": output_splits_cpu.tolist(),
                    "expected_output_splits": expected_output_splits.tolist(),
                    "input_sum": int(input_splits_cpu.sum().item()),
                    "expected_input_sum": topk_ids.numel(),
                    "output_sum": int(output_splits_cpu.sum().item()),
                    "tokens_per_expert_sum": tokens_per_expert_sum,
                    "transpose_mismatch_count": int(output_split_diff.sum().item()),
                    "transpose_first_mismatch": (
                        int(torch.nonzero(output_split_diff, as_tuple=False)[0].item())
                        if output_split_diff.any()
                        else None
                    ),
                    "negative_split_count": int(((input_splits_cpu < 0).sum() + (output_splits_cpu < 0).sum()).item()),
                }
                self._vime_alltoall_split_probe = split_probe

                send_hashes = _chunk_hashes(result[0], input_splits_cpu.tolist())
                send_hash_bytes = torch.tensor(
                    [list(bytes.fromhex(value)) for value in send_hashes],
                    dtype=torch.uint8,
                    device=topk_ids.device,
                )
                gathered_send_hashes = [torch.empty_like(send_hash_bytes) for _ in range(self.ep_size)]
                dist.all_gather(gathered_send_hashes, send_hash_bytes, group=self.ep_group)
                send_hash_matrix = torch.stack(gathered_send_hashes).to(device="cpu")
                expected_recv_hashes = [
                    bytes(send_hash_matrix[src_rank, self.ep_rank].tolist()).hex() for src_rank in range(self.ep_size)
                ]
                self._vime_alltoall1_output_splits = output_splits_cpu.tolist()
                self._vime_alltoall1_send_tensor = result[0]
                self._vime_alltoall1_payload_probe = {
                    "send_hashes": send_hashes,
                    "expected_recv_hashes": expected_recv_hashes,
                    "send_buffer": _buffer_info(result[0]),
                }
                logger.warning(
                    "VIME_MOE_ALLTOALL_SPLIT_PROBE ep_rank=%s split=%s input_split_matrix=%s",
                    self.ep_rank,
                    split_probe,
                    input_split_matrix.tolist(),
                )
            except Exception:
                logger.exception("VIME_MOE_ALLTOALL_SPLIT_PROBE failed to validate split metadata")
                self._vime_alltoall_split_probe = {"split_probe_error": True}
                self._vime_alltoall1_payload_probe = {"payload_probe_error": True}

            try:
                actual_mapping = result[1].detach().to(device="cpu", dtype=torch.int64)
                flat_topk_ids = topk_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
                sorted_indices = torch.argsort(flat_topk_ids, stable=True)
                expected_mapping = torch.argsort(sorted_indices, stable=True)
                mapping_diff = actual_mapping != expected_mapping
                self._vime_unpermute_mapping_probe = {
                    "mapping_numel": actual_mapping.numel(),
                    "mapping_mismatch_count": int(mapping_diff.sum().item()),
                    "mapping_first_mismatch": (
                        int(torch.nonzero(mapping_diff, as_tuple=False)[0].item()) if mapping_diff.any() else None
                    ),
                }
            except Exception:
                logger.exception("VIME_MOE_UNPERMUTE_PROBE failed to validate the local permutation mapping")
                self._vime_unpermute_mapping_probe = {"mapping_probe_error": True}

            return result

        def _patched_dispatch_postprocess(
            self,
            global_input_tokens,
            dynamic_scale_after_all2all,
            global_input_tokens_local_experts_indices,
            with_quant,
        ):
            probe_pending = getattr(TokenDispatcherWithAll2AllV, "_vime_unpermute_probe_pending", False)
            original_tokens = global_input_tokens.detach().clone() if probe_pending else None
            if probe_pending:
                self._vime_moe_stage_probe["alltoall1_output"] = _row_stats(global_input_tokens)
                try:
                    local_expert_ids = (
                        global_input_tokens_local_experts_indices.detach()
                        .to(device="cpu", dtype=torch.int64)
                        .reshape(-1)
                    )
                    invalid_count = int(
                        ((local_expert_ids < 0) | (local_expert_ids >= self.num_local_experts)).sum().item()
                    )
                    actual_counts = (
                        torch.bincount(local_expert_ids, minlength=self.num_local_experts)
                        if invalid_count == 0
                        else torch.empty(0, dtype=torch.int64)
                    )
                    expected_counts = self._vime_tokens_per_expert
                    self._vime_expert_assignment_probe = {
                        "rows": local_expert_ids.numel(),
                        "invalid_expert_id_count": invalid_count,
                        "expert_count_mismatch_count": (
                            int((actual_counts != expected_counts).sum().item()) if invalid_count == 0 else None
                        ),
                        "actual_counts": actual_counts.tolist(),
                        "expected_counts": expected_counts.tolist(),
                    }
                    self._vime_local_expert_ids = local_expert_ids
                except Exception:
                    logger.exception("VIME_MOE_EXPERT_ASSIGNMENT_PROBE failed")
                    self._vime_expert_assignment_probe = {"probe_error": True}
                payload_probe = self._vime_alltoall1_payload_probe
                send_tensor = self._vime_alltoall1_send_tensor
                recv_hashes = _chunk_hashes(global_input_tokens, self._vime_alltoall1_output_splits)
                expected_recv_hashes = payload_probe["expected_recv_hashes"]
                mismatch_ranks = [
                    rank for rank in range(self.ep_size) if recv_hashes[rank] != expected_recv_hashes[rank]
                ]
                recv_buffer = _buffer_info(global_input_tokens)
                payload_probe.update(
                    {
                        "recv_hashes": recv_hashes,
                        "mismatch_count": len(mismatch_ranks),
                        "mismatch_src_ranks": mismatch_ranks,
                        "recv_buffer": recv_buffer,
                        "send_storage_nbytes_after_alltoall": send_tensor.untyped_storage().nbytes(),
                        "send_released_before_postprocess": send_tensor.untyped_storage().nbytes() == 0,
                        "send_recv_same_ptr": (payload_probe["send_buffer"]["data_ptr"] == recv_buffer["data_ptr"]),
                    }
                )
                del self._vime_alltoall1_send_tensor

            result = original_dispatch_postprocess(
                self,
                global_input_tokens,
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
                with_quant,
            )
            if not probe_pending:
                return result

            self._vime_moe_stage_probe["gmm_input"] = _row_stats(result[0])
            if result[2] is None:
                self._vime_moe_stage_probe["second_permute_roundtrip"] = {"skipped": True}
                self._vime_moe_stage_probe["second_permute_expert_order"] = {"skipped": True}
            else:
                assert original_tokens is not None
                restored_tokens = torch_npu.npu_moe_token_unpermute(result[0], result[2])
                roundtrip_diff = (restored_tokens - original_tokens).abs()
                self._vime_moe_stage_probe["second_permute_roundtrip"] = {
                    "max_abs_diff": float(roundtrip_diff.max().item()),
                    "mismatch_count": int(torch.count_nonzero(restored_tokens != original_tokens).item()),
                }
                try:
                    local_expert_ids = self._vime_local_expert_ids
                    reverse_mapping = result[2].detach().to(device="cpu", dtype=torch.int64).reshape(-1)
                    permuted_expert_ids = torch.empty_like(local_expert_ids)
                    permuted_expert_ids.scatter_(0, reverse_mapping, local_expert_ids)
                    counts = torch.bincount(local_expert_ids, minlength=self.num_local_experts)
                    expected_expert_ids = torch.repeat_interleave(
                        torch.arange(self.num_local_experts, dtype=torch.int64),
                        counts,
                    )
                    expert_order_diff = permuted_expert_ids != expected_expert_ids
                    expert_runs = torch.unique_consecutive(permuted_expert_ids)
                    self._vime_moe_stage_probe["second_permute_expert_order"] = {
                        "mismatch_count": int(expert_order_diff.sum().item()),
                        "first_mismatch": (
                            int(torch.nonzero(expert_order_diff, as_tuple=False)[0].item())
                            if expert_order_diff.any()
                            else None
                        ),
                        "run_count": expert_runs.numel(),
                        "expert_runs_head": expert_runs[:64].tolist(),
                    }
                except Exception:
                    logger.exception("VIME_MOE_SECOND_PERMUTE_ORDER_PROBE failed")
                    self._vime_moe_stage_probe["second_permute_expert_order"] = {"probe_error": True}
            return result

        def _patched_combine_preprocess(self, hidden_states, combine_metadata):
            probe_pending = getattr(TokenDispatcherWithAll2AllV, "_vime_unpermute_probe_pending", False)
            if probe_pending:
                self._vime_moe_stage_probe["gmm_output"] = _row_stats(hidden_states)
            output = original_combine_preprocess(self, hidden_states, combine_metadata)
            if probe_pending:
                self._vime_moe_stage_probe["after_second_unpermute"] = _row_stats(output)
            return output

        def _patched_combine_postprocess(self, permutated_local_input_tokens, combine_metadata):
            probe_pending = getattr(TokenDispatcherWithAll2AllV, "_vime_unpermute_probe_pending", False)
            # Preserve the exact operator input before calling unpermute. This
            # avoids a false comparison if the backend reuses or aliases the
            # input storage while producing the output.
            probe_input = permutated_local_input_tokens.detach().clone() if probe_pending else None
            output = original_combine_postprocess(self, permutated_local_input_tokens, combine_metadata)
            if not probe_pending:
                return output

            TokenDispatcherWithAll2AllV._vime_unpermute_probe_pending = False
            try:
                assert probe_input is not None
                permuted_cpu = probe_input.to(device="cpu", dtype=torch.float32)
                sorted_indices_cpu = (
                    combine_metadata.reversed_local_input_permutation_mapping.detach()
                    .to(device="cpu", dtype=torch.int64)
                    .reshape(-1)
                )
                probs_cpu = combine_metadata.topk_weights.detach().to(device="cpu", dtype=torch.float32)
                actual_cpu = output.detach().to(device="cpu", dtype=torch.float32).reshape(-1, output.shape[-1])

                # npu_moe_token_permute returns the inverse permutation
                # (argsort(argsort(flat_topk_ids))). Restore the expanded token
                # order by gathering from the permuted tensor with that inverse.
                expanded_reference = permuted_cpu.index_select(0, sorted_indices_cpu)
                reference_cpu = (
                    expanded_reference.reshape(probs_cpu.shape[0], probs_cpu.shape[1], -1) * probs_cpu.unsqueeze(-1)
                ).sum(dim=1)

                # Keep the previous scatter interpretation in the log so a
                # single run can distinguish an inverse-mapping interpretation
                # error from an actual NPU unpermute error.
                scatter_expanded_reference = torch.zeros(
                    (probs_cpu.numel(), permuted_cpu.shape[-1]),
                    dtype=torch.float32,
                )
                scatter_expanded_reference.index_copy_(0, sorted_indices_cpu, permuted_cpu)
                scatter_reference_cpu = (
                    scatter_expanded_reference.reshape(probs_cpu.shape[0], probs_cpu.shape[1], -1)
                    * probs_cpu.unsqueeze(-1)
                ).sum(dim=1)

                abs_diff = (actual_cpu - reference_cpu).abs()
                scatter_abs_diff = (actual_cpu - scatter_reference_cpu).abs()
                rel_diff = abs_diff / reference_cpu.abs().clamp_min(1e-6)
                mismatch = abs_diff > (1e-2 + 1e-2 * reference_cpu.abs())
                row_max_abs_diff = abs_diff.amax(dim=1)
                worst_row = int(row_max_abs_diff.argmax().item()) if row_max_abs_diff.numel() else None
                sample_rows = sorted({0, max(actual_cpu.shape[0] // 2, 0), max(actual_cpu.shape[0] - 1, 0)})
                row_samples = [
                    {
                        "row": row,
                        "max_abs_diff": float(row_max_abs_diff[row].item()),
                        "actual": actual_cpu[row, :4].tolist(),
                        "reference": reference_cpu[row, :4].tolist(),
                    }
                    for row in sample_rows
                    if row < actual_cpu.shape[0]
                ]
                self._vime_moe_stage_probe["moe_output"] = _row_stats(output)
                mapping_probe = getattr(self, "_vime_unpermute_mapping_probe", {})
                split_probe = getattr(self, "_vime_alltoall_split_probe", {})
                stage_probe = getattr(self, "_vime_moe_stage_probe", {})
                report = {
                    "ep_rank": self.ep_rank,
                    "input_shape": list(permuted_cpu.shape),
                    "output_shape": list(actual_cpu.shape),
                    "mapping": mapping_probe,
                    "split": split_probe,
                    "payload": getattr(self, "_vime_alltoall1_payload_probe", {}),
                    "expert_assignment": getattr(self, "_vime_expert_assignment_probe", {}),
                    "stages": stage_probe,
                    "unpermute": {
                        "max_abs_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
                        "mean_abs_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
                        "max_rel_diff": float(rel_diff.max().item()) if rel_diff.numel() else 0.0,
                        "mismatch_count": int(mismatch.sum().item()),
                        "scatter_max_abs_diff": (
                            float(scatter_abs_diff.max().item()) if scatter_abs_diff.numel() else 0.0
                        ),
                        "scatter_mean_abs_diff": (
                            float(scatter_abs_diff.mean().item()) if scatter_abs_diff.numel() else 0.0
                        ),
                        "worst_row": worst_row,
                        "row_samples": row_samples,
                    },
                }
                logger.warning("VIME_MOE_PROBE_JSON %s", json.dumps(report, separators=(",", ":"), sort_keys=True))
            except Exception:
                logger.exception("VIME_MOE_UNPERMUTE_PROBE failed to compare the final unpermute output")

            return output

        TokenDispatcherWithAll2AllV._dispatch_preprocess = _patched_dispatch_preprocess
        TokenDispatcherWithAll2AllV._dispatch_postprocess = _patched_dispatch_postprocess
        TokenDispatcherWithAll2AllV._combine_preprocess = _patched_combine_preprocess
        TokenDispatcherWithAll2AllV._combine_postprocess = _patched_combine_postprocess
        TokenDispatcherWithAll2AllV._vime_unpermute_probe_pending = False
        TokenDispatcherWithAll2AllV._vime_unpermute_probe_patched = True
        logger.info("Colocated Ascend A3 detected: installed MoE ALLTOALL unpermute probe")

    @staticmethod
    def _patch_one_worker(worker_cls: type) -> None:
        import inspect

        _orig_load_model = worker_cls.load_model
        _orig_start_weight_update = worker_cls.start_weight_update
        _orig_wake_up = worker_cls.wake_up
        has_dummy_kw = "load_dummy_weights" in inspect.signature(_orig_load_model).parameters

        if has_dummy_kw:

            def _patched_load_model(self, *, load_dummy_weights: bool = False, _orig=_orig_load_model) -> None:
                _orig(self, load_dummy_weights=load_dummy_weights)
                _VLLMHijack.patch_moe_weight_loader(self.model_runner.model)

        else:

            def _patched_load_model(self, _orig=_orig_load_model) -> None:
                _orig(self)
                _VLLMHijack.patch_moe_weight_loader(self.model_runner.model)

        def _patched_start_weight_update(
            self, is_checkpoint_format: bool = True, _orig=_orig_start_weight_update
        ) -> None:
            _VLLMHijack.patch_moe_weight_loader(self.model_runner.model)
            try:
                from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV

                if getattr(TokenDispatcherWithAll2AllV, "_vime_unpermute_probe_patched", False):
                    TokenDispatcherWithAll2AllV._vime_unpermute_probe_pending = True
            except ImportError:
                pass
            _orig(self, is_checkpoint_format=is_checkpoint_format)

        def _patched_wake_up(self, tags=None, _orig=_orig_wake_up) -> None:
            quant_config = self.vllm_config.quant_config
            if quant_config is not None:
                _orig(self, tags=tags)
                return

            # vllm-ascend transposes unquantized w13_weight/w2_weight in
            # wake_up(). Keep the native allocator and buffer restoration, but
            # skip that branch: layerwise reload owns the final runtime layout.
            self.vllm_config.quant_config = object()
            try:
                _orig(self, tags=tags)
            finally:
                self.vllm_config.quant_config = quant_config

        worker_cls.load_model = _patched_load_model  # type: ignore[attr-defined]
        worker_cls.start_weight_update = _patched_start_weight_update  # type: ignore[attr-defined]
        worker_cls.wake_up = _patched_wake_up  # type: ignore[attr-defined]

    @staticmethod
    def patch_moe_weight_loader(model: torch.nn.Module) -> None:
        inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
        if inner_model is None:
            return
        if not hasattr(inner_model, "layers"):
            inner_model = getattr(inner_model, "model", None)
            if inner_model is None or not hasattr(inner_model, "layers"):
                return

        for layer in inner_model.layers:
            mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
            if mlp is None:
                continue
            experts = getattr(mlp, "experts", None)
            if experts is None or not hasattr(experts, "weight_loader"):
                continue
            for name, param in mlp.named_parameters():
                if "w13_weight" in name or "w2_weight" in name:
                    if not hasattr(param, "weight_loader"):
                        param.weight_loader = experts.weight_loader  # type: ignore[attr-defined]

    @staticmethod
    def _patch_npu_rotary_emb() -> None:
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        if getattr(ApplyRotaryEmb, "_npu_rotary_patched", False):
            return

        def _npu_rotary_emb_init(
            self,
            enforce_enable: bool = False,
            is_neox_style: bool = True,
            enable_fp32_compute: bool = False,
        ) -> None:
            super(ApplyRotaryEmb, self).__init__(enforce_enable=enforce_enable)
            self.is_neox_style = is_neox_style
            self.enable_fp32_compute = enable_fp32_compute
            self.apply_rotary_emb_flash_attn = None

        ApplyRotaryEmb.__init__ = _npu_rotary_emb_init  # type: ignore[attr-defined]
        ApplyRotaryEmb._npu_rotary_patched = True


class vLLMColocateWorkerExtension:
    """vLLM ``--worker-extension-cls`` entry for colocated rollout workers."""

    def __new__(cls, **kwargs):
        if is_npu():
            _VLLMHijack._patch_a3_moe_alltoall_padding()
            _VLLMHijack._patch_a3_moe_unpermute_probe()
            _VLLMHijack._patch_npu_worker()
            _VLLMHijack._patch_npu_rotary_emb()
        return super().__new__(cls)


class vLLMWorkerExtension:
    """vLLM ``--worker-extension-cls`` entry for general bugfix."""

    def __new__(cls, **kwargs):
        if is_npu():
            _VLLMHijack._patch_npu_worker()
            _VLLMHijack._patch_npu_rotary_emb()
        return super().__new__(cls)
