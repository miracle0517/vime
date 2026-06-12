import copy
import logging
import os
import time
from argparse import Namespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from slime.utils.async_utils import run

logger = logging.getLogger(__name__)

TRAIN_PARTITION_PREFIX = "train_"
FIELD_MANIFEST_SEPARATOR = "_fields_"
SHARD_SIZE_SEPARATOR = "_n_"
ACTOR_TRAIN_TASK = "actor_train"
CRITIC_TRAIN_TASK = "critic_train"
CRITIC_VALUE_FIELDS = ["values"]
NON_SAMPLE_TRAIN_DATA_FIELDS = {
    "partition",
    "total_lengths",
    "raw_reward",
    "global_batch_sizes",
    "num_microbatches",
    "micro_batch_indices",
}
NON_TENSOR_TRAIN_DATA_FIELDS = {
    "metadata",
    "multimodal_train_inputs",
    "prompt",
    *NON_SAMPLE_TRAIN_DATA_FIELDS,
}

REQUIRED_TRAIN_DATA_FIELDS = [
    "tokens",
    "response_lengths",
    "loss_masks",
    "rewards",
]


def _import_transfer_queue():
    try:
        import transfer_queue as tq
    except ImportError as exc:
        raise ImportError(
            "--enable-vime-transfer-queue requires the external `transfer_queue` package. "
            "Install TransferQueue in the runtime environment before enabling this path."
        ) from exc
    return tq


def _import_tensordict():
    try:
        from tensordict import TensorDict
    except ImportError as exc:
        raise ImportError(
            "--enable-vime-transfer-queue requires `tensordict`, which is normally installed with TransferQueue."
        ) from exc
    return TensorDict


class TransferQueueBridge:
    """Facade for the optional TransferQueue rollout-to-training data path."""

    def __init__(self, args: Namespace, client=None):
        self.args = args
        self.client = client

    @classmethod
    def enabled(cls, args: Namespace) -> bool:
        return bool(getattr(args, "enable_vime_transfer_queue", False))

    @classmethod
    def initialize(cls, args: Namespace) -> None:
        """Create the TransferQueue controller/storage and attach config to args."""
        if not cls.enabled(args):
            args.tq_config = None
            return

        os.environ.update(cls.env_vars(args))
        tq = _import_transfer_queue()

        total_storage_size = args.rollout_batch_size * args.n_samples_per_prompt * (args.max_staleness + 1)
        sampler = cls._build_sampler(args, tq)
        tq_config = OmegaConf.create(
            {
                "controller": {
                    "sampler": sampler,
                    "polling_mode": args.polling_mode,
                },
                "backend": {
                    "SimpleStorage": {
                        "total_storage_size": total_storage_size,
                        "num_data_storage_units": args.num_data_storage_units,
                    },
                },
            },
            flags={"allow_objects": True},
        )
        args.tq_config = tq.init(conf=tq_config) or tq_config
        logger.info(
            "Initialized TransferQueue: total_storage_size=%s, num_data_storage_units=%s, max_staleness=%s",
            total_storage_size,
            args.num_data_storage_units,
            args.max_staleness,
        )
        if getattr(args, "use_critic", False) and not cls.critic_values_via_transfer_queue(args):
            logger.warning(
                "TransferQueue critic values write-back is disabled because context_parallel_size=%s; "
                "critic values will use the existing Ray ObjectRef path.",
                getattr(args, "context_parallel_size", 1),
            )

    @classmethod
    def env_vars(cls, args: Namespace) -> dict[str, str]:
        if not cls.enabled(args):
            return {}
        return {
            "TQ_PRE_ALLOC_SAMPLE_NUM": str(args.rollout_batch_size * args.n_samples_per_prompt),
            "TQ_ZERO_COPY_SERIALIZATION": "true",
        }

    @classmethod
    def connect(cls, args: Namespace) -> "TransferQueueBridge":
        """Connect a Ray actor process to the already-created TransferQueue."""
        if not cls.enabled(args):
            return cls(args)
        if getattr(args, "tq_config", None) is None:
            raise ValueError(
                "args.tq_config is missing. TransferQueueBridge.initialize(args) must run before actors start."
            )
        tq = _import_transfer_queue()
        tq.init(args.tq_config)
        return cls(args, tq.get_client())

    def close(self) -> None:
        if not self.enabled(self.args):
            return
        tq = _import_transfer_queue()
        tq.close()

    @classmethod
    def _build_sampler(cls, args: Namespace, tq):
        if getattr(args, "balance_data", False):
            dp_size = cls.data_parallel_size(args)
            logger.info("Using TransferQueue SeqlenBalancedSampler with dp_size=%s", dp_size)
            return tq.SeqlenBalancedSampler(n_samples_per_prompt=args.n_samples_per_prompt, dp_size=dp_size)
        return tq.GRPOGroupNSampler(n_samples_per_prompt=args.n_samples_per_prompt)

    @classmethod
    def data_parallel_size(cls, args: Namespace) -> int:
        """Return the Megatron DP size used by TQ before Megatron is initialized."""
        world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        model_parallel_size = (
            int(getattr(args, "tensor_model_parallel_size", 1))
            * int(getattr(args, "pipeline_model_parallel_size", 1))
            * int(getattr(args, "context_parallel_size", 1))
        )
        if model_parallel_size <= 0:
            raise ValueError(f"Invalid model parallel size for TransferQueue: {model_parallel_size}")
        if world_size % model_parallel_size != 0:
            raise ValueError(
                "Actor world size must be divisible by tensor*pipeline*context parallel size when using TransferQueue: "
                f"world_size={world_size}, model_parallel_size={model_parallel_size}"
            )
        return world_size // model_parallel_size

    @staticmethod
    def add_total_lengths(train_data: dict[str, Any]) -> dict[str, Any]:
        train_data = dict(train_data)
        train_data["total_lengths"] = [len(tokens) for tokens in train_data["tokens"]]
        return train_data

    @classmethod
    def normalize_train_data(cls, train_data: dict[str, Any]) -> dict[str, Any]:
        """Make rollout train_data match the fields actor/critic request from TQ."""
        missing = [field for field in REQUIRED_TRAIN_DATA_FIELDS if field not in train_data]
        if missing:
            raise ValueError(f"TransferQueue rollout data is missing required fields: {missing}")

        train_data = dict(train_data)
        batch_size = len(train_data["tokens"])

        if "total_lengths" not in train_data:
            train_data = cls.add_total_lengths(train_data)

        if "raw_reward" not in train_data:
            train_data["raw_reward"] = list(train_data["rewards"])
        if "truncated" not in train_data:
            train_data["truncated"] = [0] * batch_size
        if "sample_indices" not in train_data:
            train_data["sample_indices"] = list(range(batch_size))

        return train_data

    @classmethod
    def dict_to_tensordict(
        cls,
        data: dict[str, list],
        batch_size: int | torch.Size | None = None,
        device=None,
    ):
        """Convert slime rollout data into the TensorDict format expected by TransferQueue."""
        TensorDict = _import_tensordict()
        if not data:
            return TensorDict({}, batch_size=0 if batch_size is None else batch_size, device=device)
        if batch_size is None:
            batch_size_int = len(next(iter(data.values())))
        else:
            batch_size_int = int(batch_size[0]) if isinstance(batch_size, torch.Size) else int(batch_size)
        data = cls._expand_non_sample_fields(data, batch_size_int)

        def nesting_depth(value):
            if isinstance(value, list) and value:
                return 1 + nesting_depth(value[0])
            return 0

        def scalar_dtype(sample):
            if isinstance(sample, bool):
                return torch.bool
            if isinstance(sample, float):
                return torch.float32
            return None

        def tensor_1d(value):
            dtype = scalar_dtype(value[0]) if value else None
            return torch.tensor(value, dtype=dtype, device=device)

        def tensor_2d(value):
            dtype = scalar_dtype(value[0][0]) if value and value[0] else None
            tensors = [torch.tensor(seq, dtype=dtype, device=device) for seq in value]
            return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

        result = {}
        for key, value in data.items():
            if not isinstance(value, list):
                raise TypeError(f"TransferQueue field '{key}' must be a list, got {type(value)}")

            if key == "rollout_routed_experts":
                tensors = []
                for item in value:
                    arr = item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
                    arr = np.ascontiguousarray(arr.reshape(arr.shape[0], -1))
                    tensors.append(torch.from_numpy(arr).to(torch.int32))
                result[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
                continue

            if key in NON_TENSOR_TRAIN_DATA_FIELDS:
                result[key] = value
                continue

            if value and isinstance(value[0], torch.Tensor):
                tensors = []
                for item in value:
                    tensor = item.detach()
                    if tensor.device.type != "cpu":
                        tensor = tensor.cpu()
                    if device is not None:
                        tensor = tensor.to(device)
                    tensors.append(tensor)
                if tensors[0].ndim == 0:
                    result[key] = torch.stack(tensors)
                else:
                    result[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
                continue

            depth = nesting_depth(value)
            if depth == 0:
                result[key] = torch.empty(0, device=device)
            elif depth == 1:
                result[key] = tensor_1d(value)
            elif depth == 2:
                result[key] = tensor_2d(value)
            else:
                raise ValueError(f"TransferQueue field '{key}' has unsupported nesting depth {depth}; max depth is 2.")

        return TensorDict(result, batch_size=batch_size, device=device)

    @staticmethod
    def _expand_non_sample_fields(data: dict[str, Any], batch_size: int) -> dict[str, Any]:
        data = dict(data)
        for key in NON_SAMPLE_TRAIN_DATA_FIELDS:
            if key in data:
                data[key] = [copy.deepcopy(data[key]) for _ in range(batch_size)]
        return data

    def transfer_rollout_data(self, rollout_id: int, train_data: dict[str, Any]) -> None:
        """Write one rollout partition to TransferQueue."""
        self.transfer_rollout_data_shard(rollout_id, train_data, dp_rank=None, wait_for_staleness=True)

    def transfer_rollout_data_shards(self, rollout_id: int, train_data_by_dp: list[dict[str, Any]]) -> None:
        """Write per-DP rollout shards to TransferQueue."""
        self.wait_for_staleness()
        for dp_rank, train_data in enumerate(train_data_by_dp):
            self.transfer_rollout_data_shard(rollout_id, train_data, dp_rank=dp_rank, wait_for_staleness=False)

    def transfer_rollout_data_shard(
        self,
        rollout_id: int,
        train_data: dict[str, Any],
        *,
        dp_rank: int | None,
        wait_for_staleness: bool,
    ) -> None:
        """Write one rollout partition or one already-DP-sharded partition."""
        client = self._require_client()
        if wait_for_staleness:
            self.wait_for_staleness()
        train_data = self.normalize_train_data(train_data)
        batch_size = len(train_data["tokens"])
        rollout_batch = self.dict_to_tensordict(train_data, batch_size=batch_size)
        partition_id = self.partition_id(
            rollout_id,
            dp_rank=dp_rank,
            shard_size=batch_size if dp_rank is not None else None,
            fields=train_data.keys() if dp_rank is not None else None,
        )
        metadata = run(client.async_put(data=rollout_batch, partition_id=partition_id))
        self._set_total_length_custom_meta(metadata, self._sample_total_lengths(train_data))
        logger.info(
            "Transferred rollout_id=%s to TransferQueue partition=%s with %s samples; fields=%s",
            rollout_id,
            partition_id,
            batch_size,
            sorted(train_data.keys()),
        )

    def put_data(
        self,
        rollout_id: int,
        data: dict[str, Any],
        *,
        data_fields: list[str],
        batch_meta=None,
    ) -> None:
        """Write derived fields back to an existing TransferQueue batch."""
        if not self.enabled(self.args) or self.client is None:
            return

        missing = [field for field in data_fields if field not in data]
        if missing:
            raise ValueError(f"TransferQueue write-back data is missing fields: {missing}")

        payload = {field: data[field] for field in data_fields}
        batch_size = len(next(iter(payload.values()))) if payload else 0
        rollout_batch = self.dict_to_tensordict(payload, batch_size=batch_size)
        if batch_meta is None:
            raise ValueError("TransferQueue write-back requires batch_meta from the matching get_meta/get_data call.")

        run(self.client.async_put(data=rollout_batch, metadata=batch_meta))
        logger.info(
            "Wrote TransferQueue fields: partition=%s fields=%s samples=%s",
            self.partition_id(rollout_id),
            data_fields,
            batch_size,
        )

    def _set_total_length_custom_meta(self, metadata, total_lengths: list[int]) -> None:
        if metadata is None or getattr(metadata, "size", 0) == 0 or not hasattr(metadata, "update_custom_meta"):
            return
        metadata.update_custom_meta([{"total_lengths": int(length)} for length in total_lengths])
        run(self._require_client().async_set_custom_meta(metadata))

    @staticmethod
    def _sample_total_lengths(train_data: dict[str, Any]) -> list[int]:
        total_lengths = train_data.get("total_lengths")
        batch_size = len(train_data["tokens"])
        if total_lengths is None:
            return [len(tokens) for tokens in train_data["tokens"]]
        if len(total_lengths) == batch_size:
            return total_lengths
        partition = train_data.get("partition")
        if partition is not None:
            return [total_lengths[i] for i in partition]
        return [len(tokens) for tokens in train_data["tokens"]]

    def wait_for_staleness(self) -> None:
        """Apply simple partition-count backpressure before writing a new rollout."""
        client = self._require_client()
        max_staleness = int(getattr(self.args, "max_staleness", 0))
        poll_interval = float(getattr(self.args, "transfer_queue_staleness_poll_interval", 1.0))
        while True:
            partitions = run(client.async_get_partition_list())
            train_rollouts = {
                rollout_key
                for partition in partitions
                if (rollout_key := self._partition_rollout_key(partition)) is not None
            }
            if len(train_rollouts) <= max_staleness:
                return
            logger.info(
                "TransferQueue staleness backpressure: %s train rollouts > max_staleness=%s; waiting %.2fs",
                len(train_rollouts),
                max_staleness,
                poll_interval,
            )
            time.sleep(poll_interval)

    def clear_partition(self, rollout_id: int) -> None:
        if not self.enabled(self.args) or self.client is None:
            return
        base_partition_id = self.partition_id(rollout_id)
        partitions = run(self.client.async_get_partition_list())
        target_partitions = [
            partition
            for partition in partitions
            if str(partition) == base_partition_id or str(partition).startswith(f"{base_partition_id}_dp_")
        ]
        if not target_partitions:
            target_partitions = [base_partition_id]
        for partition in target_partitions:
            run(self.client.async_clear_partition(partition_id=partition))
        logger.info("Cleared TransferQueue partitions %s", target_partitions)

    @staticmethod
    def partition_id(
        rollout_id: int,
        dp_rank: int | None = None,
        shard_size: int | None = None,
        fields=None,
    ) -> str:
        partition_id = f"{TRAIN_PARTITION_PREFIX}{rollout_id}"
        if dp_rank is not None:
            partition_id = f"{partition_id}_dp_{dp_rank}"
        if shard_size is not None:
            partition_id = f"{partition_id}{SHARD_SIZE_SEPARATOR}{shard_size}"
        if fields is not None:
            partition_id = f"{partition_id}{FIELD_MANIFEST_SEPARATOR}{'.'.join(sorted(fields))}"
        return partition_id

    @staticmethod
    def _partition_rollout_key(partition_id) -> str | None:
        partition_id = str(partition_id)
        if not partition_id.startswith(TRAIN_PARTITION_PREFIX):
            return None
        suffix = partition_id[len(TRAIN_PARTITION_PREFIX) :]
        return suffix.split("_dp_", 1)[0]

    @classmethod
    def default_train_data_fields(cls, args: Namespace) -> list[str]:
        fields = [
            "tokens",
            "total_lengths",
            "response_lengths",
            "loss_masks",
            "rewards",
            "raw_reward",
            "truncated",
            "sample_indices",
            "multimodal_train_inputs",
            "partition",
            "rollout_ids",
            "rollout_mask_sums",
            "global_batch_sizes",
            "num_microbatches",
            "micro_batch_indices",
            "round_number",
            "rollout_log_probs",
            "rollout_routed_experts",
            "teacher_log_probs",
        ]
        for field in getattr(args, "transfer_queue_extra_data_fields", []) or []:
            if field not in fields:
                fields.append(field)
        return fields

    @classmethod
    def critic_values_via_transfer_queue(cls, args: Namespace) -> bool:
        return (
            cls.enabled(args)
            and getattr(args, "use_critic", False)
            and int(getattr(args, "context_parallel_size", 1)) == 1
        )

    @classmethod
    def actor_train_data_fields(cls, args: Namespace) -> list[str]:
        fields = cls.default_train_data_fields(args)
        if cls.critic_values_via_transfer_queue(args):
            for field in CRITIC_VALUE_FIELDS:
                if field not in fields:
                    fields.append(field)
        return fields

    def get_data(
        self,
        rollout_id: int,
        *,
        task_name: str,
        data_fields: list[str] | None = None,
    ):
        """Fetch this DP rank's rollout data from TransferQueue and broadcast it to model-parallel ranks."""
        from megatron.core import mpu

        client = self._require_client()
        data_fields = data_fields or self.default_train_data_fields(self.args)
        total_batch_size = self.args.rollout_batch_size * self.args.n_samples_per_prompt
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        partition_id, batch_size, available_fields = self._resolve_read_partition(
            rollout_id, dp_rank, total_batch_size, dp_size
        )
        if available_fields is not None:
            deferred_fields = set(CRITIC_VALUE_FIELDS) if self.critic_values_via_transfer_queue(self.args) else set()
            requested_fields = data_fields
            data_fields = [field for field in data_fields if field in available_fields or field in deferred_fields]
            skipped_fields = sorted(set(requested_fields) - set(data_fields) - deferred_fields)
            if skipped_fields:
                logger.debug(
                    "Skipped TransferQueue fields absent from shard manifest: partition=%s fields=%s",
                    partition_id,
                    skipped_fields,
                )
        sampling_config = {
            "dp_rank": dp_rank,
            "task_name": task_name,
            "batch_index": 0,
            "partition_id": partition_id,
        }

        should_fetch = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
        )
        payload = [None, None]
        if should_fetch:
            batch_meta = client.get_meta(
                data_fields=data_fields,
                batch_size=batch_size,
                partition_id=partition_id,
                sampling_config=sampling_config,
                task_name=task_name,
            )
            if batch_meta.size != 0:
                payload = [client.get_data(batch_meta), batch_meta]
                logger.info(
                    "Fetched TransferQueue data: partition=%s task=%s dp_rank=%s batch_size=%s fields=%s",
                    partition_id,
                    task_name,
                    sampling_config["dp_rank"],
                    batch_meta.size,
                    data_fields,
                )

        device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device(
            "cpu"
        )
        self._broadcast_payload(payload, device)
        rollout_data, batch_meta = payload
        if rollout_data is None:
            return None, None
        return self.tensordict_to_rollout_data(rollout_data), batch_meta

    def _resolve_read_partition(
        self,
        rollout_id: int,
        dp_rank: int,
        total_batch_size: int,
        dp_size: int,
    ) -> tuple[str, int, set[str] | None]:
        base_partition_id = self.partition_id(rollout_id)
        shard_prefix = self.partition_id(rollout_id, dp_rank=dp_rank) + SHARD_SIZE_SEPARATOR
        partitions = [str(partition) for partition in run(self._require_client().async_get_partition_list())]
        shard_partitions = [partition for partition in partitions if partition.startswith(shard_prefix)]
        rollout_shard_partitions = [
            partition for partition in partitions if partition.startswith(f"{base_partition_id}_dp_")
        ]
        if shard_partitions:
            if len(shard_partitions) != 1:
                raise ValueError(
                    f"Expected one TransferQueue shard partition for rollout_id={rollout_id}, dp_rank={dp_rank}, "
                    f"got {shard_partitions}"
                )
            partition_id = shard_partitions[0]
            shard_suffix = partition_id[len(shard_prefix) :]
            batch_size_text, _, field_manifest = shard_suffix.partition(FIELD_MANIFEST_SEPARATOR)
            try:
                batch_size = int(batch_size_text)
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid TransferQueue shard partition id: {partition_id}") from exc
            available_fields = set(field_manifest.split(".")) if field_manifest else None
            return partition_id, batch_size, available_fields

        if rollout_shard_partitions:
            raise ValueError(
                "TransferQueue shard partition is missing for this DP rank: "
                f"rollout_id={rollout_id}, dp_rank={dp_rank}, existing_shards={rollout_shard_partitions}"
            )

        if total_batch_size % dp_size != 0:
            raise ValueError(
                "TransferQueue requires rollout_batch_size*n_samples_per_prompt to be divisible by DP size when "
                "reading a non-sharded partition: "
                f"total_batch_size={total_batch_size}, dp_size={dp_size}"
            )
        return base_partition_id, total_batch_size // dp_size, None

    @staticmethod
    def _broadcast_payload(payload: list[Any], device: torch.device) -> None:
        from megatron.core import mpu

        def bcast(group):
            src = dist.get_global_rank(group, 0)
            dist.broadcast_object_list(payload, src=src, group=group, device=device)

        if mpu.get_context_parallel_world_size() > 1:
            bcast(mpu.get_context_parallel_group())
        bcast(mpu.get_tensor_model_parallel_group())
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            bcast(mpu.get_pipeline_model_parallel_group())

    @classmethod
    def tensordict_to_rollout_data(cls, data) -> dict[str, Any]:
        """Turn TransferQueue TensorDict payload back into slime's list-based RolloutBatch."""
        try:
            from tensordict import TensorDict
            from tensordict.tensorclass import NonTensorData
        except ImportError:
            TensorDict = ()
            NonTensorData = ()

        if not isinstance(data, TensorDict):
            return data

        rollout_data = {}
        for key, value in data.items():
            if key in NON_SAMPLE_TRAIN_DATA_FIELDS:
                items = cls._unwrap_non_tensor_stack(value, NonTensorData)
                rollout_data[key] = items[0] if items else []
            elif key in {"metadata", "multimodal_train_inputs", "prompt"}:
                rollout_data[key] = cls._unwrap_non_tensor_stack(value, NonTensorData)
            elif (
                "lengths" in key
                or key
                in {
                    "rewards",
                    "raw_reward",
                    "truncated",
                    "sample_indices",
                    "rollout_ids",
                    "rollout_mask_sums",
                    "round_number",
                }
            ):
                rollout_data[key] = value.tolist() if isinstance(value, torch.Tensor) else list(value)
            elif isinstance(value, torch.Tensor):
                rollout_data[key] = [item for item in value]
            else:
                rollout_data[key] = cls._unwrap_non_tensor_stack(value, NonTensorData)
        return rollout_data

    @staticmethod
    def _unwrap_non_tensor_stack(value, non_tensor_data_cls):
        output = []
        for item in list(value):
            raw = item.data if non_tensor_data_cls and isinstance(item, non_tensor_data_cls) else item
            if hasattr(raw, "items"):
                raw = dict(raw.items())
            output.append(raw)
        return output

    def _require_client(self):
        if self.client is None:
            raise ValueError("TransferQueue client is not connected.")
        return self.client
