import logging
import os
import time
import uuid
from argparse import Namespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from slime.utils.async_utils import run

logger = logging.getLogger(__name__)

TRAIN_PARTITION_PREFIX = "train_"
ACTOR_TRAIN_TASK = "actor_train"
CRITIC_TRAIN_TASK = "critic_train"
CRITIC_VALUE_FIELDS = ["values"]
SHARD_SIZE_FIELD = "tq_shard_size"
SHARD_METADATA_ACTOR_NAME_ATTR = "transfer_queue_shard_metadata_actor_name"
SHARD_METADATA_ACTOR_NAME_PREFIX = "vime_transfer_queue_shard_metadata"

_SHARD_METADATA_ACTOR_CLS = None

REQUIRED_TRAIN_DATA_FIELDS = [
    "tokens",
    "response_lengths",
    "loss_masks",
    "rewards",
]


class _TransferQueueShardMetadataStore:
    def __init__(self):
        self._metadata = {}

    def set_many(self, items):
        for partition_id, metadata in items:
            self._metadata[str(partition_id)] = dict(metadata)

    def get(self, partition_id):
        metadata = self._metadata.get(str(partition_id))
        return dict(metadata) if metadata is not None else None

    def clear_many(self, partition_ids):
        for partition_id in partition_ids:
            self._metadata.pop(str(partition_id), None)


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
    def _ensure_shard_metadata_actor_name(cls, args: Namespace) -> str:
        name = getattr(args, SHARD_METADATA_ACTOR_NAME_ATTR, None)
        if not name:
            name = f"{SHARD_METADATA_ACTOR_NAME_PREFIX}_{uuid.uuid4().hex}"
            setattr(args, SHARD_METADATA_ACTOR_NAME_ATTR, name)
        return name

    @classmethod
    def initialize(cls, args: Namespace) -> None:
        """Create the TransferQueue controller/storage and attach config to args."""
        if not cls.enabled(args):
            args.tq_config = None
            return

        cls._ensure_shard_metadata_actor_name(args)
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
        cls._get_shard_metadata_actor(args, create=True)
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
    def _get_shard_metadata_actor(cls, args: Namespace, *, create: bool):
        try:
            import ray
        except ImportError:
            return None

        if not ray.is_initialized():
            return None

        name = cls._ensure_shard_metadata_actor_name(args)
        try:
            return ray.get_actor(name)
        except (RuntimeError, ValueError):
            if not create:
                return None

        global _SHARD_METADATA_ACTOR_CLS
        if _SHARD_METADATA_ACTOR_CLS is None:
            _SHARD_METADATA_ACTOR_CLS = ray.remote(num_cpus=0)(_TransferQueueShardMetadataStore)
        return _SHARD_METADATA_ACTOR_CLS.options(name=name).remote()

    def _set_shard_metadata_many(self, items: list[tuple[str, dict[str, Any]]]) -> None:
        if not items:
            return
        actor = self._get_shard_metadata_actor(self.args, create=True)
        if actor is None:
            raise ValueError("Ray must be initialized before writing TransferQueue shard metadata.")

        import ray

        ray.get(actor.set_many.remote(items))

    def _get_shard_metadata(self, partition_id: str) -> dict[str, Any] | None:
        actor = self._get_shard_metadata_actor(self.args, create=False)
        if actor is None:
            return None

        import ray

        return ray.get(actor.get.remote(partition_id))

    def _clear_shard_metadata(self, partition_ids: list[Any]) -> None:
        actor = self._get_shard_metadata_actor(self.args, create=False)
        if actor is None:
            return

        import ray

        ray.get(actor.clear_many.remote([str(partition_id) for partition_id in partition_ids]))

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

        train_data = cls.add_total_lengths(train_data)
        batch_size = len(train_data["tokens"])

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

            if key in {"metadata", "multimodal_train_inputs", "prompt"}:
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

    def transfer_rollout_data(self, rollout_id: int, train_data: dict[str, Any]) -> None:
        """Write one rollout partition to TransferQueue."""
        self.transfer_rollout_data_shard(rollout_id, train_data, dp_rank=None, wait_for_staleness=True)

    def transfer_rollout_data_shards(
        self,
        rollout_id: int,
        train_data_by_dp: list[dict[str, Any]],
        shard_metadata_by_dp: list[dict[str, Any]],
    ) -> None:
        """Write already-DP-sharded rollout data to TransferQueue."""
        if len(train_data_by_dp) != len(shard_metadata_by_dp):
            raise ValueError(
                "TransferQueue shard data and metadata counts must match: "
                f"data={len(train_data_by_dp)}, metadata={len(shard_metadata_by_dp)}"
            )
        self.wait_for_staleness()
        metadata_items = []
        for dp_rank, (train_data, shard_metadata) in enumerate(
            zip(train_data_by_dp, shard_metadata_by_dp, strict=True)
        ):
            partition_id = self.transfer_rollout_data_shard(
                rollout_id,
                train_data,
                dp_rank=dp_rank,
                wait_for_staleness=False,
            )
            shard_metadata = dict(shard_metadata)
            shard_metadata.setdefault(SHARD_SIZE_FIELD, len(train_data["tokens"]))
            self._validate_shard_metadata(partition_id, shard_metadata)
            metadata_items.append((partition_id, shard_metadata))
        self._set_shard_metadata_many(metadata_items)

    def transfer_rollout_data_shard(
        self,
        rollout_id: int,
        train_data: dict[str, Any],
        *,
        dp_rank: int | None,
        wait_for_staleness: bool,
    ) -> str:
        """Write one rollout partition or one DP-sharded partition to TransferQueue."""
        client = self._require_client()
        if wait_for_staleness:
            self.wait_for_staleness()
        train_data = self.normalize_train_data(train_data)
        rollout_batch = self.dict_to_tensordict(train_data, batch_size=len(train_data["tokens"]))
        partition_id = self.partition_id(rollout_id, dp_rank=dp_rank)
        metadata = run(client.async_put(data=rollout_batch, partition_id=partition_id))
        self._set_total_length_custom_meta(metadata, train_data["total_lengths"])
        logger.info(
            "Transferred rollout_id=%s to TransferQueue partition=%s with %s samples; fields=%s",
            rollout_id,
            partition_id,
            len(train_data["tokens"]),
            sorted(train_data.keys()),
        )
        return partition_id

    @classmethod
    def _validate_shard_metadata(cls, partition_id: str, shard_metadata: dict[str, Any]) -> None:
        if SHARD_SIZE_FIELD not in shard_metadata:
            raise ValueError(
                f"TransferQueue shard metadata for partition={partition_id} is missing field: {SHARD_SIZE_FIELD}"
            )
        shard_size = int(shard_metadata[SHARD_SIZE_FIELD])
        if shard_size <= 0:
            raise ValueError(f"TransferQueue shard size must be positive for partition={partition_id}, got {shard_size}.")
        shard_metadata[SHARD_SIZE_FIELD] = shard_size

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
        self._clear_shard_metadata(target_partitions)
        logger.info("Cleared TransferQueue partitions %s", target_partitions)

    @staticmethod
    def partition_id(rollout_id: int, dp_rank: int | None = None) -> str:
        partition_id = f"{TRAIN_PARTITION_PREFIX}{rollout_id}"
        if dp_rank is not None:
            partition_id = f"{partition_id}_dp_{dp_rank}"
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
        ]
        if getattr(args, "use_rollout_logprobs", False):
            fields.append("rollout_log_probs")
        if getattr(args, "use_rollout_routing_replay", False):
            fields.append("rollout_routed_experts")
        if getattr(args, "multimodal_keys", None) is not None:
            fields.append("multimodal_train_inputs")
        if getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang":
            fields.append("teacher_log_probs")
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
        batch_size: int | None = None,
    ):
        """Fetch this DP rank's rollout data from TransferQueue and broadcast it to model-parallel ranks."""
        from megatron.core import mpu

        client = self._require_client()
        data_fields = data_fields or self.default_train_data_fields(self.args)
        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        partition_id = self.partition_id(rollout_id, dp_rank=dp_rank)
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
        payload = [None, None, None, None]
        if should_fetch:
            try:
                shard_metadata = self._get_shard_metadata(partition_id)
                if batch_size is None:
                    batch_size = self._get_batch_size_from_shard_metadata(partition_id, shard_metadata)
                batch_meta = client.get_meta(
                    data_fields=data_fields,
                    batch_size=batch_size,
                    partition_id=partition_id,
                    sampling_config=sampling_config,
                    task_name=task_name,
                )
                if batch_meta.size != 0:
                    payload = [client.get_data(batch_meta), batch_meta, shard_metadata, None]
                    logger.info(
                        "Fetched TransferQueue data: partition=%s task=%s dp_rank=%s batch_size=%s fields=%s",
                        partition_id,
                        task_name,
                        sampling_config["dp_rank"],
                        batch_meta.size,
                        data_fields,
                    )
            except Exception as exc:
                payload[3] = f"{type(exc).__name__}: {exc}"

        device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device(
            "cpu"
        )
        self._broadcast_payload(payload, device)
        rollout_data, batch_meta, shard_metadata, error = payload
        if error is not None:
            raise ValueError(f"TransferQueue fetch failed for partition={partition_id}: {error}")
        if rollout_data is None:
            return None, None
        rollout_data = self.tensordict_to_rollout_data(rollout_data)
        self._apply_shard_metadata(rollout_data, shard_metadata)
        return rollout_data, batch_meta

    @classmethod
    def _get_batch_size_from_shard_metadata(cls, partition_id: str, shard_metadata: dict[str, Any] | None) -> int:
        if shard_metadata is None:
            raise ValueError(
                f"TransferQueue shard metadata is missing for partition={partition_id}. "
                "Write rollout data with transfer_rollout_data_shards before training."
            )
        cls._validate_shard_metadata(partition_id, shard_metadata)
        return shard_metadata[SHARD_SIZE_FIELD]

    @staticmethod
    def _apply_shard_metadata(rollout_data: dict[str, Any], shard_metadata: dict[str, Any] | None) -> None:
        if shard_metadata is None:
            return
        for field, value in shard_metadata.items():
            if field != SHARD_SIZE_FIELD:
                rollout_data[field] = value

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
            if key in {"metadata", "multimodal_train_inputs", "prompt"}:
                rollout_data[key] = cls._unwrap_non_tensor_stack(value, NonTensorData)
            elif (
                "lengths" in key
                or key
                in {
                    "rewards",
                    "raw_reward",
                    "truncated",
                    "sample_indices",
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
