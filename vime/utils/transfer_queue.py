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
ACTOR_TRAIN_TASK = "actor_train"
CRITIC_TRAIN_TASK = "critic_train"
CRITIC_VALUE_FIELDS = ["values"]

REQUIRED_TRAIN_DATA_FIELDS = [
    "tokens",
    "response_lengths",
    "loss_masks",
    "rewards",
]


def transfer_queue_enabled(args: Namespace) -> bool:
    return bool(getattr(args, "use_transfer_queue", False))


def _import_transfer_queue():
    try:
        import transfer_queue as tq
    except ImportError as exc:
        raise ImportError(
            "--use-transfer-queue requires the external `transfer_queue` package. "
            "Install TransferQueue in the runtime environment before enabling this path."
        ) from exc
    return tq


def _import_tensordict():
    try:
        from tensordict import TensorDict
    except ImportError as exc:
        raise ImportError(
            "--use-transfer-queue requires `tensordict`, which is normally installed with TransferQueue."
        ) from exc
    return TensorDict


def initialize_transfer_queue(args: Namespace) -> None:
    """Create the TransferQueue controller/storage and attach config to args."""
    if not transfer_queue_enabled(args):
        args.tq_config = None
        return

    os.environ.update(transfer_queue_env_vars(args))
    tq = _import_transfer_queue()

    total_storage_size = args.rollout_batch_size * args.n_samples_per_prompt * (args.max_staleness + 1)
    sampler = _build_sampler(args, tq)
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
    if getattr(args, "use_critic", False) and not critic_values_via_transfer_queue(args):
        logger.warning(
            "TransferQueue critic values write-back is disabled because context_parallel_size=%s; "
            "critic values will use the existing Ray ObjectRef path.",
            getattr(args, "context_parallel_size", 1),
        )


def transfer_queue_env_vars(args: Namespace) -> dict[str, str]:
    if not transfer_queue_enabled(args):
        return {}
    return {
        "TQ_PRE_ALLOC_SAMPLE_NUM": str(args.rollout_batch_size * args.n_samples_per_prompt),
        "TQ_ZERO_COPY_SERIALIZATION": "true",
    }


def connect_transfer_queue(args: Namespace):
    """Connect a Ray actor process to the already-created TransferQueue."""
    if not transfer_queue_enabled(args):
        return None
    if getattr(args, "tq_config", None) is None:
        raise ValueError("args.tq_config is missing. initialize_transfer_queue(args) must run before actors start.")
    tq = _import_transfer_queue()
    tq.init(args.tq_config)
    return tq.get_client()


def close_transfer_queue(args: Namespace) -> None:
    if not transfer_queue_enabled(args):
        return
    tq = _import_transfer_queue()
    tq.close()


def _build_sampler(args: Namespace, tq):
    if getattr(args, "balance_data", False):
        dp_size = transfer_queue_data_parallel_size(args)
        logger.info("Using TransferQueue SeqlenBalancedSampler with dp_size=%s", dp_size)
        return tq.SeqlenBalancedSampler(n_samples_per_prompt=args.n_samples_per_prompt, dp_size=dp_size)
    return tq.GRPOGroupNSampler(n_samples_per_prompt=args.n_samples_per_prompt)


def transfer_queue_data_parallel_size(args: Namespace) -> int:
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


def add_total_lengths(train_data: dict[str, Any]) -> dict[str, Any]:
    train_data = dict(train_data)
    train_data["total_lengths"] = [len(tokens) for tokens in train_data["tokens"]]
    return train_data


def normalize_train_data_for_transfer_queue(train_data: dict[str, Any]) -> dict[str, Any]:
    """Make rollout train_data match the fields actor/critic request from TQ."""
    missing = [field for field in REQUIRED_TRAIN_DATA_FIELDS if field not in train_data]
    if missing:
        raise ValueError(f"TransferQueue rollout data is missing required fields: {missing}")

    train_data = add_total_lengths(train_data)
    batch_size = len(train_data["tokens"])

    if "raw_reward" not in train_data:
        train_data["raw_reward"] = list(train_data["rewards"])
    if "truncated" not in train_data:
        train_data["truncated"] = [0] * batch_size
    if "sample_indices" not in train_data:
        train_data["sample_indices"] = list(range(batch_size))

    return train_data


def dict_to_tensordict(data: dict[str, list], batch_size: int | torch.Size | None = None, device=None):
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


def transfer_rollout_data(args: Namespace, client, rollout_id: int, train_data: dict[str, Any]) -> None:
    """Write one rollout partition to TransferQueue."""
    wait_for_staleness(args, client)
    train_data = normalize_train_data_for_transfer_queue(train_data)
    rollout_batch = dict_to_tensordict(train_data, batch_size=len(train_data["tokens"]))
    metadata = run(client.async_put(data=rollout_batch, partition_id=partition_id(rollout_id)))
    _set_total_length_custom_meta(client, metadata, train_data["total_lengths"])
    logger.info(
        "Transferred rollout_id=%s to TransferQueue partition=%s with %s samples; fields=%s",
        rollout_id,
        partition_id(rollout_id),
        len(train_data["tokens"]),
        sorted(train_data.keys()),
    )


def put_data_to_transfer_queue(
    args: Namespace,
    client,
    rollout_id: int,
    data: dict[str, Any],
    *,
    data_fields: list[str],
    batch_meta=None,
) -> None:
    """Write derived fields back to an existing TransferQueue batch."""
    if not transfer_queue_enabled(args) or client is None:
        return

    missing = [field for field in data_fields if field not in data]
    if missing:
        raise ValueError(f"TransferQueue write-back data is missing fields: {missing}")

    payload = {field: data[field] for field in data_fields}
    batch_size = len(next(iter(payload.values()))) if payload else 0
    rollout_batch = dict_to_tensordict(payload, batch_size=batch_size)
    if batch_meta is None:
        raise ValueError("TransferQueue write-back requires batch_meta from the matching get_meta/get_data call.")

    run(client.async_put(data=rollout_batch, metadata=batch_meta))
    logger.info(
        "Wrote TransferQueue fields: partition=%s fields=%s samples=%s",
        partition_id(rollout_id),
        data_fields,
        batch_size,
    )


def _set_total_length_custom_meta(client, metadata, total_lengths: list[int]) -> None:
    if metadata is None or getattr(metadata, "size", 0) == 0 or not hasattr(metadata, "update_custom_meta"):
        return
    metadata.update_custom_meta([{"total_lengths": int(length)} for length in total_lengths])
    run(client.async_set_custom_meta(metadata))


def wait_for_staleness(args: Namespace, client) -> None:
    """Apply simple partition-count backpressure before writing a new rollout."""
    max_staleness = int(getattr(args, "max_staleness", 0))
    poll_interval = float(getattr(args, "transfer_queue_staleness_poll_interval", 1.0))
    while True:
        partitions = run(client.async_get_partition_list())
        train_partitions = [p for p in partitions if str(p).startswith(TRAIN_PARTITION_PREFIX)]
        if len(train_partitions) <= max_staleness:
            return
        logger.info(
            "TransferQueue staleness backpressure: %s train partitions > max_staleness=%s; waiting %.2fs",
            len(train_partitions),
            max_staleness,
            poll_interval,
        )
        time.sleep(poll_interval)


def clear_partition(args: Namespace, client, rollout_id: int) -> None:
    if not transfer_queue_enabled(args) or client is None:
        return
    run(client.async_clear_partition(partition_id=partition_id(rollout_id)))
    logger.info("Cleared TransferQueue partition %s", partition_id(rollout_id))


def partition_id(rollout_id: int) -> str:
    return f"{TRAIN_PARTITION_PREFIX}{rollout_id}"


def default_train_data_fields(args: Namespace) -> list[str]:
    fields = [
        "tokens",
        "total_lengths",
        "response_lengths",
        "loss_masks",
        "rewards",
        "raw_reward",
        "truncated",
        "sample_indices",
        "rollout_log_probs",
    ]
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


def critic_values_via_transfer_queue(args: Namespace) -> bool:
    return (
        transfer_queue_enabled(args)
        and getattr(args, "use_critic", False)
        and int(getattr(args, "context_parallel_size", 1)) == 1
    )


def actor_train_data_fields(args: Namespace) -> list[str]:
    fields = default_train_data_fields(args)
    if critic_values_via_transfer_queue(args):
        for field in CRITIC_VALUE_FIELDS:
            if field not in fields:
                fields.append(field)
    return fields


def get_data_from_transfer_queue(
    args: Namespace,
    client,
    rollout_id: int,
    *,
    task_name: str,
    data_fields: list[str] | None = None,
):
    """Fetch this DP rank's rollout data from TransferQueue and broadcast it to model-parallel ranks."""
    from megatron.core import mpu

    data_fields = data_fields or default_train_data_fields(args)
    total_batch_size = args.rollout_batch_size * args.n_samples_per_prompt
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    if total_batch_size % dp_size != 0:
        raise ValueError(
            "TransferQueue requires rollout_batch_size*n_samples_per_prompt to be divisible by DP size: "
            f"total_batch_size={total_batch_size}, dp_size={dp_size}"
        )
    batch_size = total_batch_size // dp_size
    sampling_config = {
        "dp_rank": mpu.get_data_parallel_rank(with_context_parallel=False),
        "task_name": task_name,
        "batch_index": 0,
        "partition_id": partition_id(rollout_id),
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
            partition_id=partition_id(rollout_id),
            sampling_config=sampling_config,
            task_name=task_name,
        )
        if batch_meta.size != 0:
            payload = [client.get_data(batch_meta), batch_meta]
            logger.info(
                "Fetched TransferQueue data: partition=%s task=%s dp_rank=%s batch_size=%s fields=%s",
                partition_id(rollout_id),
                task_name,
                sampling_config["dp_rank"],
                batch_meta.size,
                data_fields,
            )

    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
    _broadcast_payload(payload, device)
    rollout_data, batch_meta = payload
    if rollout_data is None:
        return None, None
    return tensordict_to_rollout_data(rollout_data), batch_meta


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


def tensordict_to_rollout_data(data) -> dict[str, Any]:
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
            rollout_data[key] = _unwrap_non_tensor_stack(value, NonTensorData)
        elif "lengths" in key or key in {"rewards", "raw_reward", "truncated", "sample_indices", "round_number"}:
            rollout_data[key] = value.tolist() if isinstance(value, torch.Tensor) else list(value)
        elif isinstance(value, torch.Tensor):
            rollout_data[key] = [item for item in value]
        else:
            rollout_data[key] = _unwrap_non_tensor_stack(value, NonTensorData)
    return rollout_data


def _unwrap_non_tensor_stack(value, non_tensor_data_cls):
    output = []
    for item in list(value):
        raw = item.data if non_tensor_data_cls and isinstance(item, non_tensor_data_cls) else item
        if hasattr(raw, "items"):
            raw = dict(raw.items())
        output.append(raw)
    return output
