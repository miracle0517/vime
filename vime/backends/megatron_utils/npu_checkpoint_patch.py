import inspect
from argparse import Namespace
from functools import wraps


def configure_npu_checkpoint(args: Namespace) -> None:
    """Avoid gathering distributed optimizer state through HCCL when saving."""
    if getattr(args, "use_distributed_optimizer", False) and not getattr(args, "no_save_optim", False):
        args.ckpt_fully_parallel_save = True


def patch_mindspeed_moe_checkpoint() -> None:
    """Save MindSpeed grouped experts with expert-parallel metadata."""
    from megatron.core import mpu
    from mindspeed.te.pytorch.module import grouped_linear

    if getattr(grouped_linear, "_vime_moe_checkpoint_patched", False):
        return

    original = grouped_linear.make_sharded_tensors_for_checkpoint
    original_signature = inspect.signature(original)

    @wraps(original)
    def make_expert_sharded_tensors(*args, **kwargs):
        # MindSpeed omits both groups here, so Megatron falls back to the
        # regular TP/DP groups even though these tensors are expert-sharded.
        supplied_arguments = original_signature.bind_partial(*args, **kwargs).arguments
        if "tp_group" not in supplied_arguments:
            kwargs["tp_group"] = mpu.get_expert_tensor_parallel_group()
        if "dp_cp_group" not in supplied_arguments:
            kwargs["dp_cp_group"] = mpu.get_expert_data_parallel_group()
        return original(*args, **kwargs)

    grouped_linear.make_sharded_tensors_for_checkpoint = make_expert_sharded_tensors
    grouped_linear._vime_moe_checkpoint_patched = True
