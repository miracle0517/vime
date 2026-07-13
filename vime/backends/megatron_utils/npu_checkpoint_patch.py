from functools import wraps


def patch_mindspeed_moe_checkpoint() -> None:
    """Use expert TP metadata when MindSpeed saves grouped MoE weights."""
    from megatron.core import mpu
    from mindspeed.te.pytorch.module import grouped_linear

    if getattr(grouped_linear, "_vime_moe_checkpoint_patched", False):
        return

    original = grouped_linear.make_sharded_tensors_for_checkpoint

    @wraps(original)
    def make_expert_sharded_tensors(*args, **kwargs):
        # MindSpeed's grouped-linear helper omits these groups, causing Megatron
        # to fall back to the regular TP group for expert checkpoint metadata.
        if len(args) < 6:
            kwargs.setdefault("tp_group", mpu.get_expert_tensor_parallel_group())
        if len(args) < 7:
            kwargs.setdefault(
                "dp_cp_group",
                mpu.get_data_parallel_group(with_context_parallel=True),
            )
        return original(*args, **kwargs)

    grouped_linear.make_sharded_tensors_for_checkpoint = make_expert_sharded_tensors
    grouped_linear._vime_moe_checkpoint_patched = True
