import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock


def _load_patch_module():
    path = Path(__file__).resolve().parents[4] / "vime" / "backends" / "megatron_utils" / "npu_checkpoint_patch.py"
    spec = importlib.util.spec_from_file_location("npu_checkpoint_patch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mindspeed_moe_checkpoint_uses_expert_parallel_groups(monkeypatch):
    calls = []

    def make_sharded_tensors(
        state_dict,
        prefix,
        tensor_parallel_layers_axis_map,
        sharded_offsets,
        tp_group=None,
        dp_cp_group=None,
    ):
        calls.append((tp_group, dp_cp_group))
        return "sharded"

    mpu = MagicMock()
    mpu.get_expert_tensor_parallel_group.return_value = "expert_tp"
    mpu.get_expert_data_parallel_group.return_value = "expert_dp"

    megatron_core = types.ModuleType("megatron.core")
    megatron_core.mpu = mpu
    megatron = types.ModuleType("megatron")
    megatron.core = megatron_core

    grouped_linear = types.ModuleType("mindspeed.te.pytorch.module.grouped_linear")
    grouped_linear.make_sharded_tensors_for_checkpoint = make_sharded_tensors
    grouped_module = types.ModuleType("mindspeed.te.pytorch.module")
    grouped_module.grouped_linear = grouped_linear

    modules = {
        "megatron": megatron,
        "megatron.core": megatron_core,
        "mindspeed": types.ModuleType("mindspeed"),
        "mindspeed.te": types.ModuleType("mindspeed.te"),
        "mindspeed.te.pytorch": types.ModuleType("mindspeed.te.pytorch"),
        "mindspeed.te.pytorch.module": grouped_module,
        "mindspeed.te.pytorch.module.grouped_linear": grouped_linear,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    patch_module = _load_patch_module()
    patch_module.patch_mindspeed_moe_checkpoint()
    patched = grouped_linear.make_sharded_tensors_for_checkpoint
    patch_module.patch_mindspeed_moe_checkpoint()

    assert grouped_linear.make_sharded_tensors_for_checkpoint is patched
    assert patched("state", "prefix", {}, ()) == "sharded"
    assert calls[0] == ("expert_tp", "expert_dp")

    patched("state", "prefix", {}, (), "provided_tp")
    assert calls[1] == ("provided_tp", "expert_dp")

    patched("state", "prefix", {}, (), "provided_tp", "provided_dp")
    assert calls[2] == ("provided_tp", "provided_dp")

    patched("state", "prefix", {}, (), tp_group="keyword_tp")
    assert calls[3] == ("keyword_tp", "expert_dp")


def test_npu_distributed_optimizer_uses_fully_parallel_checkpoint():
    patch_module = _load_patch_module()
    args = Namespace(
        use_distributed_optimizer=True,
        no_save_optim=False,
        ckpt_fully_parallel_save=False,
    )

    patch_module.configure_npu_checkpoint(args)

    assert args.ckpt_fully_parallel_save is True


def test_npu_checkpoint_preserves_strategy_when_optimizer_is_not_saved():
    patch_module = _load_patch_module()
    args = Namespace(
        use_distributed_optimizer=True,
        no_save_optim=True,
        ckpt_fully_parallel_save=False,
    )

    patch_module.configure_npu_checkpoint(args)

    assert args.ckpt_fully_parallel_save is False
