import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


def _load_patch_module():
    path = Path(__file__).resolve().parents[4] / "vime" / "backends" / "megatron_utils" / "npu_checkpoint_patch.py"
    spec = importlib.util.spec_from_file_location("npu_checkpoint_patch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mindspeed_moe_checkpoint_uses_expert_tp_group(monkeypatch):
    calls = []

    def make_sharded_tensors(*args, **kwargs):
        calls.append((args, kwargs))
        return "sharded"

    mpu = MagicMock()
    mpu.get_expert_tensor_parallel_group.return_value = "expert_tp"
    mpu.get_data_parallel_group.return_value = "dp_cp"

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
    assert calls[0][1] == {"tp_group": "expert_tp", "dp_cp_group": "dp_cp"}
    mpu.get_data_parallel_group.assert_called_once_with(with_context_parallel=True)

    patched("state", "prefix", {}, (), "extra", "provided_tp", "provided_dp")
    assert calls[1][1] == {}
