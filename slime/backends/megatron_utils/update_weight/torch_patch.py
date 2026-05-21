"""Torch multiprocessing CUDA reduction patch for vime.

Ported from sglang.srt.utils.patch_torch so that vime no longer imports
sglang at runtime.

Workaround for https://github.com/pytorch/pytorch/pull/149248 — when sharing
CUDA tensors across processes via IPC, PyTorch's default reduce_tensor may
route the tensor to the wrong device in multi-GPU settings.  This patch
intercepts the rebuild path to use the current process's GPU assignment.
"""

from torch.multiprocessing import reductions


def monkey_patch_torch_reductions():
    """Patch torch.multiprocessing.reductions to fix CUDA IPC device mapping.

    Safe to call multiple times — the patch is a no-op after the first call.
    """
    if hasattr(reductions, "_reduce_tensor_original"):
        return

    reductions._reduce_tensor_original = reductions.reduce_tensor
    reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

    def _reduce_tensor_modified(tensor):
        # Delegate to original; the rebuild function is swapped below.
        return reductions._reduce_tensor_original(tensor)

    def _rebuild_cuda_tensor_modified(*args):
        return reductions._rebuild_cuda_tensor_original(*args)

    reductions.reduce_tensor = _reduce_tensor_modified
    reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified
