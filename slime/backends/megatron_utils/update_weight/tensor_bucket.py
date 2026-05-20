"""Flattened tensor bucket for efficient bucketed weight transfer.

Ported from sglang.srt.weight_sync.tensor_bucket so that vime no longer
imports sglang at runtime.
"""

from dataclasses import dataclass

import torch


@dataclass
class FlattenedTensorMetadata:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int


class FlattenedTensorBucket:
    """Flattens multiple named tensors into a single uint8 tensor for transport.

    Supports mixed dtypes by casting each tensor to a byte view before
    concatenation.
    """

    supports_multi_dtypes = True

    def __init__(
        self,
        named_tensors: list[tuple[str, torch.Tensor]] = None,
        flattened_tensor: torch.Tensor = None,
        metadata: list[FlattenedTensorMetadata] = None,
    ):
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")
            self.metadata: list[FlattenedTensorMetadata] = [None] * len(named_tensors)
            flat_parts: list[torch.Tensor] = [None] * len(named_tensors)
            current_idx = 0
            for i, (name, tensor) in enumerate(named_tensors):
                flat = tensor.flatten().view(torch.uint8)
                flat_parts[i] = flat
                numel = flat.numel()
                self.metadata[i] = FlattenedTensorMetadata(
                    name=name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    start_idx=current_idx,
                    end_idx=current_idx + numel,
                    numel=numel,
                )
                current_idx += numel
            self.flattened_tensor: torch.Tensor = torch.cat(flat_parts, dim=0)
        else:
            if flattened_tensor is None or metadata is None:
                raise ValueError("Must provide either named_tensors or both flattened_tensor and metadata")
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata

    def get_flattened_tensor(self) -> torch.Tensor:
        return self.flattened_tensor

    def get_metadata(self) -> list[FlattenedTensorMetadata]:
        return self.metadata

    def reconstruct_tensors(self) -> list[tuple[str, torch.Tensor]]:
        result = [None] * len(self.metadata)
        for i, meta in enumerate(self.metadata):
            tensor = self.flattened_tensor[meta.start_idx : meta.end_idx].view(meta.dtype).reshape(meta.shape)
            result[i] = (meta.name, tensor)
        return result
