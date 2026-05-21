"""Multiprocessing serialization utilities for cross-process tensor transfer.

Ported from sglang.srt.utils.common (MultiprocessingSerializer) so that vime
no longer imports sglang at runtime.
"""

import io
from multiprocessing.reduction import ForkingPickler

import pybase64


class _SafeUnpickler(io.BytesIO.__class__):
    pass


def _safe_load(data: bytes):
    import pickle  # noqa: S403

    return pickle.loads(data)  # noqa: S301


class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """Serialize *obj* using ForkingPickler (handles CUDA tensors via IPC).

        Args:
            obj: The object to serialize.
            output_str: If True return a base64-encoded string instead of raw bytes.

        Returns:
            bytes or str
        """
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()
        if output_str:
            output = pybase64.b64encode(output).decode("utf-8")
        return output

    @staticmethod
    def deserialize(data):
        """Deserialize data previously produced by :meth:`serialize`.

        Args:
            data: bytes or base64-encoded str

        Returns:
            The deserialized object.
        """
        if isinstance(data, str):
            data = pybase64.b64decode(data, validate=True)
        import pickle  # noqa: S403

        return pickle.loads(data)  # noqa: S301
