import pytest
import torch

from vime.utils.tensor_backper import TensorBackuper


@pytest.mark.unit
def test_tensor_backuper_verify_detects_restore_mismatch(monkeypatch):
    original_empty_like = torch.empty_like
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda tensor, **kwargs: original_empty_like(
            tensor,
            device=kwargs.get("device"),
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    source = {"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    backuper = TensorBackuper.create(lambda: source.items(), single_tag=None)
    backuper.backup("actor")

    source["weight"].zero_()
    backuper.restore("actor")
    backuper.verify("actor")

    source["weight"][1] = -1
    with pytest.raises(AssertionError, match="changed=.*weight"):
        backuper.verify("actor")
