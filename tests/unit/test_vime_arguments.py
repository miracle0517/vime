from __future__ import annotations

import argparse

import pytest


@pytest.mark.unit
def test_non_colocate_weight_sync_backend_arg_defaults_to_broadcast():
    from vime.utils.arguments import get_vime_extra_args_provider

    parser = argparse.ArgumentParser()
    get_vime_extra_args_provider()(parser)

    args, _ = parser.parse_known_args(["--rollout-batch-size", "1"])

    assert args.non_colocate_weight_sync_backend == "broadcast"


@pytest.mark.unit
def test_non_colocate_weight_sync_backend_arg_accepts_nccl_xfer():
    from vime.utils.arguments import get_vime_extra_args_provider

    parser = argparse.ArgumentParser()
    get_vime_extra_args_provider()(parser)

    args, _ = parser.parse_known_args(
        ["--rollout-batch-size", "1", "--non-colocate-weight-sync-backend", "nccl-xfer"]
    )

    assert args.non_colocate_weight_sync_backend == "nccl-xfer"
