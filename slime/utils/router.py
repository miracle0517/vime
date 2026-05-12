"""Rollout HTTP gateway helpers (sglang-router vs vllm-router): CLI registration, RouterArgs, control-plane tier."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import Any, Literal

from packaging.version import parse

logger = logging.getLogger(__name__)

RouterBackend = Literal["sglang", "vllm"]


def register_router_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register gateway CLI: first importable of ``sglang_router`` / ``vllm_router`` wins.

    ``--worker-urls`` stays unprefixed; other flags use ``--router-*`` when ``use_router_prefix=True``.
    """
    last_err: Exception | None = None
    for name, loader in (
        ("sglang_router", _load_sglang_router_args_cls),
        ("vllm_router", _load_vllm_router_args_cls),
    ):
        try:
            router_args_cls = loader()
        except ImportError as e:
            last_err = e
            logger.debug("Router CLI registration: %s not available (%s)", name, e)
            continue
        router_args_cls.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
        logger.debug("Router CLI registration: using %s", name)
        return
    raise ImportError(
        "Neither sglang-router nor vllm-router is installed; "
        "install one of them to use rollout router flags."
    ) from last_err


def _load_sglang_router_args_cls():
    from sglang_router.launch_router import RouterArgs

    return RouterArgs


def _load_vllm_router_args_cls():
    from vllm_router.router_args import RouterArgs

    return RouterArgs


def _sanitize_vllm_router_args(ra: Any) -> Any:
    """Reset negative ints that are valid in sglang CLI but invalid for vllm-router Rust (e.g. u32).

    Slime registers sglang's ``RouterArgs`` first when both wheels are installed; sglang defaults
    ``max_concurrent_requests`` to ``-1`` (unlimited), which must not be forwarded to vllm.
    """
    from vllm_router.router_args import RouterArgs as VR

    fixes: dict[str, Any] = {}
    for f in dataclasses.fields(VR):
        val = getattr(ra, f.name, None)
        if not isinstance(val, int) or val >= 0:
            continue
        if f.default is not dataclasses.MISSING:
            fixes[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            fixes[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            logger.warning(
                "vllm-router: field %r is negative (%s) but has no dataclass default; leaving as-is",
                f.name,
                val,
            )
    return dataclasses.replace(ra, **fixes) if fixes else ra


def router_args_from_cli(impl: RouterBackend, args: argparse.Namespace) -> Any:
    """Build a RouterArgs dataclass from the global argparse namespace."""
    if impl == "vllm":
        from vllm_router.router_args import RouterArgs

        ra = RouterArgs.from_cli_args(args, use_router_prefix=True)
        return _sanitize_vllm_router_args(ra)
    from sglang_router.launch_router import RouterArgs

    return RouterArgs.from_cli_args(args, use_router_prefix=True)


def assert_router_backend_available(impl: RouterBackend) -> None:
    """Raise ``ImportError`` if the chosen gateway package is not installed."""
    if impl == "vllm":
        import vllm_router  # noqa: F401
        from vllm_router.launch_router import launch_router  # noqa: F401

        return
    import sglang_router  # noqa: F401
    from sglang_router.launch_router import launch_router  # noqa: F401


def router_http_api_tier(args: argparse.Namespace) -> Literal["legacy", "mid", "modern"]:
    """Control-plane style for worker register/list/abort: vllm is always ``/workers`` (modern)."""
    impl = getattr(args, "router_impl", "sglang")
    if impl == "vllm":
        return "modern"
    import sglang_router

    ver = parse(sglang_router.__version__)
    if ver <= parse("0.2.1"):
        return "legacy"
    if ver < parse("0.3.0"):
        return "mid"
    return "modern"
