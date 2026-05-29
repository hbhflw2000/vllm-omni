"""Compatibility helpers for vLLM APIs that move across releases."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any


def _routed_experts_capturer_module() -> Any | None:
    try:
        return import_module("vllm.model_executor.layers.fused_moe.routed_experts_capturer")
    except ImportError:
        return None


def split_routed_experts(routed_experts: Any, prompt_len: int, num_gen: int | None = None) -> tuple[Any, Any]:
    module = _routed_experts_capturer_module()
    if module is not None:
        upstream = getattr(module, "split_routed_experts", None)
        if upstream is not None:
            return upstream(routed_experts, prompt_len, num_gen)

    if routed_experts is None:
        return None, None
    try:
        gen_end = None if num_gen is None else prompt_len + num_gen
        return routed_experts[:prompt_len], routed_experts[prompt_len:gen_end]
    except Exception:
        return None, routed_experts


def extract_routed_experts_for_current_batch(*args: Any, **kwargs: Any) -> Any | None:
    module = _routed_experts_capturer_module()
    if module is not None:
        upstream = getattr(module, "extract_routed_experts_for_current_batch", None)
        if upstream is not None:
            return upstream(*args, **kwargs)
    return None


def issue_routing_d2h_copy(*args: Any, **kwargs: Any) -> None:
    module = _routed_experts_capturer_module()
    if module is not None:
        upstream = getattr(module, "issue_routing_d2h_copy", None)
        if upstream is not None:
            upstream(*args, **kwargs)


def get_global_experts_capturer() -> Any | None:
    module = _routed_experts_capturer_module()
    if module is None:
        return None

    upstream = getattr(module, "get_global_experts_capturer", None)
    if upstream is not None:
        return upstream()

    capturer_cls = getattr(module, "RoutedExpertsCapturer", None)
    if capturer_cls is None:
        return None
    return capturer_cls.get_instance()


class _Gemma4ProposerCompatMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        module = sys.modules.get("vllm.v1.spec_decode.gemma4")
        real_cls = getattr(module, "Gemma4Proposer", None) if module is not None else None
        return real_cls is not None and isinstance(instance, real_cls)


class Gemma4Proposer(metaclass=_Gemma4ProposerCompatMeta):
    """Placeholder used only for isinstance unions when vLLM lacks Gemma4."""

    pass
