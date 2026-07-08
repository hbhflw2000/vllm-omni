#!/usr/bin/env python3
"""Manual Qwen3-Omni sleep/wake regression probe for vLLM-Omni #4473."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from vllm import SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni


def _extract_text(output: Any) -> str:
    request_output = getattr(output, "request_output", None)
    if request_output is None:
        return ""
    parts = []
    for item in getattr(request_output, "outputs", []) or []:
        text = getattr(item, "text", "")
        if text:
            parts.append(text)
    return "".join(parts)


def _looks_garbled(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) >= 16 and len(set(stripped[:64])) <= 2:
        return True
    bang_count = stripped[:128].count("!")
    return bang_count >= 16 and bang_count / max(1, min(len(stripped), 128)) > 0.5


async def _generate_text(engine: AsyncOmni, prompt: str, request_id: str, sampling_params: SamplingParams) -> str:
    last_output = None
    async for output in engine.generate(
        prompt,
        request_id=request_id,
        sampling_params=sampling_params,
        output_modalities=["text"],
    ):
        last_output = output
    return _extract_text(last_output)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result: dict[str, Any] = {
        "model": args.model_dir,
        "stage_config": args.stage_config,
        "prompt": args.prompt,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    engine = AsyncOmni(
        model=args.model_dir,
        stage_configs_path=args.stage_config,
        init_timeout=args.init_timeout,
        stage_init_timeout=args.stage_init_timeout,
        enable_sleep_mode=True,
    )
    try:
        cold = await _generate_text(engine, args.prompt, "cold", sampling_params)
        result["cold_text"] = cold[: args.print_chars]
        result["cold_garbled"] = _looks_garbled(cold)

        await engine.sleep(stage_ids=[0], level=1)
        first_wake_acks = await engine.wake_up(stage_ids=[0])
        post_full = await _generate_text(engine, args.prompt, "post_full_wake", sampling_params)
        result["level1_full_wake"] = {
            "wake_ack_count": len(first_wake_acks),
            "text": post_full[: args.print_chars],
            "garbled": _looks_garbled(post_full),
        }

        second_wake_acks = await engine.wake_up(stage_ids=[0])
        result["duplicate_wake"] = {
            "ack_count": len(second_wake_acks),
            "ok": second_wake_acks == [],
        }

        await engine.sleep(stage_ids=[0], level=1)
        await engine.wake_up(stage_ids=[0], tags=["weights"])
        try:
            partial_text = await _generate_text(engine, args.prompt, "partial_wake", sampling_params)
            result["partial_wake_generate"] = {
                "rejected": False,
                "text": partial_text[: args.print_chars],
                "garbled": _looks_garbled(partial_text),
            }
        except RuntimeError as exc:
            result["partial_wake_generate"] = {
                "rejected": "partially or fully asleep" in str(exc),
                "error": str(exc),
            }
        finally:
            await engine.wake_up(stage_ids=[0])

        await engine.sleep(stage_ids=[0], level=2)
        try:
            await engine.wake_up(stage_ids=[0])
            result["level2_wake"] = {"rejected": False}
        except NotImplementedError as exc:
            result["level2_wake"] = {
                "rejected": "sleep(level=2)" in str(exc),
                "error": str(exc),
            }
    finally:
        engine.shutdown()
        await asyncio.sleep(1.0)

    result["passed"] = (
        result.get("cold_garbled") is False
        and result.get("level1_full_wake", {}).get("garbled") is False
        and result.get("duplicate_wake", {}).get("ok") is True
        and result.get("partial_wake_generate", {}).get("rejected") is True
        and result.get("level2_wake", {}).get("rejected") is True
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--stage-config", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--prompt",
        default="Solve this math problem step by step: Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--print-chars", type=int, default=512)
    parser.add_argument("--init-timeout", type=int, default=1200)
    parser.add_argument("--stage-init-timeout", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(_run(args))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
