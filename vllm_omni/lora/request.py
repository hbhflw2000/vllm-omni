# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from msgspec import field

# As these are user-facing variables, define them here so users can import
# LoRA request types directly from vllm_omni.
from vllm.lora.request import LoRARequest


class TensorLoRARequest(LoRARequest):
    """LoRA request carrying adapter tensors directly instead of a file path."""

    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


__all__ = ["LoRARequest", "TensorLoRARequest"]
