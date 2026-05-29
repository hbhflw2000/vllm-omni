# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import TensorLoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_diffusion_lora_manager_loads_tensor_backed_request(monkeypatch):
    import vllm_omni.diffusion.lora.manager as manager_mod

    peft_config = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]}
    lora_tensors = {"language_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 1)}
    peft_helper = SimpleNamespace(r=16, lora_alpha=32, target_modules=["q_proj"])
    lora_model = SimpleNamespace(id=1, loras={})

    from_dict_calls = []
    from_tensor_calls = []

    monkeypatch.setattr(
        manager_mod.PEFTHelper,
        "from_dict",
        staticmethod(lambda config: from_dict_calls.append(config) or peft_helper),
    )
    monkeypatch.setattr(
        manager_mod.LoRAModel,
        "from_lora_tensors",
        staticmethod(lambda **kwargs: from_tensor_calls.append(kwargs) or lora_model),
    )

    manager = DiffusionLoRAManager.__new__(DiffusionLoRAManager)
    manager._expected_lora_modules = ["q_proj"]
    manager.dtype = torch.bfloat16

    request = TensorLoRARequest(
        lora_name="adapter",
        lora_int_id=1,
        lora_path="/unused",
        peft_config=peft_config,
        lora_tensors=lora_tensors,
    )

    loaded_model, loaded_helper = manager._load_adapter(request)

    assert loaded_model is lora_model
    assert loaded_helper is peft_helper
    assert from_dict_calls == [peft_config]
    assert from_tensor_calls[0]["tensors"] is lora_tensors
    assert from_tensor_calls[0]["lora_model_id"] == 1
    assert from_tensor_calls[0]["dtype"] is torch.bfloat16
