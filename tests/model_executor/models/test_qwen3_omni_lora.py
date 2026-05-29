# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_qwen3_omni_thinker_declares_lora_metadata():
    model_cls = Qwen3OmniMoeThinkerForConditionalGeneration

    assert model_cls.supports_lora is True
    assert model_cls.is_3d_moe_weight is True
    assert model_cls.is_non_gated_moe is False
    assert model_cls.embedding_modules == {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    assert model_cls.lora_skip_prefixes == []
