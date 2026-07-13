from types import SimpleNamespace

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    _ensure_qwen3_omni_text_rope_parameters,
)


def test_qwen3_omni_thinker_copies_mrope_scaling_to_rope_parameters():
    text_config = SimpleNamespace(
        rope_scaling={
            "rope_type": "default",
            "interleaved": True,
            "mrope_section": [24, 20, 20],
        },
        rope_parameters=None,
        rope_theta=1000000,
    )

    _ensure_qwen3_omni_text_rope_parameters(text_config)

    assert text_config.rope_parameters == {
        "rope_type": "default",
        "interleaved": True,
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_theta": 1000000,
    }


def test_qwen3_omni_thinker_preserves_existing_rope_parameters():
    text_config = SimpleNamespace(
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
        rope_parameters={"rope_type": "default", "rope_theta": 1000000},
    )

    _ensure_qwen3_omni_text_rope_parameters(text_config)

    assert text_config.rope_parameters["rope_theta"] == 1000000
    assert text_config.rope_parameters["mrope_section"] == [24, 20, 20]
    assert text_config.rope_parameters["mrope_interleaved"] is True
