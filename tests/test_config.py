"""Tests for OmniWatch configuration loader and presets."""

from omniwatch.config import load_config


def test_load_config_default():
    config = load_config()
    assert config.active_preset in ("fast", "balanced", "accurate")
    current = config.current
    assert current.sample_fps > 0
    assert current.weights.w_emb > 0


def test_preset_override():
    config_fast = load_config(preset="fast")
    assert config_fast.active_preset == "fast"
    assert config_fast.current.sample_fps == 1.0

    config_acc = load_config(preset="accurate")
    assert config_acc.active_preset == "accurate"
    assert config_acc.current.sample_fps == 4.0
