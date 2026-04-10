from drone_pufferlib.utils.config import apply_overrides


def test_apply_overrides_updates_nested_values_without_mutating_input():
    base = {
        "training": {"total_env_steps": 100},
        "env": {"difficulty": "easy"},
    }
    updated = apply_overrides(base, ["training.total_env_steps=200", "env.difficulty=medium"])
    assert updated["training"]["total_env_steps"] == 200
    assert updated["env"]["difficulty"] == "medium"
    assert base["training"]["total_env_steps"] == 100
    assert base["env"]["difficulty"] == "easy"
