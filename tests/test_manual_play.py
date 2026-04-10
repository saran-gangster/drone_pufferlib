from drone_pufferlib.tools.manual_play import resolve_seed


def test_resolve_seed_preserves_zero():
    assert resolve_seed(0, 7) == 0


def test_resolve_seed_uses_config_default_when_missing():
    assert resolve_seed(None, 7) == 7
