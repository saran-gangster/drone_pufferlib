import pytest

torch = pytest.importorskip("torch")

from drone_pufferlib.models.cnn_policy import CnnActorCritic


def test_deterministic_action_uses_argmax():
    model = CnnActorCritic(obs_shape=(3, 64, 64), action_dim=3, channels=[32, 64, 64], hidden_size=512)
    obs = torch.zeros((1, 3, 64, 64))
    with torch.no_grad():
        logits, _ = model.forward(obs)
        action, _, _ = model.act(obs, deterministic=True)
    assert int(action.item()) == int(torch.argmax(logits, dim=-1).item())
