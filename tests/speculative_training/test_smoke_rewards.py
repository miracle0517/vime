from argparse import Namespace
from dataclasses import dataclass, field

from vime.backends.speculative_training.smoke_rewards import ensure_nonzero_grpo_signal


@dataclass
class _Sample:
    group_index: int
    index: int
    reward: float
    metadata: dict = field(default_factory=dict)

    def get_reward_value(self, args):
        return self.reward if not args.reward_key else self.reward[args.reward_key]


def _args():
    return Namespace(
        advantage_estimator="grpo",
        reward_key=None,
        rewards_normalization=True,
        grpo_std_normalization=True,
    )


def test_smoke_reward_fallback_makes_zero_std_group_trainable():
    samples = [
        _Sample(group_index=0, index=0, reward=0.0),
        _Sample(group_index=0, index=1, reward=0.0),
    ]

    raw_rewards, rewards = ensure_nonzero_grpo_signal(_args(), samples)

    assert raw_rewards == [0.0, 1.0]
    assert rewards[0] < 0
    assert rewards[1] > 0
    assert samples[0].metadata["smoke_original_raw_reward"] == 0.0
    assert samples[1].metadata["smoke_original_raw_reward"] == 0.0


def test_smoke_reward_fallback_preserves_nonzero_std_group():
    samples = [
        _Sample(group_index=0, index=0, reward=0.0),
        _Sample(group_index=0, index=1, reward=1.0),
    ]

    raw_rewards, rewards = ensure_nonzero_grpo_signal(_args(), samples)

    assert raw_rewards == [0.0, 1.0]
    assert rewards[0] < 0
    assert rewards[1] > 0
    assert samples[0].metadata == {}
    assert samples[1].metadata == {}
