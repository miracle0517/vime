"""Reward helpers for external-draft smoke tests.

These helpers are intentionally opt-in through ``--custom-reward-post-process-path``.
They make tiny smoke runs stable enough to validate the Actor backward path even
when a two-sample GRPO group receives identical rule-based rewards.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return float(value)


def _normalize_group(values: list[float], *, normalize: bool, std_normalize: bool) -> list[float]:
    if not normalize:
        return values
    mean = sum(values) / len(values)
    rewards = [value - mean for value in values]
    if std_normalize:
        if len(values) <= 1:
            return [0.0 for _ in values]
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = variance**0.5
        rewards = [value / (std + 1e-6) for value in rewards]
    return rewards


def ensure_nonzero_grpo_signal(args, samples):
    """Return a deterministic non-zero GRPO signal for tiny smoke batches.

    The regular GRPO post-process subtracts each prompt group's mean reward. If
    all samples in a group are equally correct or equally wrong, the whole group
    becomes zero and the Actor gradient is exactly zero. That is a valid training
    outcome, but it is a poor smoke-test signal.

    This hook preserves real rewards whenever a group already has reward
    variance. For zero-variance groups it alternates 0/1 by sample position so
    the smoke can verify non-zero Actor gradients deterministically. The original
    rule-based reward is kept in ``sample.metadata["smoke_original_raw_reward"]``.
    """

    raw_rewards = [_as_float(sample.get_reward_value(args)) for sample in samples]

    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for position, (sample, raw_reward) in enumerate(zip(samples, raw_rewards)):
        group_index = sample.group_index if sample.group_index is not None else position
        groups[int(group_index)].append((position, raw_reward))

    synthetic_raw = list(raw_rewards)
    for indexed in groups.values():
        values = [reward for _, reward in indexed]
        if len(values) <= 1 or max(values) - min(values) > 1e-6:
            continue
        for local_position, (position, _reward) in enumerate(indexed):
            synthetic_raw[position] = float(local_position % 2)
            metadata = samples[position].metadata
            if metadata is None:
                metadata = {}
                samples[position].metadata = metadata
            metadata["smoke_original_raw_reward"] = raw_rewards[position]

    rewards_out = [0.0] * len(samples)
    normalize = (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", True)
    )
    std_normalize = (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]
        and getattr(args, "grpo_std_normalization", True)
    )
    for indexed in groups.values():
        positions = [position for position, _ in indexed]
        values = [synthetic_raw[position] for position in positions]
        normalized_values = _normalize_group(values, normalize=normalize, std_normalize=std_normalize)
        for position, reward in zip(positions, normalized_values):
            rewards_out[position] = reward

    return synthetic_raw, rewards_out
