"""Backward-compatibility tests: RolloutConfig still accepts legacy 'sglang:' YAML key."""

import tempfile

import pytest
import yaml


def _write_yaml(data: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.flush()
    return f.name


class TestRolloutConfigLegacySglangKey:
    def test_update_weights_default_none_with_sglang_key(self):
        """Legacy 'sglang:' key is still parsed; update_weights defaults to None."""
        from slime.utils.rollout_config import RolloutConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    }
                ]
            }
        )
        config = RolloutConfig.from_yaml(path)
        assert len(config.models) == 1
        assert config.models[0].update_weights is None

    def test_update_weights_explicit_false_with_sglang_key(self):
        """Legacy 'sglang:' key with explicit update_weights values is parsed correctly."""
        from slime.utils.rollout_config import RolloutConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "update_weights": True,
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "model_path": "/path/to/ref",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 2}],
                    },
                ]
            }
        )
        config = RolloutConfig.from_yaml(path)
        assert len(config.models) == 2
        assert config.models[0].name == "actor"
        assert config.models[0].update_weights is True
        assert config.models[1].name == "ref"
        assert config.models[1].update_weights is False
        assert config.models[1].model_path == "/path/to/ref"

    def test_multi_model_total_gpus_with_sglang_key(self):
        """total_num_gpus sums correctly when parsed from legacy 'sglang:' key."""
        from slime.utils.rollout_config import RolloutConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                ]
            }
        )
        config = RolloutConfig.from_yaml(path)
        assert config.total_num_gpus == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
