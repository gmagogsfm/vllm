# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Helion ConfigManager and ConfigSet.

Tests the simplified configuration management system for Helion custom kernels.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from vllm.utils.import_utils import has_helion

# Skip entire module if helion is not available
if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

from vllm.kernels.helion.config_manager import (
    ConfigManager,
    ConfigSet,
)


class TestConfigSet:
    """Test suite for ConfigSet class."""

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_config_set_creation(self, mock_helion):
        """Test creating an empty ConfigSet."""
        config_set = ConfigSet("test_kernel")

        assert config_set.kernel_name == "test_kernel"
        assert config_set.get_platforms() == []

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_config_set_from_dict(self, mock_helion):
        """Test creating ConfigSet from dictionary data."""
        mock_config = Mock()
        mock_helion.Config.from_dict.return_value = mock_config

        data = {"h100": {"batch_32_hidden_4096": {"test": "data"}}}

        config_set = ConfigSet.from_dict("test_kernel", data)

        assert config_set.kernel_name == "test_kernel"
        assert config_set.get_platforms() == ["h100"]
        mock_helion.Config.from_dict.assert_called_once_with({"test": "data"})
        assert config_set.get_config("h100", "batch_32_hidden_4096") == mock_config

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_config_set_get_config_keyerror(self, mock_helion):
        """Test that accessing non-existent configs raises informative KeyErrors."""
        config_set = ConfigSet("test_kernel")

        with pytest.raises(KeyError, match="platform 'h100' not found"):
            config_set.get_config("h100", "batch_32_hidden_4096")

        data = {"h100": {"batch_64_hidden_2048": {"test": "data"}}}
        config_set = ConfigSet.from_dict("test_kernel", data)

        with pytest.raises(
            KeyError, match="config_key 'batch_32_hidden_4096' not found"
        ):
            config_set.get_config("h100", "batch_32_hidden_4096")

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_config_set_get_platforms(self, mock_helion):
        """Test get_platforms method."""
        mock_config1 = Mock()
        mock_config2 = Mock()
        mock_helion.Config.from_dict.side_effect = [mock_config1, mock_config2]

        data = {
            "h100": {"batch_32_hidden_4096": {"config": "1"}},
            "a100": {"batch_16_hidden_2048": {"config": "2"}},
        }
        config_set = ConfigSet.from_dict("test_kernel", data)

        platforms = config_set.get_platforms()
        assert platforms == ["a100", "h100"]  # Should be sorted

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_config_set_get_config_keys(self, mock_helion):
        """Test get_config_keys method."""
        mock_config1 = Mock()
        mock_config2 = Mock()
        mock_helion.Config.from_dict.side_effect = [mock_config1, mock_config2]

        data = {
            "h100": {
                "batch_32_hidden_4096": {"config": "1"},
                "batch_64_hidden_2048": {"config": "2"},
            }
        }
        config_set = ConfigSet.from_dict("test_kernel", data)

        config_keys = config_set.get_config_keys("h100")
        assert config_keys == ["batch_32_hidden_4096", "batch_64_hidden_2048"]

        assert config_set.get_config_keys("v100") == []

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_config_set_to_dict(self, mock_helion):
        """Test converting ConfigSet to dictionary."""
        mock_config = Mock()
        mock_config.to_dict.return_value = {"test": "data"}
        mock_helion.Config.from_dict.return_value = mock_config

        original_data = {"h100": {"batch_32_hidden_4096": {"test": "data"}}}
        config_set = ConfigSet.from_dict("test_kernel", original_data)

        result_data = config_set.to_dict()

        assert result_data == original_data
        mock_config.to_dict.assert_called()


class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_config_manager_creation_default_base_dir(self):
        """Test creating ConfigManager with default base directory."""
        manager = ConfigManager()
        assert manager._base_dir.name == "configs"

    def test_config_manager_creation_custom_base_dir(self):
        """Test creating ConfigManager with custom base directory."""
        custom_dir = "/tmp/custom_configs"
        manager = ConfigManager(base_dir=custom_dir)

        assert str(manager._base_dir) == custom_dir

    def test_get_config_file_path(self):
        """Test getting config file path for a kernel."""
        manager = ConfigManager(base_dir="/tmp")

        file_path = manager.get_config_file_path("silu_mul_fp8")

        expected_path = Path("/tmp/silu_mul_fp8.json")
        assert file_path == expected_path

    def test_ensure_base_dir_exists(self):
        """Test ensuring base directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "non_existent" / "configs"
            manager = ConfigManager(base_dir=base_dir)
            assert not base_dir.exists()

            returned_path = manager.ensure_base_dir_exists()

            assert base_dir.exists()
            assert base_dir.is_dir()
            assert returned_path == base_dir

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_load_config_set_file_not_exists(self, mock_helion):
        """Test loading config set when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(base_dir=temp_dir)
            config_set = manager.load_config_set("non_existent_kernel")

            assert isinstance(config_set, ConfigSet)
            assert config_set.kernel_name == "non_existent_kernel"
            assert config_set.get_platforms() == []

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_load_config_set_valid_file(self, mock_helion):
        """Test loading config set from valid file."""
        mock_config = Mock()
        mock_helion.Config.from_dict.return_value = mock_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {"h100": {"batch_32_hidden_4096": {"test": "data"}}}
            config_file = Path(temp_dir) / "test_kernel.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            manager = ConfigManager(base_dir=temp_dir)
            config_set = manager.load_config_set("test_kernel")

            assert isinstance(config_set, ConfigSet)
            assert config_set.kernel_name == "test_kernel"
            assert config_set.get_platforms() == ["h100"]

    def test_load_config_set_invalid_json(self):
        """Test loading config set from file with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_kernel.json"
            with open(config_file, "w") as f:
                f.write("invalid json content {")

            manager = ConfigManager(base_dir=temp_dir)
            config_set = manager.load_config_set("test_kernel")

            assert isinstance(config_set, ConfigSet)
            assert config_set.kernel_name == "test_kernel"
            assert config_set.get_platforms() == []

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_save_config_set(self, mock_helion):
        """Test saving ConfigSet to file."""
        mock_config = Mock()
        mock_config.to_dict.return_value = {"test": "data"}
        mock_helion.Config.from_dict.return_value = mock_config

        with tempfile.TemporaryDirectory() as temp_dir:
            data = {"h100": {"batch_32_hidden_4096": {"test": "data"}}}
            config_set = ConfigSet.from_dict("test_kernel", data)

            manager = ConfigManager(base_dir=temp_dir)
            saved_path = manager.save_config_set(config_set)

            expected_path = Path(temp_dir) / "test_kernel.json"
            assert saved_path == expected_path
            assert saved_path.exists()

            with open(saved_path) as f:
                loaded_data = json.load(f)
            assert loaded_data == data

    def test_save_config_set_creates_directory(self):
        """Test that save_config_set creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nested" / "configs"
            config_set = ConfigSet("test_kernel")

            manager = ConfigManager(base_dir=nested_dir)
            saved_path = manager.save_config_set(config_set)

            assert nested_dir.exists()
            assert nested_dir.is_dir()
            assert saved_path.exists()

    @patch("vllm.kernels.helion.config_manager.helion")
    def test_get_platform_configs(self, mock_helion):
        """Test getting all configs for a specific platform."""
        mock_helion.Config.from_dict.return_value = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                "h100": {
                    "batch_32_hidden_4096": {"config": "1"},
                    "batch_64_hidden_2048": {"config": "2"},
                    "default": {"config": "default"},
                },
                "a100": {"batch_16_hidden_1024": {"config": "3"}},
            }
            config_file = Path(temp_dir) / "test_kernel.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            manager = ConfigManager(base_dir=temp_dir)

            h100_configs = manager.get_platform_configs("test_kernel", "h100")
            assert len(h100_configs) == 3
            assert "batch_32_hidden_4096" in h100_configs
            assert "batch_64_hidden_2048" in h100_configs
            assert "default" in h100_configs
            for config in h100_configs.values():
                assert isinstance(config, Mock)

            a100_configs = manager.get_platform_configs("test_kernel", "a100")
            assert len(a100_configs) == 1
            assert "batch_16_hidden_1024" in a100_configs

            nonexistent_configs = manager.get_platform_configs("test_kernel", "v100")
            assert len(nonexistent_configs) == 0
