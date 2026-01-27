# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration management for Helion kernels.

This module provides centralized configuration file management for Helion custom
operations, including naming conventions, directory resolution, and file I/O.

Config File Structure
---------------------
Each kernel has a single JSON config file: {kernel_name}.json

The file uses a simplified 2-layer hierarchical structure:
{
    "h100": {                             # GPU platform
        "default": { ... },               # Fallback configuration
        "batch_32_hidden_4096": { ... },
        "batch_64_hidden_8192": { ... }
    },
    "a100": {
        "default": { ... },
        "batch_16_hidden_2048": { ... }
    }
}

Example file: silu_mul_fp8.json

Config keys should be structured strings that encode the relevant
parameters (e.g., "batch_32_hidden_4096", "seq_512_heads_16", "fp8_batch_64", etc.).

Classes
-------
- ConfigSet: In-memory collection of configs for a kernel with lookup/query APIs.
- ConfigManager: File-level operations for config persistence.
"""

import json
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "ConfigManager requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion

logger = init_logger(__name__)


class ConfigSet:
    """
    Represents all configurations for a particular kernel (read-only).

    ConfigSet provides an in-memory collection of Helion configs with efficient
    lookup and query capabilities. Configs are stored in a 2-level nested dict
    structure (platform -> config_key -> config) for fast lookups.

    This class is designed for reading and querying configs loaded from files.
    Use ConfigSet.from_dict() to create instances from raw config data.

    Example:
        config_set = manager.load_config_set("silu_mul_fp8")
        platforms = config_set.get_platforms()  # ["h100", "a100"]
        config = config_set.get_config("h100", "batch_32_hidden_4096")
    """

    # Type alias for nested config structure:
    # platform -> config_key -> helion.Config
    _ConfigDict = dict[str, dict[str, "helion.Config"]]

    def __init__(self, kernel_name: str):
        """
        Initialize an empty ConfigSet.

        Args:
            kernel_name: Name of the kernel this config set belongs to.
        """
        self._kernel_name = kernel_name
        self._configs: ConfigSet._ConfigDict = {}

    @property
    def kernel_name(self) -> str:
        """Name of the kernel this config set belongs to."""
        return self._kernel_name

    def get_config(self, platform: str, config_key: str) -> helion.Config:
        """
        Get config for a specific platform and config key.

        Args:
            platform: Platform identifier (e.g., "h100", "a100")
            config_key: Configuration key (e.g., "batch_32_hidden_4096")

        Returns:
            The Helion config for the key.

        Raises:
            KeyError: If config not found, with helpful message listing
                      available options at the failing level.
        """
        platform_dict = self._configs.get(platform)
        if platform_dict is None:
            avail_platforms = self.get_platforms()
            raise KeyError(
                f"Config not found for kernel '{self._kernel_name}': "
                f"platform '{platform}' not found. "
                f"Available platforms: {avail_platforms or '(none)'}"
            )

        config = platform_dict.get(config_key)
        if config is None:
            avail_keys = self.get_config_keys(platform)
            raise KeyError(
                f"Config not found for kernel '{self._kernel_name}': "
                f"config_key '{config_key}' not found for platform '{platform}'. "
                f"Available config_keys: {avail_keys or '(none)'}"
            )

        return config

    def get_platforms(self) -> list[str]:
        """Return all available platform names (sorted)."""
        return sorted(self._configs.keys())

    def get_config_keys(self, platform: str) -> list[str]:
        """
        Return all config keys for a given platform.

        Args:
            platform: Platform name to query.

        Returns:
            Sorted list of config keys, or empty list if platform not found.
        """
        platform_dict = self._configs.get(platform.lower())
        if platform_dict is None:
            return []
        return sorted(platform_dict.keys())

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to hierarchical dictionary for JSON serialization.

        Returns:
            Dictionary with platform/config_key hierarchy.
        """
        result: dict[str, Any] = {}

        for platform, config_keys_dict in self._configs.items():
            result[platform] = {}

            for config_key, config in config_keys_dict.items():
                result[platform][config_key] = config.to_dict()

        return result

    @classmethod
    def from_dict(cls, kernel_name: str, data: dict[str, Any]) -> "ConfigSet":
        """
        Create ConfigSet from hierarchical dictionary.

        Args:
            kernel_name: Name of the kernel.
            data: Hierarchical config dictionary with platform/config_key structure.

        Returns:
            New ConfigSet populated with the configs from data.
        """
        config_set = cls(kernel_name)
        count = 0

        for platform, platform_data in data.items():
            if platform not in config_set._configs:
                config_set._configs[platform] = {}

            for config_key, config_data in platform_data.items():
                config = helion.Config.from_dict(config_data)
                config_set._configs[platform][config_key] = config
                count += 1

        if count > 0:
            logger.debug(
                "Loaded %d configs for kernel '%s'",
                count,
                kernel_name,
            )

        return config_set


class ConfigManager:
    """
    File-level configuration management for Helion kernels.

    ConfigManager handles all file I/O operations for config persistence:
    - Directory resolution and creation
    - Config file path management
    - Raw file loading and saving
    - File existence checks

    For in-memory config queries, use ConfigSet (loaded via load_config_set).

    File naming convention: {kernel_name}.json
    Each file contains all configs for that kernel in a hierarchical structure.

    Example:
        manager = ConfigManager()
        config_set = manager.load_config_set("silu_mul_fp8")
        config = config_set.get_config("h100", "batch_32_hidden_4096")
    """

    def __init__(self, base_dir: str | Path | None = None):
        """
        Initialize ConfigManager.

        Args:
            base_dir: Base directory for configs. If None, uses the configs
                     directory relative to this module.
        """
        self._base_dir = self._resolve_base_dir(base_dir)
        logger.debug("ConfigManager initialized with base_dir: %s", self._base_dir)

    def _resolve_base_dir(self, base_dir: str | Path | None) -> Path:
        """
        Resolve the base directory for configs.

        Args:
            base_dir: User-provided base directory, or None for auto-detection

        Returns:
            Path to the config directory
        """
        if base_dir is not None:
            return Path(base_dir)
        return Path(__file__).parent / "configs"

    def get_config_file_path(self, kernel_name: str) -> Path:
        """
        Get path to the config file for a kernel.

        Args:
            kernel_name: Kernel name string

        Returns:
            Path to {kernel_name}.json
        """
        return self._base_dir / f"{kernel_name}.json"

    def ensure_base_dir_exists(self) -> Path:
        """Ensure base directory exists and return it."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        return self._base_dir

    def load_config_set(self, kernel_name: str) -> ConfigSet:
        """
        Load a ConfigSet for a kernel.

        Args:
            kernel_name: Kernel name string

        Returns:
            ConfigSet populated with configs from the file, or empty ConfigSet
            if file doesn't exist.
        """
        config_path = self.get_config_file_path(kernel_name)
        if not config_path.exists():
            return ConfigSet.from_dict(kernel_name, {})

        try:
            with open(config_path) as f:
                data = json.load(f)
            return ConfigSet.from_dict(kernel_name, data)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load config file %s: %s", config_path, e)
            return ConfigSet.from_dict(kernel_name, {})

    def get_platform_configs(
        self, kernel_name: str, platform: str
    ) -> dict[str, helion.Config]:
        """
        Get all configs for a specific platform.

        Convenience method that loads the config set and returns all configs
        for the specified platform as a dictionary mapping config keys to
        helion.Config objects.

        Args:
            kernel_name: Kernel name to load configs for
            platform: Platform identifier (e.g., "h100", "a100")

        Returns:
            Dictionary mapping config keys to helion.Config objects.
            Empty dictionary if platform has no configs.

        Example:
            configs = manager.get_platform_configs("silu_mul_fp8", "h100")
            # configs = {
            #     "batch_32_hidden_4096": <helion.Config>,
            #     "batch_64_hidden_8192": <helion.Config>,
            #     "default": <helion.Config>
            # }
        """
        config_set = self.load_config_set(kernel_name)
        config_keys = config_set.get_config_keys(platform)

        return {
            config_key: config_set.get_config(platform, config_key)
            for config_key in config_keys
        }

    def save_config_set(self, config_set: ConfigSet) -> Path:
        """
        Save a ConfigSet to its config file.

        Args:
            config_set: The ConfigSet to save.

        Returns:
            Path where config file was saved.
        """
        config_path = self.get_config_file_path(config_set.kernel_name)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config_set.to_dict(), f, indent=2)

        logger.info("Saved config to: %s", config_path)
        return config_path
