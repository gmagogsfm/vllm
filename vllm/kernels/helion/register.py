# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Helion kernel registration with pre-tuned config selection.

This module leverages Helion's internal config selection infrastructure to use
pre-tuned configs instead of runtime autotuning.

How Helion Normally Works
-------------------------
For each kernel invocation, Helion:
1. Computes a cache key from input arguments
2. Looks up the key in its internal compilation cache
3. On cache miss, runs autotuning to find the best config
4. Compiles and caches the kernel with that config

How We Override It
------------------
We override two Helion hooks to use pre-tuned configs:

1. **custom_key**: We provide a key function (derived from config_picker) that
   computes cache keys matching our pre-tuned config keys. This ensures Helion's
   internal cache uses keys that correspond to configs we've prepared.

2. **autotuner_fn**: We provide PresetConfigSearch which, instead of autotuning,
   simply returns the pre-tuned config for the computed key. On cache miss,
   Helion calls our autotuner which returns the author-prepared config.

Both hooks use the same config_picker logic to ensure the cache key computed
by custom_key matches the config returned by the autotuner.

Key Classes
-----------
- HelionKernelWrapper: Wraps raw kernel + config_picker, creates configured ops
- ConfiguredHelionKernel: Platform-specific kernel registered as PyTorch custom op
- PresetConfigSearch: Custom autotuner that returns pre-tuned configs
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.library import Library

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.kernels.helion.config_manager import ConfigManager

if not has_helion():
    raise ImportError(
        "register module requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
from helion.autotuner.base_search import BaseAutotuner
from helion.runtime.config import Config

logger = init_logger(__name__)

vllm_helion_lib = Library("vllm_helion", "FRAGMENT")  # noqa


def validate_helion_settings(
    helion_settings: "helion.Settings | None", op_name: str
) -> None:
    """
    Validate that helion_settings doesn't contain conflicting options.

    HelionKernelWrapper uses custom autotuner and key functions derived from
    config_picker. User-provided helion_settings must not conflict with these.

    Args:
        helion_settings: Helion settings object with to_dict() method, or None
        op_name: Name of the kernel operation (for error messages)

    Raises:
        ValueError: If helion_settings contains conflicting options
    """
    if helion_settings is None:
        return

    settings_dict = helion_settings.to_dict()

    if "autotuner_fn" in settings_dict:
        raise ValueError(
            f"HelionKernelWrapper for '{op_name}' uses a custom autotuner via "
            f"config picker. Remove 'autotuner_fn' from helion_settings and use "
            f"@{op_name}.register_config_picker instead."
        )

    if "custom_key" in settings_dict:
        raise ValueError(
            f"HelionKernelWrapper for '{op_name}' uses a custom key function "
            f"derived from config picker. Remove 'custom_key' from helion_settings."
        )

    # Warn if static_shapes is explicitly set to True since most vLLM ops need
    # dynamic shapes for variable batch sizes and sequence lengths
    if settings_dict.get("static_shapes") is True:
        logger.warning(
            "Kernel '%s' has static_shapes=True in helion_settings. "
            "Most vLLM ops require dynamic shapes for variable batch sizes "
            "and sequence lengths. Consider removing this setting.",
            op_name,
        )


class PresetConfigSearch(BaseAutotuner):
    """
    Custom autotuner that uses a preset config selector instead of autotuning.

    This autotuner bypasses Helion's default autotuning by calling a user-provided
    config selector function that returns a pre-determined config based on the
    kernel arguments.
    """

    def __init__(
        self,
        args: tuple[Any, ...],
        config_selector: Callable[[tuple[Any, ...]], Config],
    ):
        """
        Initialize the preset config search.

        Args:
            args: Tuple of arguments passed to the kernel
            config_selector: Function that takes args and returns a Config object
        """
        self.args = args
        self.config_selector = config_selector

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """Return the config selected by the config_selector function."""
        return self.config_selector(self.args)


class ConfiguredHelionKernel:
    """
    A configured Helion kernel bound to a specific platform and model_config.

    This class is registered as a PyTorch custom op. When called, it:
    1. Calls the registered config picker with model_config and runtime args
    2. Selects the best config key based on the picker's decision
    3. Compiles the kernel with that config (cached for reuse)
    4. Executes and returns the result
    """

    def __init__(
        self,
        wrapper: "HelionKernelWrapper",
        platform: str,
        model_config,
        configs: dict[str, Config],
    ):
        self.wrapper = wrapper
        self.platform = platform
        self.model_config = model_config
        self.configs = configs
        self._decorated_kernel: callable | None = None

    def __call__(self, *args, **kwargs):
        """Execute the kernel with dynamic config selection via custom autotuner."""
        return self._get_decorated_kernel()(*args, **kwargs)

    def _create_key_computer(self):
        """
        Create a key computer function derived from the config picker.

        The returned function receives kernel arguments unpacked (*args) to match
        Helion's custom_key signature (called as self._key_fn(*args)).
        """
        if self.wrapper._config_picker is None:
            raise RuntimeError(
                f"No config picker registered for kernel '{self.wrapper.op_name}'. "
                f"Use @{self.wrapper.op_name}.register_config_picker to register one."
            )

        def key_computer(*args):
            config_keys = list(self.configs.keys())
            selected_key = self.wrapper._config_picker(
                self.model_config, args, config_keys
            )
            if selected_key:
                return selected_key
            return "default" if "default" in self.configs else None

        return key_computer

    def _create_config_selector(self, key_computer):
        """
        Create a config selector function that uses the key_computer to ensure
        config key is consistent between custom autotuner and Helion's internal
        compilation cache.

        The returned function receives args as a tuple (from PresetConfigSearch)
        and returns the corresponding helion.Config object.
        """

        def config_selector(args):
            # args is a tuple; key_computer expects unpacked args
            selected_config_key = key_computer(*args)

            if selected_config_key is None:
                raise ValueError(
                    f"Config picker returned None for kernel '{self.wrapper.op_name}' "
                    f"with available config keys: {list(self.configs.keys())}"
                )

            if selected_config_key not in self.configs:
                raise ValueError(
                    f"Config picker returned invalid config key "
                    f"'{selected_config_key}' for kernel '{self.wrapper.op_name}'. "
                    f"Available keys: {list(self.configs.keys())}"
                )

            return self.configs[selected_config_key]

        return config_selector

    def _get_decorated_kernel(self) -> callable:
        """Get or create a decorated kernel using the custom autotuner."""
        if self._decorated_kernel is not None:
            return self._decorated_kernel

        if not self.configs:
            raise ValueError(
                f"No configs available for kernel '{self.wrapper.op_name}' "
                f"with platform='{self.platform}'"
            )

        key_computer = self._create_key_computer()
        config_selector = self._create_config_selector(key_computer)

        kernel_kwargs = {}
        if self.wrapper.helion_settings:
            kernel_kwargs.update(self.wrapper.helion_settings.to_dict())

        # Set static_shapes=False by default if user didn't explicitly set it to True
        # This is needed for dynamic batch sizes and sequence lengths in vLLM
        if kernel_kwargs.get("static_shapes") is not True:
            kernel_kwargs["static_shapes"] = False

        kernel_kwargs["autotuner_fn"] = lambda _, args: PresetConfigSearch(
            args, config_selector
        )
        kernel_kwargs["custom_key"] = key_computer

        logger.debug(
            "Creating decorated kernel %s with custom autotuner on platform %s",
            self.wrapper.op_name,
            self.platform,
        )
        self._decorated_kernel = helion.kernel(**kernel_kwargs)(
            self.wrapper.raw_kernel_func
        )
        return self._decorated_kernel


class HelionKernelWrapper:
    """
    Wrapper for Helion kernels that creates config-specific PyTorch custom ops.

    This wrapper manages the base Helion kernel and creates a single PyTorch
    custom op that dynamically selects batch_size configs at runtime.
    """

    def __init__(
        self,
        raw_kernel_func: Callable,
        op_name: str,
        fake_impl: Callable,
        helion_settings: "helion.Settings | None" = None,
    ):
        # Validate helion_settings doesn't conflict with our custom autotuner
        validate_helion_settings(helion_settings, op_name)

        self.raw_kernel_func = raw_kernel_func
        self.op_name = op_name
        self._fake_impl = fake_impl
        self.helion_settings = helion_settings
        self._config_picker = None

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"HelionKernelWrapper '{self.op_name}' should not be called directly. "
            f"Use create_configured_op_from_model() to get a callable op."
        )

    def register_config_picker(
        self, picker_func: Callable[[Any, tuple[Any, ...], list[str]], str | None]
    ) -> Callable[[Any, tuple[Any, ...], list[str]], str | None]:
        """
        Register a function to pick the best config key from available options.

        Args:
            picker_func: Function with signature:
                (model_config: ModelConfig, args: tuple, config_keys: list[str])
                -> str | None

        Example:
            @kernel_wrapper.register_config_picker
            def pick_config(model_config, args, config_keys):
                # def linear_fp8(x, weight, scale=1.0): ...
                x = args[0]
                input_hidden_size = x.shape[-1]
                input_batch_size = x.shape[0]

                # Find available batch sizes for matching hidden size
                available_batches = []
                for key in config_keys:
                    if key == "default":
                        continue
                    parts = key.split("_")
                    if (len(parts) == 4 and parts[0] == "hiddensize" and
                        parts[1].isdigit() and parts[2] == "batchsize" and
                        parts[3].isdigit()):
                        hidden_size = int(parts[1])
                        batch_size = int(parts[3])
                        if hidden_size == input_hidden_size:
                            available_batches.append(batch_size)

                if not available_batches:
                    return "default" if "default" in config_keys else None

                available_batches.sort()

                # Find first available batch_size that can fit input_batch_size
                for batch_size in available_batches:
                    if batch_size >= input_batch_size:
                        return f"hiddensize_{input_hidden_size}_batchsize_{batch_size}"

                # Fallback to smallest available batch_size
                return (
                    f"hiddensize_{input_hidden_size}_batchsize_{available_batches[0]}"
                )

        The registered config picker is automatically wrapped with a custom
        Helion autotuner (PresetConfigSearch) that calls your picker function
        at runtime and returns the selected helion.Config object to the kernel.

        A custom_key function is also automatically derived from your config
        picker by calling it with a comprehensive set of config keys. This
        ensures cache consistency between the autotuner and key computation.
        """
        self._config_picker = picker_func
        return picker_func

    def create_configured_op_from_model(
        self, model_config: "ModelConfig", config_manager: "ConfigManager"
    ) -> Any:
        """
        Create and register a configured kernel as a PyTorch custom op.

        Returns the torch.ops callable that wraps a ConfiguredHelionKernel.
        The kernel dynamically selects configs at runtime using the registered
        config picker, which receives both model_config and runtime args.

        Args:
            model_config: vLLM ModelConfig for config selection
            config_manager: ConfigManager instance for loading configs

        Returns:
            PyTorch ops callable (torch.ops.vllm_helion.{op_name}_{platform})
        """
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        assert model_config is not None, (
            f"{self.op_name}.create_configured_op_from_model() requires model_config"
        )
        assert config_manager is not None, (
            f"{self.op_name}.create_configured_op_from_model() requires config_manager"
        )
        platform = get_canonical_gpu_name()
        configs = config_manager.get_platform_configs(self.op_name, platform)
        assert len(configs) > 0, (
            f"No configs available for kernel '{self.op_name}' on platform '{platform}'"
        )
        assert self._config_picker is not None, (
            f"No config picker registered for kernel '{self.op_name}'. "
            f"Use @{self.op_name}.register_config_picker to register one."
        )

        configured_op_name = f"{self.op_name}_{platform}"

        if hasattr(torch.ops.vllm_helion, configured_op_name):
            logger.debug("Op vllm_helion::%s already registered", configured_op_name)
            return getattr(torch.ops.vllm_helion, configured_op_name)

        configured_kernel = ConfiguredHelionKernel(
            wrapper=self,
            platform=platform,
            model_config=model_config,
            configs=configs,
        )

        logger.info("Registering op: vllm_helion::%s", configured_op_name)
        direct_register_custom_op(
            op_name=configured_op_name,
            op_func=configured_kernel,
            # TODO(gmagogsfm): Implement automatic mutation/aliasing detection
            # for Helion kernels.
            mutates_args=None,
            fake_impl=self._fake_impl,
            target_lib=vllm_helion_lib,
        )
        return getattr(torch.ops.vllm_helion, configured_op_name)


# Global registry for tracking all registered HelionKernelWrapper instances
_REGISTERED_KERNELS: dict[str, HelionKernelWrapper] = {}


def get_registered_kernels() -> dict[str, HelionKernelWrapper]:
    """
    Get all registered Helion kernel wrappers from the global registry.

    Returns:
        Dictionary mapping kernel names to HelionKernelWrapper instances
    """
    return _REGISTERED_KERNELS.copy()


def get_kernel_by_name(kernel_name: str) -> HelionKernelWrapper | None:
    """
    Get a specific registered kernel by name.

    Args:
        kernel_name: Name of the kernel to retrieve

    Returns:
        HelionKernelWrapper instance if found, None otherwise
    """
    return _REGISTERED_KERNELS.get(kernel_name)


def infer_fake_impl(
    kernel_func: Callable,
    helion_settings: "helion.Settings | None" = None,
) -> Callable:
    """
    Create an auto-generated fake implementation for a Helion kernel.

    This function creates a fake_impl that uses Helion's bind/compile pattern
    with a dummy launcher to generate output tensors with correct shapes and
    dtypes without actually executing the kernel.

    Args:
        kernel_func: The raw Helion kernel function
        helion_settings: Optional helion.Settings object for kernel configuration

    Returns:
        A callable fake implementation suitable for torch.compile
    """

    def helion_fake_kernel(*args, **kwargs):
        """
        Run the Helion kernel with a dummy launcher to generate correct
        output shapes.

        Uses Helion's bind/compile pattern where _launcher is passed to
        the compiled runner. The dummy launcher skips actual execution
        while still producing tensors with correct shapes/dtypes.
        """
        kernel_kwargs = {}
        if helion_settings:
            kernel_kwargs.update(helion_settings.to_dict())

        temp_decorated_kernel = helion.kernel(**kernel_kwargs)(kernel_func)

        # Bind with args to get config_spec, then get a valid default config
        bound = temp_decorated_kernel.bind(args)
        default_config = bound.config_spec.default_config()
        compiled_runner = bound.compile_config(default_config)

        return compiled_runner(*args, **kwargs, _launcher=lambda *a, **kw: None)

    return helion_fake_kernel


def register_kernel(
    op_name_or_func: str | Callable | None = None,
    *,
    fake_impl: Callable | None = None,
    helion_settings: "helion.Settings | None" = None,
) -> Callable:
    """
    Decorator to register a Helion kernel function as a HelionKernelWrapper.

    This decorator:
    1. Wraps the raw kernel function in a HelionKernelWrapper
    2. Registers the wrapper in the global kernel registry

    The resulting HelionKernelWrapper can be used to create platform-specific
    configured ops via create_configured_op_from_model().

    Can be used with or without parentheses:
        @register_kernel
        def my_kernel(...): ...

        @register_kernel("custom_name")
        def my_kernel(...): ...

    Args:
        op_name_or_func: Name of the operation, or the decorated function when
                used without parentheses. If None or a function, automatically
                uses the decorated function's __name__.
        fake_impl: Optional fake implementation for torch.compile compatibility.
                  If not provided, one is auto-generated using Helion's
                  bind/compile pattern with a dummy launcher.
        helion_settings: Optional helion.Settings object for kernel configuration

    Returns:
        Decorator function that wraps the kernel and returns HelionKernelWrapper

    Example:
        # With auto-generated fake_impl:
        @register_kernel("silu_mul_fp8", helion_settings=my_settings)
        def silu_mul_fp8_kernel(x: torch.Tensor, scale: torch.Tensor):
            # Helion kernel implementation
            ...

        # With custom fake_impl:
        def silu_mul_fp8_fake(x: torch.Tensor, scale: torch.Tensor):
            return torch.empty(x.shape, dtype=torch.float8_e4m3fn,
                               device=x.device)

        @register_kernel("silu_mul_fp8", fake_impl=silu_mul_fp8_fake)
        def silu_mul_fp8_kernel(x: torch.Tensor, scale: torch.Tensor):
            ...

        # Register config picker:
        @silu_mul_fp8_kernel.register_config_picker
        def pick_config(model_config, args, config_keys):
            ...

        # Create configured op at runtime:
        torch_op = silu_mul_fp8_kernel.create_configured_op_from_model(
            model_config, config_manager)
    """

    def decorator(kernel_func: Callable) -> HelionKernelWrapper:
        op_name = op_name_or_func if isinstance(op_name_or_func, str) else None
        final_op_name = op_name if op_name else kernel_func.__name__

        if final_op_name in _REGISTERED_KERNELS:
            raise ValueError(
                f"Helion kernel '{final_op_name}' is already registered. "
                f"Use a different op_name or check for duplicate registrations."
            )

        final_fake_impl = fake_impl
        if final_fake_impl is None:
            final_fake_impl = infer_fake_impl(kernel_func, helion_settings)
            logger.debug(
                "Auto-generated fake_impl for Helion kernel '%s'",
                kernel_func.__name__,
            )

        kernel_wrapper = HelionKernelWrapper(
            raw_kernel_func=kernel_func,
            op_name=final_op_name,
            fake_impl=final_fake_impl,
            helion_settings=helion_settings,
        )

        _REGISTERED_KERNELS[final_op_name] = kernel_wrapper

        logger.info(
            "Registered Helion kernel '%s' as HelionKernelWrapper",
            kernel_func.__name__,
        )

        return kernel_wrapper

    if callable(op_name_or_func) and not isinstance(op_name_or_func, str):
        # Bare decorator usage: @register_kernel
        return decorator(op_name_or_func)
    else:
        # Decorator with arguments: @register_kernel(...)
        return decorator
