# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Helion kernel registration.

Tests ConfiguredHelionKernel, HelionKernelWrapper, and PresetConfigSearch
including config picker registration, custom autotuner integration, and
PyTorch op registration.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

import helion.language as hl

from vllm.kernels.helion.register import (
    _REGISTERED_KERNELS,
    ConfiguredHelionKernel,
    HelionKernelWrapper,
    PresetConfigSearch,
    get_kernel_by_name,
    get_registered_kernels,
    register_kernel,
    validate_helion_settings,
)


@pytest.fixture
def mock_wrapper():
    """Create a mock HelionKernelWrapper."""
    wrapper = Mock(spec=HelionKernelWrapper)
    wrapper.op_name = "test_kernel"
    wrapper.raw_kernel_func = Mock()
    wrapper.helion_settings = None
    wrapper._fake_impl = Mock()
    wrapper._config_picker = None
    return wrapper


@pytest.fixture
def configured_kernel(mock_wrapper):
    """Create a ConfiguredHelionKernel for testing."""
    mock_model_config = Mock()
    configs = {
        "hiddensize_4096_batchsize_32": Mock(),
        "hiddensize_4096_batchsize_64": Mock(),
        "hiddensize_4096_batchsize_128": Mock(),
        "default": Mock(),
    }
    return ConfiguredHelionKernel(
        wrapper=mock_wrapper,
        platform="nvidia_h200",
        model_config=mock_model_config,
        configs=configs,
    )


class TestPresetConfigSearch:
    """Test suite for PresetConfigSearch custom autotuner."""

    def test_init_stores_args_and_selector(self):
        """Test that PresetConfigSearch stores args and selector."""
        args = (torch.randn(32, 128), 1.0)
        selector = Mock()

        autotuner = PresetConfigSearch(args, selector)

        assert autotuner.args is args
        assert autotuner.config_selector is selector

    def test_autotune_calls_config_selector(self):
        """Test that autotune calls the config selector with args."""
        args = (torch.randn(32, 128), 1.0)
        mock_config = Mock()
        selector = Mock(return_value=mock_config)

        autotuner = PresetConfigSearch(args, selector)
        result = autotuner.autotune()

        selector.assert_called_once_with(args)
        assert result is mock_config


class TestValidateHelionSettings:
    """Test suite for validate_helion_settings utility function."""

    def test_accepts_none_settings(self):
        """Test that None settings are accepted without error."""
        validate_helion_settings(None, "test_kernel")  # Should not raise

    def test_accepts_valid_settings(self):
        """Test that valid settings without conflicts are accepted."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"debug": True, "static_shapes": False}
        validate_helion_settings(mock_settings, "test_kernel")  # Should not raise

    def test_rejects_autotuner_fn(self):
        """Test that settings with autotuner_fn raise ValueError."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"autotuner_fn": Mock()}

        with pytest.raises(ValueError, match="uses a custom autotuner"):
            validate_helion_settings(mock_settings, "test_kernel")

    def test_rejects_custom_key(self):
        """Test that settings with custom_key raise ValueError."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"custom_key": lambda *args: "key"}

        with pytest.raises(ValueError, match="uses a custom key function"):
            validate_helion_settings(mock_settings, "test_kernel")

    def test_warns_on_static_shapes_true(self):
        """Test that static_shapes=True emits a warning."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"static_shapes": True}

        with patch("vllm.kernels.helion.register.logger") as mock_logger:
            validate_helion_settings(mock_settings, "test_kernel")
            mock_logger.warning.assert_called_once()
            assert "static_shapes=True" in mock_logger.warning.call_args[0][0]


class TestConfiguredHelionKernel:
    """Test suite for ConfiguredHelionKernel."""

    def test_create_key_computer_raises_without_picker(self, configured_kernel):
        """Test that _create_key_computer raises when no picker registered."""
        with pytest.raises(RuntimeError, match="No config picker registered"):
            configured_kernel._create_key_computer()

    def test_key_computer_fallback_to_default_on_none(self, configured_kernel):
        """Test that key_computer falls back to 'default' when picker returns None."""
        configured_kernel.wrapper._config_picker = Mock(return_value=None)

        key_computer = configured_kernel._create_key_computer()
        result = key_computer(torch.randn(32, 4096))

        assert result == "default"

    def test_config_selector_validates_picker_result(self, configured_kernel):
        """Test that config selector validates picker returns valid key."""
        configured_kernel.wrapper._config_picker = Mock(return_value="invalid_key")

        key_computer = configured_kernel._create_key_computer()
        selector = configured_kernel._create_config_selector(key_computer)

        with pytest.raises(
            ValueError, match="Config picker returned invalid config key"
        ):
            selector((torch.randn(32, 4096),))

    def test_config_selector_handles_none_from_picker(self, configured_kernel):
        """Test that config selector falls back to 'default' on None."""
        configured_kernel.wrapper._config_picker = Mock(return_value=None)

        key_computer = configured_kernel._create_key_computer()
        selector = configured_kernel._create_config_selector(key_computer)

        result = selector((torch.randn(32, 4096),))
        assert result is configured_kernel.configs["default"]

    def test_get_decorated_kernel_caches_results(self, configured_kernel):
        """Test that _get_decorated_kernel caches decorated kernels."""
        configured_kernel.wrapper._config_picker = Mock(return_value="default")

        mock_decorated = Mock()

        with patch("vllm.kernels.helion.register.helion") as mock_helion:
            mock_helion.kernel.return_value = Mock(return_value=mock_decorated)

            result1 = configured_kernel._get_decorated_kernel()
            result2 = configured_kernel._get_decorated_kernel()

            assert result1 is mock_decorated
            assert result2 is mock_decorated
            assert mock_helion.kernel.call_count == 1

    def test_get_decorated_kernel_passes_helion_settings(self, configured_kernel):
        """Test that _get_decorated_kernel passes helion_settings."""
        configured_kernel.wrapper._config_picker = Mock(return_value="default")

        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"debug": True}
        configured_kernel.wrapper.helion_settings = mock_settings

        with patch("vllm.kernels.helion.register.helion") as mock_helion:
            mock_helion.kernel.return_value = Mock(return_value=Mock())

            configured_kernel._get_decorated_kernel()

            call_kwargs = mock_helion.kernel.call_args[1]
            assert "debug" in call_kwargs
            assert call_kwargs["debug"] is True
            # static_shapes should be set to False by default
            assert call_kwargs["static_shapes"] is False

    def test_get_decorated_kernel_sets_static_shapes_false_by_default(
        self, configured_kernel
    ):
        """Test that _get_decorated_kernel sets static_shapes=False by default."""
        configured_kernel.wrapper._config_picker = Mock(return_value="default")

        with patch("vllm.kernels.helion.register.helion") as mock_helion:
            mock_helion.kernel.return_value = Mock(return_value=Mock())

            configured_kernel._get_decorated_kernel()

            call_kwargs = mock_helion.kernel.call_args[1]
            assert call_kwargs["static_shapes"] is False

    def test_get_decorated_kernel_preserves_static_shapes_true(self, configured_kernel):
        """Test that explicit static_shapes=True is preserved."""
        configured_kernel.wrapper._config_picker = Mock(return_value="default")

        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"static_shapes": True}
        configured_kernel.wrapper.helion_settings = mock_settings

        with patch("vllm.kernels.helion.register.helion") as mock_helion:
            mock_helion.kernel.return_value = Mock(return_value=Mock())

            configured_kernel._get_decorated_kernel()

            call_kwargs = mock_helion.kernel.call_args[1]
            assert call_kwargs["static_shapes"] is True

    def test_custom_key_and_config_selector_use_same_logic(self, configured_kernel):
        """Test that custom_key and config_selector produce identical results."""

        def tracking_picker(model_config, args, config_keys):
            x = args[0]
            batch_size = x.shape[0]
            if batch_size <= 32:
                return "hiddensize_4096_batchsize_32"
            elif batch_size <= 64:
                return "hiddensize_4096_batchsize_64"
            return "hiddensize_4096_batchsize_128"

        configured_kernel.wrapper._config_picker = tracking_picker

        with patch("vllm.kernels.helion.register.helion.kernel") as mock_helion_kernel:
            mock_decorated = Mock()
            mock_helion_kernel.return_value = Mock(return_value=mock_decorated)

            configured_kernel._get_decorated_kernel()

            call_kwargs = mock_helion_kernel.call_args[1]
            custom_key_fn = call_kwargs["custom_key"]
            autotuner_fn = call_kwargs["autotuner_fn"]

            tensor = torch.randn(50, 4096)  # batch=50, should select batchsize_64

            # custom_key receives unpacked args, autotuner receives args as tuple
            key_result = custom_key_fn(tensor)
            autotuner = autotuner_fn(None, (tensor,))
            config = autotuner.autotune()

            assert key_result == "hiddensize_4096_batchsize_64"
            assert config is configured_kernel.configs["hiddensize_4096_batchsize_64"]


class TestHelionKernelWrapper:
    """Test suite for HelionKernelWrapper."""

    def test_call_raises_error(self):
        """Test that calling wrapper directly raises RuntimeError."""
        wrapper = HelionKernelWrapper(
            raw_kernel_func=Mock(),
            op_name="test_kernel",
            fake_impl=Mock(),
        )

        with pytest.raises(RuntimeError, match="should not be called directly"):
            wrapper("arg")

    def test_register_config_picker(self):
        """Test register_config_picker stores and returns picker function."""
        wrapper = HelionKernelWrapper(
            raw_kernel_func=Mock(),
            op_name="test_kernel",
            fake_impl=Mock(),
        )

        def my_picker(model_config, args, config_keys):
            return "default"

        result = wrapper.register_config_picker(my_picker)

        assert wrapper._config_picker is my_picker
        assert result is my_picker

    def test_init_errors_on_conflicting_autotuner_fn(self):
        """Test that conflicting autotuner_fn in helion_settings raises ValueError."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {
            "debug": True,
            "autotuner_fn": Mock(),
        }

        with pytest.raises(
            ValueError, match="uses a custom autotuner via config picker"
        ):
            HelionKernelWrapper(
                raw_kernel_func=Mock(),
                op_name="test_kernel",
                fake_impl=Mock(),
                helion_settings=mock_settings,
            )

    def test_init_errors_on_conflicting_custom_key(self):
        """Test that conflicting custom_key in helion_settings raises ValueError."""
        mock_settings = Mock()
        mock_settings.to_dict.return_value = {
            "debug": True,
            "custom_key": lambda *args: "some_key",
        }

        with pytest.raises(
            ValueError, match="uses a custom key function derived from config picker"
        ):
            HelionKernelWrapper(
                raw_kernel_func=Mock(),
                op_name="test_kernel",
                fake_impl=Mock(),
                helion_settings=mock_settings,
            )

    def test_create_configured_op_validates_inputs(self):
        """Test create_configured_op_from_model validates required inputs."""
        wrapper = HelionKernelWrapper(
            raw_kernel_func=Mock(),
            op_name="test_kernel",
            fake_impl=Mock(),
        )

        with pytest.raises(AssertionError, match="requires model_config"):
            wrapper.create_configured_op_from_model(None, Mock())

        with pytest.raises(AssertionError, match="requires config_manager"):
            wrapper.create_configured_op_from_model(Mock(), None)

    def test_create_configured_op_validates_configs_available(self):
        """Test create_configured_op_from_model validates configs are available."""
        wrapper = HelionKernelWrapper(
            raw_kernel_func=Mock(),
            op_name="test_kernel",
            fake_impl=Mock(),
        )

        config_manager = Mock()
        config_manager.get_platform_configs = Mock(return_value={})

        with pytest.raises(AssertionError, match="No configs available"):
            wrapper.create_configured_op_from_model(Mock(), config_manager)

    def test_create_configured_op_validates_platform_configs(self):
        """Test create_configured_op_from_model validates platform has configs."""
        wrapper = HelionKernelWrapper(
            raw_kernel_func=Mock(),
            op_name="test_kernel",
            fake_impl=Mock(),
        )

        config_manager = Mock()
        config_manager.get_platform_configs = Mock(return_value={})

        with (
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            pytest.raises(AssertionError, match="No configs available.*platform"),
        ):
            wrapper.create_configured_op_from_model(Mock(), config_manager)

    def test_create_configured_op_validates_config_picker(self):
        """Test create_configured_op_from_model validates config picker."""
        wrapper = HelionKernelWrapper(
            raw_kernel_func=Mock(),
            op_name="test_kernel",
            fake_impl=Mock(),
        )

        config_manager = Mock()
        config_manager.get_platform_configs = Mock(return_value={"default": Mock()})

        with (
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            pytest.raises(AssertionError, match="No config picker registered"),
        ):
            wrapper.create_configured_op_from_model(Mock(), config_manager)

    def test_create_configured_op_returns_existing_op(self):
        """Test create_configured_op_from_model returns existing op."""
        wrapper = HelionKernelWrapper(
            raw_kernel_func=Mock(),
            op_name="test_kernel",
            fake_impl=Mock(),
        )
        wrapper._config_picker = Mock()

        config_manager = Mock()
        config_manager.get_platform_configs = Mock(return_value={"default": Mock()})

        existing_op = Mock()
        mock_namespace = Mock()
        mock_namespace.test_kernel_nvidia_h200 = existing_op

        with (
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch.object(torch.ops, "vllm_helion", mock_namespace),
        ):
            result = wrapper.create_configured_op_from_model(Mock(), config_manager)
            assert result is existing_op

    def test_create_configured_op_registers_new_op(self):
        """Test create_configured_op_from_model creates and registers new op."""
        wrapper = HelionKernelWrapper(
            raw_kernel_func=Mock(),
            op_name="test_kernel",
            fake_impl=Mock(),
        )
        wrapper._config_picker = Mock()

        config_manager = Mock()
        config_manager.get_platform_configs = Mock(return_value={"default": Mock()})

        new_op = Mock()
        registered_ops: dict[str, Mock] = {}

        class MockNamespace:
            def __getattr__(self, name):
                if name in registered_ops:
                    return registered_ops[name]
                raise AttributeError(name)

        mock_namespace = MockNamespace()

        def register_side_effect(op_name, op_func, **kwargs):
            registered_ops[op_name] = new_op

        with (
            patch(
                "vllm.kernels.helion.utils.get_canonical_gpu_name",
                return_value="nvidia_h200",
            ),
            patch.object(torch.ops, "vllm_helion", mock_namespace),
            patch(
                "vllm.kernels.helion.register.direct_register_custom_op",
                side_effect=register_side_effect,
            ) as mock_register,
        ):
            result = wrapper.create_configured_op_from_model(Mock(), config_manager)

            mock_register.assert_called_once()
            assert result is new_op
            assert isinstance(
                mock_register.call_args[1]["op_func"], ConfiguredHelionKernel
            )


class TestKernelRegistry:
    """Test suite for kernel registry functions."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Clean up registered kernels before and after each test."""
        # Store original registry state
        original_kernels = _REGISTERED_KERNELS.copy()
        _REGISTERED_KERNELS.clear()
        yield
        # Restore original registry state
        _REGISTERED_KERNELS.clear()
        _REGISTERED_KERNELS.update(original_kernels)

    def test_get_registered_kernels_returns_copy(self):
        """Test that get_registered_kernels returns a copy of the registry."""
        mock_wrapper = Mock(spec=HelionKernelWrapper)
        _REGISTERED_KERNELS["test_kernel"] = mock_wrapper

        result = get_registered_kernels()

        assert result == {"test_kernel": mock_wrapper}
        # Verify it's a copy, not the original
        result["another_kernel"] = Mock()
        assert "another_kernel" not in _REGISTERED_KERNELS

    def test_get_kernel_by_name_returns_kernel(self):
        """Test that get_kernel_by_name returns the correct kernel."""
        mock_wrapper = Mock(spec=HelionKernelWrapper)
        _REGISTERED_KERNELS["test_kernel"] = mock_wrapper

        result = get_kernel_by_name("test_kernel")

        assert result is mock_wrapper

    def test_get_kernel_by_name_returns_none_for_missing(self):
        """Test that get_kernel_by_name returns None for missing kernel."""
        result = get_kernel_by_name("nonexistent_kernel")

        assert result is None


class TestRegisterKernel:
    """Test suite for register_kernel decorator."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Clean up registered kernels before and after each test."""
        original_kernels = _REGISTERED_KERNELS.copy()
        _REGISTERED_KERNELS.clear()
        yield
        _REGISTERED_KERNELS.clear()
        _REGISTERED_KERNELS.update(original_kernels)

    def test_register_kernel_auto_generates_fake_impl(self):
        """Test that register_kernel auto-generates fake_impl when not provided."""

        @register_kernel("silu_mul_fp8_test")
        def silu_mul_fp8_test(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            """Simplified silu_mul_fp8 kernel for testing."""
            d = input.shape[-1] // 2
            output_shape = input.shape[:-1] + (d,)
            out = torch.empty(
                output_shape, device=input.device, dtype=torch.float8_e4m3fn
            )

            input_part_a = input[..., :d]
            input_part_b = input[..., d:]

            for tile_idx in hl.tile(out.shape):
                a_vals = input_part_a[tile_idx].to(torch.float32)
                silu_result = a_vals * torch.sigmoid(a_vals)
                silu_result = silu_result.to(input.dtype)
                b_vals = input_part_b[tile_idx]
                result = silu_result * b_vals
                result_f32 = result.to(torch.float32)
                scale_val = hl.load(scale, [0])
                result_scaled = result_f32 * (1.0 / scale_val)
                out[tile_idx] = result_scaled.to(out.dtype)

            return out

        assert isinstance(silu_mul_fp8_test, HelionKernelWrapper)
        assert silu_mul_fp8_test._fake_impl is not None
        assert callable(silu_mul_fp8_test._fake_impl)

        # Verify auto-generated fake_impl produces correct output shape and dtype
        batch_size = 32
        hidden_size = 4096
        input_tensor = torch.randn(
            batch_size, 2 * hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        fake_output = silu_mul_fp8_test._fake_impl(input_tensor, scale)

        expected_shape = (batch_size, hidden_size)
        assert fake_output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {fake_output.shape}"
        )
        assert fake_output.dtype == torch.float8_e4m3fn, (
            f"Expected dtype float8_e4m3fn, got {fake_output.dtype}"
        )
        assert fake_output.device == input_tensor.device

    def test_register_kernel_creates_wrapper(self):
        """Test that register_kernel creates a HelionKernelWrapper."""

        def fake_impl(x):
            return torch.empty_like(x)

        @register_kernel("test_kernel", fake_impl=fake_impl)
        def test_kernel(x):
            return x

        assert isinstance(test_kernel, HelionKernelWrapper)
        assert test_kernel.op_name == "test_kernel"
        assert test_kernel._fake_impl is fake_impl

    def test_register_kernel_auto_detects_name(self):
        """Test that register_kernel auto-detects op name from function name."""

        def fake_impl(x):
            return torch.empty_like(x)

        @register_kernel(fake_impl=fake_impl)
        def my_kernel_func(x):
            return x

        assert my_kernel_func.op_name == "my_kernel_func"

    def test_register_kernel_registers_in_global_registry(self):
        """Test that register_kernel adds kernel to global registry."""

        def fake_impl(x):
            return torch.empty_like(x)

        @register_kernel("test_kernel", fake_impl=fake_impl)
        def test_kernel(x):
            return x

        assert "test_kernel" in _REGISTERED_KERNELS
        assert _REGISTERED_KERNELS["test_kernel"] is test_kernel

    def test_register_kernel_passes_helion_settings(self):
        """Test that register_kernel passes helion_settings to wrapper."""

        def fake_impl(x):
            return torch.empty_like(x)

        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"debug": True}

        @register_kernel(
            "test_kernel", fake_impl=fake_impl, helion_settings=mock_settings
        )
        def test_kernel(x):
            return x

        assert test_kernel.helion_settings is mock_settings

    def test_register_kernel_supports_decorator_syntax(self):
        """Test that register_kernel works with all decorator syntaxes."""

        def fake_impl(x):
            return torch.empty_like(x)

        # With explicit name
        @register_kernel("explicit_name", fake_impl=fake_impl)
        def kernel1(x):
            return x

        # With parentheses but without explicit name
        @register_kernel(fake_impl=fake_impl)
        def kernel2(x):
            return x

        assert kernel1.op_name == "explicit_name"
        assert kernel2.op_name == "kernel2"

    def test_register_kernel_bare_decorator(self):
        """Test that register_kernel works without parentheses."""

        # Bare decorator usage: @register_kernel without parentheses
        @register_kernel
        def bare_kernel(x):
            return x

        assert isinstance(bare_kernel, HelionKernelWrapper)
        assert bare_kernel.op_name == "bare_kernel"
        assert bare_kernel._fake_impl is not None

    def test_registered_wrapper_can_register_config_picker(self):
        """Test that registered wrapper can use register_config_picker."""

        def fake_impl(x):
            return torch.empty_like(x)

        @register_kernel("test_kernel", fake_impl=fake_impl)
        def test_kernel(x):
            return x

        @test_kernel.register_config_picker
        def pick_config(model_config, args, config_keys):
            return "default"

        assert test_kernel._config_picker is pick_config

    def test_register_kernel_raises_on_duplicate_registration(self):
        """Test that register_kernel raises error for duplicate registrations."""

        def fake_impl(x):
            return torch.empty_like(x)

        @register_kernel("duplicate_test", fake_impl=fake_impl)
        def kernel1(x):
            return x

        with pytest.raises(ValueError, match="already registered"):

            @register_kernel("duplicate_test", fake_impl=fake_impl)
            def kernel2(x):
                return x * 2

    def test_register_kernel_rejects_autotuner_fn_in_settings(self):
        """Test that register_kernel rejects helion_settings with autotuner_fn."""

        def fake_impl(x):
            return torch.empty_like(x)

        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"autotuner_fn": Mock()}

        with pytest.raises(ValueError, match="uses a custom autotuner"):

            @register_kernel(
                "test_kernel", fake_impl=fake_impl, helion_settings=mock_settings
            )
            def test_kernel(x):
                return x

    def test_register_kernel_rejects_custom_key_in_settings(self):
        """Test that register_kernel rejects helion_settings with custom_key."""

        def fake_impl(x):
            return torch.empty_like(x)

        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"custom_key": lambda *args: "key"}

        with pytest.raises(ValueError, match="uses a custom key function"):

            @register_kernel(
                "test_kernel", fake_impl=fake_impl, helion_settings=mock_settings
            )
            def test_kernel(x):
                return x

    def test_register_kernel_warns_with_static_shapes_true(self):
        """Test that register_kernel warns when static_shapes=True."""

        def fake_impl(x):
            return torch.empty_like(x)

        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"static_shapes": True}

        with patch("vllm.kernels.helion.register.logger") as mock_logger:

            @register_kernel(
                "test_kernel", fake_impl=fake_impl, helion_settings=mock_settings
            )
            def test_kernel(x):
                return x

            # Verify warning was called with static_shapes message
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "static_shapes=True" in warning_msg

    def test_register_kernel_no_warning_with_static_shapes_false(self):
        """Test that register_kernel doesn't warn when static_shapes=False."""

        def fake_impl(x):
            return torch.empty_like(x)

        mock_settings = Mock()
        mock_settings.to_dict.return_value = {"static_shapes": False}

        with patch("vllm.kernels.helion.register.logger") as mock_logger:

            @register_kernel(
                "test_kernel", fake_impl=fake_impl, helion_settings=mock_settings
            )
            def test_kernel(x):
                return x

            # Verify warning was NOT called
            mock_logger.warning.assert_not_called()
