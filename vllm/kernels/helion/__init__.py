# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion integration for vLLM."""

import os as _os
import time as _time

_pkg_t0 = _time.perf_counter()
import vllm.kernels.helion.ops  # noqa: E402, F401  Auto-register all Helion ops

_pkg_dt = _time.perf_counter() - _pkg_t0
_mf = _os.environ.get("HELION_INSTRUMENT_FILE")
if _mf:
    with open(_mf, "a") as _f:
        _f.write(f"helion_package_init {_pkg_dt:.3f}s\n")
from vllm.kernels.helion.case_key import CaseKey  # noqa: E402
from vllm.kernels.helion.config_manager import (  # noqa: E402
    ConfigManager,
    ConfigSet,
)
from vllm.kernels.helion.ir_ops import register_as_simple_vllm_ir_impl  # noqa: E402
from vllm.kernels.helion.register import (  # noqa: E402
    ConfigPicker,
    ConfiguredHelionKernel,
    HelionKernelWrapper,
    get_kernel_by_name,
    get_registered_kernels,
    register_kernel,
    vllm_helion_lib,
)
from vllm.kernels.helion.utils import (  # noqa: E402
    canonicalize_gpu_name,
    get_canonical_gpu_name,
)

__all__ = [
    # Config management
    "CaseKey",
    "ConfigManager",
    "ConfigSet",
    # Kernel registration
    "ConfigPicker",
    "ConfiguredHelionKernel",
    "HelionKernelWrapper",
    "get_kernel_by_name",
    "get_registered_kernels",
    "register_as_simple_vllm_ir_impl",
    "register_kernel",
    "vllm_helion_lib",
    # Utilities
    "canonicalize_gpu_name",
    "get_canonical_gpu_name",
]
