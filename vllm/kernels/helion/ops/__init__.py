# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-import all Helion op modules to trigger kernel registration."""

import importlib
import os
import pkgutil
import time

_instrument = os.environ.get("HELION_INSTRUMENT_FILE")
_t0_all = time.perf_counter()
for _module_info in pkgutil.iter_modules(__path__):
    _t0 = time.perf_counter()
    importlib.import_module(f"{__name__}.{_module_info.name}")
    _dt = time.perf_counter() - _t0
    if _instrument:
        with open(_instrument, "a") as _f:
            _f.write(f"op_import {_module_info.name} {_dt:.3f}s\n")
_dt_all = time.perf_counter() - _t0_all
if _instrument:
    with open(_instrument, "a") as _f:
        _f.write(f"all_ops_import {_dt_all:.3f}s\n")
