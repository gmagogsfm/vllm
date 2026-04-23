# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "silu_mul_fp8 Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_silu_mul_fp8_inputs() -> dict[str, tuple[Any, ...]]:
    intermediate_sizes = [2048, 2880, 4096, 8192, 11008, 14336]

    # Use the same num_tokens values as vLLM's default cudagraph capture sizes.
    # See vllm/config/vllm.py _set_cudagraph_sizes() for the canonical formula.
    num_tokens_list = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))

    inputs = {}
    for num_tokens in num_tokens_list:
        for intermediate_size in intermediate_sizes:
            input_tensor = torch.randn(
                num_tokens,
                2 * intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

            config_key = f"intermediate_{intermediate_size}_numtokens_{num_tokens}"
            inputs[config_key] = (input_tensor, scale)

    return inputs


_parsed_silu_configs_cache: dict[int, dict[int, list[int]]] = {}
_pick_silu_result_cache: dict[tuple[int, int], str | None] = {}


def _get_parsed_silu_configs(config_keys: list[str]) -> dict[int, list[int]]:
    cache_key = id(config_keys)
    if cache_key in _parsed_silu_configs_cache:
        return _parsed_silu_configs_cache[cache_key]

    configs: dict[int, list[int]] = {}
    prefix = "intermediate_"
    mid = "_numtokens_"
    for key in config_keys:
        if key == "default":
            continue
        try:
            rest = key[len(prefix) :]
            idx = rest.index(mid)
            isize = int(rest[:idx])
            ntokens = int(rest[idx + len(mid) :])
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'intermediate_{{int}}_numtokens_{{int}}'"
            ) from e
        configs.setdefault(isize, []).append(ntokens)

    for isize in configs:
        configs[isize].sort()

    _parsed_silu_configs_cache[cache_key] = configs
    return configs


def pick_silu_mul_fp8_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    if not config_keys:
        return None

    input_tensor, _scale = args
    intermediate_size = input_tensor.shape[-1] // 2
    num_tokens = input_tensor.view(-1, input_tensor.shape[-1]).shape[0]

    shape_key = (int(num_tokens), int(intermediate_size))
    cached = _pick_silu_result_cache.get(shape_key)
    if cached is not None:
        return cached

    configs = _get_parsed_silu_configs(config_keys)

    if not configs:
        result = "default" if "default" in config_keys else None
        if result is not None:
            _pick_silu_result_cache[shape_key] = result
        return result

    best_isize = min(configs, key=lambda s: abs(s - intermediate_size))
    available_ntokens = configs[best_isize]
    best_ntokens = next(
        (n for n in available_ntokens if n >= num_tokens), available_ntokens[-1]
    )

    result = f"intermediate_{best_isize}_numtokens_{best_ntokens}"
    _pick_silu_result_cache[shape_key] = result
    return result


@register_kernel(
    config_picker=pick_silu_mul_fp8_config,
    input_generator=generate_silu_mul_fp8_inputs,
)
def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    original_shape = input.shape
    two_d = hl.specialize(original_shape[-1])
    d = two_d // 2
    output_shape = original_shape[:-1] + (d,)

    input_2d = input.view(-1, original_shape[-1])
    m = input_2d.shape[0]

    # TODO(gmagogsfm): Support for more float8 subtypes (e4m3fnuz, e5m2) coming
    out = torch.empty((m, d), device=input.device, dtype=torch.float8_e4m3fn)

    input_part_a = input_2d[:, :d]
    input_part_b = input_2d[:, d:]

    assert scale.numel() == 1, "Scale must be a scalar Tensor"

    for tile_m, tile_n in hl.tile([m, d]):
        a_vals = input_part_a[tile_m, tile_n]
        silu_result = torch.nn.functional.silu(a_vals)
        b_vals = input_part_b[tile_m, tile_n]
        result = silu_result * b_vals
        result_f32 = result.to(torch.float32)
        scale_val = hl.load(scale, [0])
        inv_scale = 1.0 / scale_val
        result_scaled = result_f32 * inv_scale
        out[tile_m, tile_n] = result_scaled.to(out.dtype)

    return out.view(output_shape)


def silu_mul_fp8_baseline(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    output_shape = input.shape[:-1] + (input.shape[-1] // 2,)
    out = torch.empty(output_shape, dtype=torch.float8_e4m3fn, device=input.device)
    torch.ops._C.silu_and_mul_quant(out, input, scale)
    return out
