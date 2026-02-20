# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

from pydantic import Field, field_validator

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


@config
class KernelConfig:
    """Configuration for kernel selection and warmup behavior."""

    enable_flashinfer_autotune: bool = Field(default=None)
    """If True, run FlashInfer autotuning during kernel warmup."""

    helion_platform: str | None = Field(default=None)
    """Override auto-detected GPU platform name for Helion kernel config
    lookup. When set, Helion kernels will use this platform name instead of
    auto-detecting from the GPU hardware. Use the canonical form, e.g.
    'nvidia_h100', 'nvidia_a100'."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = [self.helion_platform]
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("enable_flashinfer_autotune", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialization is delayed."""
        if value is None:
            return value
        return handler(value)

    @field_validator("helion_platform", mode="before")
    @classmethod
    def _validate_helion_platform(cls, value: Any) -> str | None:
        if value is None:
            return None
        # Avoid top-level import to prevent circular import:
        # vllm.config → vllm.kernels.helion.__init__ → vllm.config
        from vllm.kernels.helion.utils import canonicalize_gpu_name
        return canonicalize_gpu_name(str(value))
