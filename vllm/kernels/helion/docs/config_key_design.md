# Helion Config Key Redesign

## Problem

The original Helion kernel registration system used freeform strings as config keys (e.g., `"intermediate_2048_numtokens_256"`). Each kernel's `pick_config` function parsed these strings with `re.fullmatch()` on every call. During CUDA graph capture with many batch sizes, this resulted in ~47K calls × 300 regex matches each, consuming ~200s of startup time.

## Design

### CaseKey

`CaseKey` is an immutable, hashable `dict` subclass that identifies a kernel config/autotune/benchmark case. It lives in `vllm/kernels/helion/case_key.py`.

```python
from vllm.kernels.helion.case_key import CaseKey

# Construction
key = CaseKey({"intermediate": 2048, "numtokens": 256})
default = CaseKey.default()  # empty dict, fallback config

# Dict-style access
key["intermediate"]  # 2048

# Hashable — works as dict key
configs = {key: helion.Config(...), CaseKey.default(): helion.Config(...)}

# Stable string form (sorted JSON)
str(key)  # '{"intermediate":2048,"numtokens":256}'

# Immutable
key["x"] = 1  # TypeError: CaseKey is immutable

# Check for default
key.is_default()  # False
CaseKey.default().is_default()  # True

# Empty construction is an error
CaseKey()  # TypeError: use CaseKey.default()
```

### Config file format

Config files use a JSON array of `{"key": ..., "config": ...}` entries:

```json
[
  {"key": {}, "config": {"block_sizes": [64, 32], "num_warps": 4}},
  {"key": {"intermediate": 2048, "numtokens": 256}, "config": {"block_sizes": [128, 64]}}
]
```

The empty dict `{}` represents the default config.

### config_picker signature

```python
ConfigPicker = Callable[[tuple[Any, ...], list[CaseKey]], CaseKey | None]
```

The framework deserializes keys once on load and passes `CaseKey` instances to the picker. The picker receives all keys including default, and returns `CaseKey | None` (`None` = fall back to default).

### Kernel author API

```python
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.register import register_kernel

def generate_inputs():
    inputs = {}
    for size in [2048, 4096]:
        key = CaseKey({"intermediate": size, "numtokens": 256})
        inputs[key] = (tensor, scale)
    return inputs

def pick_config(args, config_keys):
    # config_keys is list[CaseKey] — dict access, no parsing
    for k in config_keys:
        if k.is_default():
            continue
        if k["intermediate"] == target:
            return k
    return None  # framework uses default

@register_kernel(config_picker=pick_config, input_generator=generate_inputs)
def my_kernel(...): ...
```

### Internal flow

1. JSON file loaded → `ConfigSet.from_dict()` wraps each key dict in `CaseKey`
2. `ConfiguredHelionKernel._create_key_computer()` builds `list[CaseKey]` from configs once
3. On each kernel call, `key_computer` calls `picker(args, all_keys)` → `CaseKey | None`
4. `str(selected)` produces a stable string for Helion's internal cache
5. `_create_config_selector` maps `str → CaseKey` back to look up the `helion.Config`

## Pros

- **No parsing on the hot path.** Config keys are deserialized once on load. The picker receives pre-parsed dicts and does direct `k["param"]` access. Benchmark shows 43× faster than regex without caching, 739× with per-shape caching.

- **Values can be any JSON-serializable type.** Unlike the old `name_int` string format, keys can contain lists, tuples, floats, nested structures — e.g., `CaseKey({"split": [1, 2], "intermediate": 2048})`.

- **Uniform type everywhere.** `CaseKey` is used as dict keys in configs, return values from pickers, keys in `input_generator` output, and keys in JSON files. No `str | dict` unions. Default is `CaseKey.default()`, not `None` or `"default"`.

- **Immutable and hashable.** Safe to use as dict keys and in sets. Mutation raises `TypeError`. Hash is stable (based on sorted JSON).

- **Self-documenting construction.** `CaseKey.default()` is clearer than `CaseKey()` or `None`. Empty construction raises `TypeError` with a helpful message.

- **Clean file format.** The JSON array of `{"key": ..., "config": ...}` entries is readable and supports arbitrary key types without escaping.

## Cons

- **str ↔ CaseKey bridge in register.py.** Helion's `key` hook must return a string, so the framework calls `str(selected)` and maintains a `str → CaseKey` reverse lookup. This is an internal complexity that exists because Helion's API expects strings.

- **`CaseKey` is a dict subclass with overridden mutation methods.** The `# type: ignore[assignment]` comments on `__setitem__`, `__delitem__`, etc. are ugly. Python doesn't have a built-in frozen dict, so this is the least-bad option that preserves dict-style `k["param"]` access.

- **Hash computed via JSON serialization.** `hash(str(self))` involves `json.dumps` on every hash call. For the small dicts used as config keys (2-5 entries), this is negligible, but it's not zero-cost. A tuple-based hash would be faster but would require more implementation work.

- **Two ways to construct.** `CaseKey({"a": 1})` (dict arg) and `CaseKey(a=1)` (kwargs) both work due to dict inheritance. The codebase uses the dict form for consistency with lookup syntax, but kwargs still compile without error.

- **Breaking change for existing kernel authors.** The `config_picker` signature changed from `(args, list[str]) → str | None` to `(args, list[CaseKey]) → CaseKey | None`. Existing pickers need updating. The config file format also changed, requiring migration of JSON files.

- **`is_default()` check in pickers.** Since the picker now receives all keys including default, kernel authors who iterate over keys need `if k.is_default(): continue` to skip the empty default key. This is a minor footgun — forgetting the check causes a `KeyError` on `k["param"]`.

## Performance

Benchmark: 80,000 calls to `pick_config` with 300 config keys, 8 different input shapes. Measured on NVIDIA H100.

| Approach | us/call | Speedup |
| --- | --- | --- |
| Old (regex, no cache) | 1289 | 1.0× |
| New (CaseKey, no cache) | 46 | 28× |
| New (CaseKey, cached) | 1.8 | 719× |

The regex approach spends ~1.3ms per call parsing 300 key strings. `CaseKey` dict access is 28× faster even without caching. With per-shape result caching (the realistic path after the first call per shape), it's 719× faster.

In an end-to-end vLLM startup test with `max_cudagraph_capture_size=8192` (512 batch sizes, compile cache disabled):

| Configuration | Startup time |
| --- | --- |
| Helion ON, old regex | 259s |
| Helion ON, CaseKey | 107s |
| Helion OFF | 88s |

The CaseKey approach closes most of the gap between Helion-enabled and Helion-disabled startup.
