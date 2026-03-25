# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # TargetData Demo
#
# This notebook demonstrates how to use and configure `TargetData` in QAT, including:
# - default construction
# - nested qubit/resonator overrides
# - validation behaviour
# - random generation (seeded and unseeded)
# - serialisation and YAML loading
# - migration from deprecated helpers

# %%
import tempfile
import warnings
from pathlib import Path

from pydantic import ValidationError

from qat.core.config.descriptions import CompilePipelineDescription
from qat.model.target_data import (
    CustomTargetData,
    DefaultTargetData,
    QubitDescription,
    ResonatorDescription,
    TargetData,
)

warnings.simplefilter("always", DeprecationWarning)

# %% [markdown]
# ## 1) Start With Defaults
#
# `TargetData()` now directly creates a complete default instance and is the preferred entry point for new code.

# %%
target_data = TargetData()

print("max_acquisitions:", target_data.max_acquisitions)
print("default_shots:", target_data.default_shots)
print("clock_cycle:", target_data.clock_cycle)
print("qubit passive_reset_time:", target_data.QUBIT_DATA.passive_reset_time)
print("qubit sample_time:", target_data.QUBIT_DATA.sample_time)
print("resonator sample_time:", target_data.RESONATOR_DATA.sample_time)

print("\nTop-level model keys:")
print(sorted(target_data.model_dump().keys()))

# %% [markdown]
# ## 2) Customise TargetData
#
# You can override top-level and nested values directly when constructing `TargetData`.

# %%
custom_target_data = TargetData(
    max_acquisitions=20_000,
    default_shots=2_048,
    QUBIT_DATA=QubitDescription(
        passive_reset_time=5e-4,
        instruction_memory_size=60_000,
    ),
    RESONATOR_DATA=ResonatorDescription(
        instruction_memory_size=70_000,
    ),
)

print("custom default_shots:", custom_target_data.default_shots)
print("custom passive_reset_time:", custom_target_data.QUBIT_DATA.passive_reset_time)
print(
    "custom qubit instruction_memory_size:",
    custom_target_data.QUBIT_DATA.instruction_memory_size,
)
print(
    "custom resonator instruction_memory_size:",
    custom_target_data.RESONATOR_DATA.instruction_memory_size,
)

# %% [markdown]
# ## 3) Validation Behaviour
#
# `TargetData` enforces consistency and strict value constraints. The examples below show two common validation failures.

# %%
# Invalid 1: pulse_duration_min > pulse_duration_max
try:
    _ = QubitDescription(pulse_duration_min=2e-6, pulse_duration_max=1e-6)
except ValidationError as exc:
    print("Invalid duration config caught:")
    print(exc)

# Invalid 2: incompatible clock cycles between qubit and resonator
try:
    _ = TargetData(
        QUBIT_DATA=QubitDescription(sample_time=1e-9, samples_per_clock_cycle=2),
        RESONATOR_DATA=ResonatorDescription(sample_time=2e-9, samples_per_clock_cycle=2),
    )
except ValidationError as exc:
    print("\nClock-cycle mismatch caught:")
    print(exc)

# %% [markdown]
# ## 4) Random TargetData
#
# `TargetData.random()` supports both patterns:
# - without a seed: convenient random examples
# - with a seed: deterministic examples for testing and debugging

# %%
# Without a seed: values may differ across calls
random_td_unseeded_1 = TargetData.random()
random_td_unseeded_2 = TargetData.random()

print("Unseeded call 1 clock_cycle:", random_td_unseeded_1.clock_cycle)
print("Unseeded call 2 clock_cycle:", random_td_unseeded_2.clock_cycle)
print(
    "Unseeded reproducible:",
    random_td_unseeded_1.model_dump() == random_td_unseeded_2.model_dump(),
)

# With a seed: deterministic across calls
random_td_seeded_1 = TargetData.random(seed=123)
random_td_seeded_2 = TargetData.random(seed=123)

print("\nSeeded call 1 clock_cycle:", random_td_seeded_1.clock_cycle)
print("Seeded call 2 clock_cycle:", random_td_seeded_2.clock_cycle)
print(
    "Seeded reproducible:",
    random_td_seeded_1.model_dump() == random_td_seeded_2.model_dump(),
)

compile_desc = CompilePipelineDescription(name="demo")
print("\nCompilePipelineDescription.target_data default:")
print(compile_desc.target_data)

# %% [markdown]
# ## 5) Serialisation, YAML Loading, and Deprecation Migration
#
# This section shows how to serialise/deserialise `TargetData` and how legacy helpers map to the preferred constructor style.

# %%
# JSON serialisation
payload_json = target_data.model_dump_json(indent=2)
print("Serialised JSON (first 250 chars):")
print(payload_json[:250] + "...")

# YAML loading from disk via TargetData.from_yaml(...)
yaml_text = """
max_acquisitions: 12000
default_shots: 512
QUBIT_DATA:
  sample_time: 1e-09
  samples_per_clock_cycle: 1
  instruction_memory_size: 50000
  waveform_memory_size: 1500
  pulse_duration_min: 6.4e-08
  pulse_duration_max: 0.001
  pulse_channel_lo_freq_min: 1000000
  pulse_channel_lo_freq_max: 10000000000
  pulse_channel_if_freq_min: 0
  pulse_channel_if_freq_max: 10000000000
  passive_reset_time: 0.001
RESONATOR_DATA:
  sample_time: 1e-09
  samples_per_clock_cycle: 1
  instruction_memory_size: 50000
  waveform_memory_size: 1500
  pulse_duration_min: 6.4e-08
  pulse_duration_max: 0.001
  pulse_channel_lo_freq_min: 1000000
  pulse_channel_lo_freq_max: 10000000000
  pulse_channel_if_freq_min: 0
  pulse_channel_if_freq_max: 10000000000
""".strip()

with tempfile.TemporaryDirectory() as tmp_dir:
    yaml_path = Path(tmp_dir) / "target_data.yaml"
    yaml_path.write_text(yaml_text)
    loaded_td = TargetData.from_yaml(yaml_path)

print("\nLoaded from YAML default_shots:", loaded_td.default_shots)

# Deprecated helpers still work, but emit DeprecationWarning.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    _ = TargetData.default()
    _ = TargetData.create_with(passive_reset_time=2e-3)
    _ = DefaultTargetData()
    _ = CustomTargetData(passive_reset_time=3e-3)

dep_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
print("\nDeprecation warnings captured:", len(dep_msgs))
for msg in dep_msgs:
    print("-", msg)

# %% [markdown]
# ## Summary
#
# For new code, use `TargetData(...)` directly.
#
# Migration quick reference:
# - `TargetData.default()` -> `TargetData()`
# - `TargetData.create_with(...)` -> `TargetData(...)`
# - `DefaultTargetData()` -> `TargetData()`
# - `CustomTargetData(...)` -> `TargetData(...)`
