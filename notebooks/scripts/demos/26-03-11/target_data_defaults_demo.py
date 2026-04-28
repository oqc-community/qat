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
# # TargetData Defaults Demo
#
# This notebook showcases the `TargetData` modernisation introduced in the latest updates.
#
# Main changes:
# - `TargetData()` now has practical defaults for device, qubit, and resonator fields.
# - Legacy helpers (`.default()`, `.create_with()`, `DefaultTargetData`, `CustomTargetData`) are deprecated in favour of direct construction.
# - Pipeline config defaults now reference `qat.model.target_data.TargetData`.

# %%
import warnings

from qat.model.target_data import (
    CustomTargetData,
    DefaultTargetData,
    QubitDescription,
    ResonatorDescription,
    TargetData,
)

warnings.simplefilter("always", DeprecationWarning)

# %% [markdown]
# ## 1) Zero-argument `TargetData()` now works
#
# Before this change, callers frequently used factory helpers. Now a direct constructor gives a complete, valid default instance.

# %%
target_data = TargetData()

print("max_acquisitions:", target_data.max_acquisitions)
print("default_shots:", target_data.default_shots)
print("qubit passive_reset_time:", target_data.QUBIT_DATA.passive_reset_time)
print("qubit instruction_memory_size:", target_data.QUBIT_DATA.instruction_memory_size)
print(
    "resonator instruction_memory_size:", target_data.RESONATOR_DATA.instruction_memory_size
)
print("clock_cycle:", target_data.clock_cycle)

# %% [markdown]
# ## 2) Legacy constructors are still available but deprecated
#
# The commit keeps backward compatibility while guiding callers towards `TargetData(...)`.

# %%
# These calls should emit DeprecationWarning and still construct TargetData objects.
td_default = TargetData.default()
td_create_with = TargetData.create_with(passive_reset_time=2e-3)
td_fn_default = DefaultTargetData()
td_fn_custom = CustomTargetData(passive_reset_time=3e-3)

print(type(td_default).__name__, td_default.default_shots)
print(type(td_create_with).__name__, td_create_with.QUBIT_DATA.passive_reset_time)
print(type(td_fn_default).__name__)
print(type(td_fn_custom).__name__, td_fn_custom.QUBIT_DATA.passive_reset_time)

# %% [markdown]
# ## 3) Preferred customisation: construct directly
#
# The modern style is to pass keyword arguments directly to `TargetData(...)`, including nested `QUBIT_DATA` and `RESONATOR_DATA` overrides.

# %%
custom_target_data = TargetData(
    default_shots=2048,
    QUBIT_DATA=QubitDescription(passive_reset_time=5e-4),
    RESONATOR_DATA=ResonatorDescription(),
)

print("custom default_shots:", custom_target_data.default_shots)
print("custom passive_reset_time:", custom_target_data.QUBIT_DATA.passive_reset_time)

# %% [markdown]
# ## 4) Related config default points to `TargetData`
#
# The pipeline description defaults now reference `qat.model.target_data.TargetData`, matching the direct-construction approach.

# %%
from qat.core.config.descriptions import CompilePipelineDescription

compile_desc = CompilePipelineDescription(name="demo")
print(compile_desc.target_data)

# %% [markdown]
# ## Summary
#
# For new code, use `TargetData(...)` directly.
#
# Migration pattern:
# - `TargetData.default()` -> `TargetData()`
# - `TargetData.create_with(...)` -> `TargetData(...)`
# - `DefaultTargetData()` -> `TargetData()`
# - `CustomTargetData(...)` -> `TargetData(...)`
