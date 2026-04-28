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
# # Reproducibility and Randomness
#
# This notebook focuses on the user-facing reproducibility and randomness improvements.
#
# Key points:
# - More reproducible generated models via consistent optional seed propagation.
# - Deterministic hardware setup options for Echo and realtime simulator helpers.
# - Previously implicit default seeds were removed from core generation paths; `get_default_<x>_hardware` helpers are the exception.
# - No workflow change required for normal usage (defaults still work as before).

# %%
from qat.model.loaders.converted import EchoModelLoader, JaggedEchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.backends.echo import get_default_echo_hardware
from qat.utils.hardware_model import generate_connectivity_data, generate_hw_model

# %% [markdown]
# ## 1) More reproducible generated models
#
# Core generation paths now accept/propagate optional seeds, improving repeatability for:
# - generated connectivity and quality maps,
# - generated target data,
# - generated physical hardware models.

# %%
seed = 77

conn_a, logical_a, quality_a = generate_connectivity_data(6, 4, seed=seed)
conn_b, logical_b, quality_b = generate_connectivity_data(6, 4, seed=seed)
conn_c, logical_c, quality_c = generate_connectivity_data(6, 4, seed=seed + 1)

print("Connectivity reproducible for same seed:", conn_a == conn_b)
print("Logical map reproducible for same seed:", logical_a == logical_b)
print("Quality map reproducible for same seed:", quality_a == quality_b)
print("Connectivity usually differs for different seed:", conn_a != conn_c)

td_a = TargetData.random(seed=seed)
td_b = TargetData.random(seed=seed)
print("TargetData.random reproducible for same seed:", td_a == td_b)

hw_a = generate_hw_model(4, seed=seed)
hw_b = generate_hw_model(4, seed=seed)
print(
    "generate_hw_model reproducible for same seed:", hw_a.model_dump() == hw_b.model_dump()
)

# %% [markdown]
# ## 2) Deterministic hardware setup options
#
# Echo and realtime simulator setup helpers now support explicit seed parameters.
# This allows deterministic setup when reproducing integration behavior or debugging specific runs.

# %%
echo_1 = get_default_echo_hardware(qubit_count=4, add_direction_couplings=True, seed=11)
echo_2 = get_default_echo_hardware(qubit_count=4, add_direction_couplings=True, seed=11)
echo_3 = get_default_echo_hardware(qubit_count=4, add_direction_couplings=True, seed=12)

qualities_1 = [c.quality for c in echo_1.qubit_direction_couplings]
qualities_2 = [c.quality for c in echo_2.qubit_direction_couplings]
qualities_3 = [c.quality for c in echo_3.qubit_direction_couplings]

print("Echo helper deterministic for same seed:", qualities_1 == qualities_2)
print("Echo helper different for different seeds:", qualities_1 != qualities_3)

# %% [markdown]
# ## 3) No workflow change required for normal usage
#
# You can keep existing unseeded calls.
# Core generation paths no longer rely on previously implicit default seeds,
# while `get_default_<x>_hardware` helper functions intentionally retain a default seed.
# Deterministic behavior is now easier to request explicitly when needed.

# %%
# Existing style: omit seed and keep default behavior
echo_default_a = EchoModelLoader(qubit_count=4).load()
echo_default_b = EchoModelLoader(qubit_count=4).load()
jagged_default = JaggedEchoModelLoader(qubit_count=6).load()
target_default = TargetData.random()

print(
    "Default Echo loader still works:",
    echo_default_a is not None and echo_default_b is not None,
)
print("Default Jagged loader still works:", jagged_default is not None)
print("Default TargetData.random still works:", target_default is not None)

# Exception: get_default_<x>_hardware helpers still keep a default seed
echo_default_seed = get_default_echo_hardware(qubit_count=4, add_direction_couplings=True)
echo_explicit_42 = get_default_echo_hardware(
    qubit_count=4, add_direction_couplings=True, seed=42
)
q_default = [c.quality for c in echo_default_seed.qubit_direction_couplings]
q_seed_42 = [c.quality for c in echo_explicit_42.qubit_direction_couplings]
print(
    "get_default_<x>_hardware default seed retained (matches seed=42):",
    q_default == q_seed_42,
)

# Explicit deterministic style for reproducibility-sensitive workflows
echo_seeded_a = EchoModelLoader(qubit_count=4, random_seed=123).load()
echo_seeded_b = EchoModelLoader(qubit_count=4, random_seed=123).load()
print(
    "Explicit seed gives deterministic Echo loader output:",
    echo_seeded_a.model_dump() == echo_seeded_b.model_dump(),
)

# %% [markdown]
# ## Summary
#
# The reproducibility update improves control without forcing a migration:
# - Keep existing unseeded usage for normal workflows.
# - Previously implicit default seeds were removed from most generation paths.
# - `get_default_<x>_hardware` helpers remain the exception and retain a default seed.
# - Add explicit seeds when you need deterministic model generation or hardware setup.
# - This is especially useful for integration pipelines and issue reproduction.
