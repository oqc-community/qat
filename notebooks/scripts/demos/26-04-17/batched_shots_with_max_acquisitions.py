# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Shot Batching with Mid-Circuit Measurements
#
# This Notebook focuses on highlighting user facing changes to the BatchedShots pass and associated TargetData nomenclature changes. These changes were implemented in preparation for post-selection.
#
# **Key Points:**
#
# - Shot batching logic now accounts for number of Acquire instructions explicitly rather than assuming single measurement per qubit per shot.
# - `max_acquisitions` replaces `max_shots` in TargetData.
# - Deprecation of `max_shots`.
# - Backward compatibility ensured.

# %% [markdown]
#
# ## 1) QASM2 Circuit With a Mid-Circuit Measurement

# %%
qasm2_str = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[6];
creg c0[1];  // first measurement result (single qubit)
creg c1[6];  // second measurement results (post-Hadamard)

// Apply Hadamard to all qubits
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];

// First measurement (q[0] only)
measure q[0] -> c0[0];

// Apply Hadamard to all qubits
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];

// Second measurement
measure q -> c1;
"""

# %% [markdown]
# ## 2) New Parameter: max_acquisitions

# %%
from qat.model.target_data import TargetData

TargetData(max_acquisitions=2)

# %% [markdown]
# ## 3) Modified IR pass to account for Mid-Circuit measurements
#
# Previously Circuits such as the one shown above could have caused sequencer memory issues, due to improper shot batching.
# Now batching is based on number of Acquire instructions.

# %%
from compiler_config.config import CompilerConfig, QuantumResultsFormat, Tket

from qat.model.loaders.lucy import LucyModelLoader
from qat.pipelines.waveform import EchoPipeline as PydanticEchoPipeline, PipelineConfig

SHOTS = 2


pipeline = PydanticEchoPipeline(
    config=PipelineConfig(name="pydantic"),
    model=LucyModelLoader(qubit_count=6).load(),
    target_data=TargetData(max_acquisitions=2),
)

from qat.core.qat import QAT

compiler_config = CompilerConfig(
    repeats=SHOTS,
    results_format=QuantumResultsFormat().binary_count(),
    optimizations=Tket().disable(),
)
_ = QAT().compile(qasm2_str, compiler_config, pipeline=pipeline)[0]

# %% [markdown]
# ## 4) Backwards Compatibility
# You can still use max_shots (deprecated)

# %%
target_data = TargetData(max_shots=2)

# %%
print(target_data.max_acquisitions == target_data.max_shots)

# %% [markdown]
# ## Summary
#
# Groundwork in place for post-selection:
# - keep deprecated max_shots parameter with backward compatibility.
# - Introduced new TargetData nomenclature with more targeted name.
# - Shot batching logic can now account for multiple measurements on the same qubit, within a single shot.
