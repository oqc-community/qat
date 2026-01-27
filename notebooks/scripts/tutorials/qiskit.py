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
# Using the Qiskit simulator in new QAT
# ========================================
#
# The Qiskit simulator is available with QAT pipelines, however it is just the legacy version of it wrapped so fit the pipeline API. See below for how it can be used to easily execute programs.

# %% tags=["remove-cell"]
# used to disable output from logs; not shown in the docs because of the
# remove-cell tag
import logging

logging.disable(logging.CRITICAL)

# %% [markdown]
# Let's start by writing a simple program that we'd like to execute.

# %%
from compiler_config.config import CompilerConfig, QuantumResultsFormat

config = CompilerConfig(results_format=QuantumResultsFormat().binary_count())
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

# %% [markdown]
# Now we can create a Qiskit simulator pipeline and execute the program.

# %%
from qat import QAT
from qat.model.loaders.purr.qiskit import QiskitModelLoader
from qat.pipelines.legacy.qiskit import LegacyQiskitPipeline

loader = QiskitModelLoader(qubit_count=20)
pipeline = LegacyQiskitPipeline(config=dict(name="qiskit"), loader=loader)

results, metrics = QAT().run(qasm, pipeline=pipeline, compiler_config=config)
print(results)
