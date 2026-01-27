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

# %% [markdown] tags=["remove-cell"]
# Getting started with QAT
# ------------------------------

# %% tags=["remove-cell"]
# used to disable output from logs; not shown in the docs because of the
# remove-cell tag
import logging

logging.disable(logging.CRITICAL)

# %% [markdown]
# Before jumping into the deeper workings of QAT, let's run a simple program against an echo engine. This doesn't do much useful execution, but compiling and executing against it is very simple, and fortunately, compiling and executing against actual hardware or useful simulators isn't much more complicated.

# %% [markdown]
# We start by defining in a hardware loader, which is used to load in a hardware model by different means. This particular loader just creates a mock-up of hardware with a ring topology.

# %%
from qat.model.loaders.lucy import LucyModelLoader

loader = LucyModelLoader(qubit_count=8)

# %% [markdown]
# We use the model to constuct an echo pipeline that compiles and executes using an "echo mode"

# %%
from qat.pipelines.waveform import EchoPipeline, PipelineConfig

config = PipelineConfig(name="echo_pipeline")
pipeline = EchoPipeline(loader=loader, config=config)

# %% [markdown]
# We will run a QASM2 program that creates and measures a bell state.

# %%
from compiler_config.config import CompilerConfig, QuantumResultsFormat

qasm_str = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
config = CompilerConfig(results_format=QuantumResultsFormat().binary_count())

# %% [markdown]
# Now we can compile and execute the program against the pipeline

# %%
from qat import QAT

core = QAT()
results, metrics = core.run(qasm_str, compiler_config=config, pipeline=pipeline)
print(results)
