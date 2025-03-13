# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---

# %% [markdown]
# #### New QAT frontdoor getting started (EXPERIMENTAL)
#
# This is a demo of the new QAT 'frontdoor' which uses the experimental QAT pipelines.
#
# We aren't using this in production yet and your mileage may vary.

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# Import QAT and instantiate a qat instance

# %%
from qat import QAT

qat = QAT()

# %% [markdown]
# Default pipelines are setup by default...

# %%
qat.pipelines

# %% [markdown]
# Define a program...

# %%
src = """
OPENQASM 3;
bit[2] c;
qubit[2] q;
h q;
measure q -> c;
"""

# %% [markdown]
# Compile it with the default pipeline...

# %%
pkg, metrics = qat.compile(src)

# %% [markdown]
# Execute the compiled program with the default pipeline...

# %%
res, metrics = qat.execute(pkg)
res

# %% [markdown]
# Comile and execute a program with a different pipeline...

# %%
pkg, metrics = qat.compile(src, pipeline="echo16")
res, metrics = qat.execute(pkg, pipeline="echo16")

# %% [markdown]
# QAT pipelines can also be configured with YAML...

# %%
qat = QAT(qatconfig="../qatconfig.eg.yaml")
qat.pipelines

# %% [markdown]
# The yaml looks like this... (It's not very flexible yet it will be)

# %%
import pathlib
import yaml


print(yaml.dump(yaml.safe_load(pathlib.Path("../qatconfig.eg.yaml").read_text())))

# %% [markdown]
# Change the default pipeline

# %%
qat.pipelines.set_default("echo8-alt")
inst, metrics = qat.compile(src)
res, metrics = qat.execute(inst)
res

# %% [markdown]
# Run a program as a one liner...

# %%
res, metrics = qat.run(src)
res

# %% [markdown]
# Make a custom pipeline

# %%
from qat import Pipeline

# %%
from qat.backend.waveform_v1.codegen import WaveformV1Backend
from qat.engines.waveform_v1 import EchoEngine
from qat.frontend import AutoFrontend
from qat.middleend.middleends import DefaultMiddleend
from qat.purr.backends.echo import get_default_echo_hardware
from qat.runtime.runtimes.simple import SimpleRuntime

model = get_default_echo_hardware(qubit_count=16)
P = Pipeline(
    name="mycoolnewpipeline",
    frontend=AutoFrontend(model),
    middleend=DefaultMiddleend(model),
    backend=WaveformV1Backend(model),
    runtime=SimpleRuntime(engine=EchoEngine()),
    model=model,
)


# %% [markdown]
# Compile and execute against the new pipeline

# %%
pkg, metrics = qat.compile(src, pipeline=P)
res, metrics = qat.execute(pkg, pipeline=P)
res

# %% [markdown]
# Keep it around for later...

# %%
qat.pipelines.add(P)
P.name

# %% [markdown]
# Now it's available by name

# %%
pkg, metrics = qat.compile(src, pipeline="mycoolnewpipeline")
res, metrics = qat.execute(pkg, pipeline="mycoolnewpipeline")
res
