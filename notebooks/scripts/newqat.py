# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
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

# %%
from qat import QAT

# %%
prog = """
OPENQASM 3;
bit[2] c;
qubit[2] q;
h q;
measure q -> c;
"""

# %%
qat = QAT("../qatconfig.eg.yaml")
qat.set_default_pipeline("echo")

# %%
inst, metrics = qat.compile(prog)
res, metrics = qat.execute(inst)
res

# %%
qat.set_default_pipeline("rtcs")
inst, metrics = qat.compile(prog)
res, metrics = qat.execute(inst)
res

# %%
inst, metrics = qat.compile(prog, pipeline="echo")
res, metrics = qat.execute(inst, pipeline="echo")
res

# %%
from qat.pipelines import DefaultCompile, DefaultExecute, DefaultPostProcessing
from qat.purr.backends.echo import get_default_echo_hardware

echo16 = get_default_echo_hardware(qubit_count=16)

# %%
qat.add_pipeline(
    "echo16",
    compile_pipeline=DefaultCompile(echo16),
    execute_pipeline=DefaultExecute(echo16),
    postprocess_pipeline=DefaultPostProcessing(echo16),
    engine=echo16.create_engine(),
)

# %%
inst, metrics = qat.compile(prog, pipeline="echo16")
res, metrics = qat.execute(inst, pipeline="echo16")
res
