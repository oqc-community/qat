# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# # Qblox dummy Q1ASM assembly
#
# Does a dummy Cluster assemble Q1ASM or only store the sequence dictionary?
# This experiment uploads a valid program, an unknown instruction and an invalid
# register to dummy QCM-RF and QRM-RF modules through `qblox_instruments.Cluster`.
# No sequencer is armed or started. Cleanup uses QAT's transport helper.
#
# On 16 November 2026, all six uploads matched expectations with each of
# qblox-instruments 1.3.1 and 1.2.2 on Linux. The valid program assembled, while
# the unknown instruction and register `R64` produced assembly errors.
# Rerun the cells to check the installed driver version.

# %%
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import json
import platform
from contextlib import chdir, redirect_stdout
from importlib.metadata import version
from io import StringIO
from tempfile import TemporaryDirectory

from qblox_instruments import Cluster, ClusterType

from qat.engines.qblox.transport import close_cluster

# %% [markdown]
# ## Upload and collect assembler diagnostics
#
# The driver runs its bundled platform-specific assembler and writes fixed-name
# files in the working directory. Run uploads serially in a temporary directory
# so the files do not collide with another experiment.
#
# A rejected upload alone is not proof of assembly. Check the module's assembler
# status and log too, and use a valid upload as the control.


# %%
def probe_dummy_assembly() -> dict:
    """Upload valid and invalid Q1ASM to dummy QCM-RF and QRM-RF modules.

    Close the Cluster and remove assembler files before returning. Unexpected exceptions
    propagate after cleanup.

    :returns: Driver version, environment details and the six upload outcomes, including
        assembler statuses, logs and any upload errors.
    """
    programs = {
        "valid": "    move 1,R0\n    nop\n    stop\n",
        "unknown_instruction": "    not_an_instruction\n    stop\n",
        "invalid_register": "    move 1,R64\n    nop\n    stop\n",
    }
    results = []

    # The driver's assembler uses fixed filenames in the working directory.
    with TemporaryDirectory(prefix="qat-dummy-assembly-") as workdir, chdir(workdir):
        cluster = Cluster(
            name="dummy_assembly_probe",
            dummy_cfg={
                1: ClusterType.CLUSTER_QCM_RF,
                13: ClusterType.CLUSTER_QRM_RF,
            },
        )
        try:
            cluster.reset()
            for slot, module in cluster.get_connected_modules().items():
                sequencer = module.sequencers[0]
                for case, program in programs.items():
                    error = None
                    upload_output = StringIO()
                    with redirect_stdout(upload_output):
                        try:
                            sequencer.sequence(
                                {
                                    "waveforms": {},
                                    "weights": {},
                                    "acquisitions": {},
                                    "program": program,
                                }
                            )
                        except RuntimeError as exc:
                            error = str(exc)
                    assembler_status = module.get_assembler_status()
                    assembler_log = module.get_assembler_log()
                    expected_acceptance = case == "valid"
                    matches_expectation = (
                        error is None
                        and assembler_status
                        and "assembler finished successfully" in assembler_log.lower()
                        if expected_acceptance
                        else error is not None
                        and not assembler_status
                        and "assembler failed" in assembler_log.lower()
                    )
                    results.append(
                        {
                            "module": (
                                f"{module.module_type}{'-RF' if module.is_rf_type else ''}"
                            ),
                            "slot": slot,
                            "sequencer": 0,
                            "case": case,
                            "program": program,
                            "expected_acceptance": expected_acceptance,
                            "accepted": error is None,
                            "error": error,
                            "upload_stdout": upload_output.getvalue(),
                            "assembler_status": assembler_status,
                            "assembler_log": assembler_log,
                            "matches_expectation": matches_expectation,
                        }
                    )
        finally:
            close_cluster(cluster)

    return {
        "qblox_instruments_version": version("qblox-instruments"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "results": results,
    }


# %% [markdown]
# ## Run the experiment
#
# The report retains the source programs and diagnostics so a changed driver
# response can be inspected. A failed expectation raises rather than presenting
# the experiment as successful.

# %%
report = probe_dummy_assembly()
print(json.dumps(report, indent=2))
assert len(report["results"]) == 6
assert all(result["matches_expectation"] for result in report["results"]), (
    "Dummy assembly behaviour differs from the expected outcomes. Inspect the report."
)

# %% [markdown]
# ## What the result proves
#
# A successful valid upload and assembler failures for the invalid programs show
# that the dummy runs an assembler for these inputs and this driver version.
# QAT's upload path reaches the same `sequencer.sequence()` API, but this notebook
# tests the driver boundary, not QAT's wrapper.
#
# Assembly is not evidence of physical execution, quantum correctness, timing
# behaviour or compatibility with a different deployed firmware version.
# Keep schema validation separate and rerun the experiment after driver upgrades.
