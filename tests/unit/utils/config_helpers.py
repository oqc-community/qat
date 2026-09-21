# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from pathlib import Path

EXPLICIT_FULL_PIPELINE_CONFIG = """\
PIPELINES:
- name: explicit-full
  pipeline: qat.pipelines.echo.echo8
  default: true
"""

DEFAULT_SEPARATE_PIPELINES_CONFIG = """\
HARDWARE:
- name: echo6loader
  type: qat.model.loaders.purr.EchoModelLoader
  config:
    qubit_count: 6

COMPILE:
- name: default-compile
  pipeline: qat.pipelines.waveform.WaveformCompilePipeline
  hardware_loader: echo6loader
  default: true

EXECUTE:
- name: default-execute
  pipeline: qat.pipelines.waveform.EchoExecutePipeline
  hardware_loader: echo6loader
  default: true
"""


def write_full_and_separate_default_configs(tmp_path: Path) -> Path:
    explicit_config = tmp_path / "customconfig.yaml"
    explicit_config.write_text(EXPLICIT_FULL_PIPELINE_CONFIG)
    (tmp_path / "qatconfig.yaml").write_text(DEFAULT_SEPARATE_PIPELINES_CONFIG)
    return explicit_config
