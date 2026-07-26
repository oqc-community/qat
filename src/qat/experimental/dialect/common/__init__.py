# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Target-agnostic IR utilities shared across the experimental dialects.

Control-flow analysis, SSA value resolution, region surgery, and name allocation
that carry no dialect knowledge. Dialects supply their specifics through the
:class:`~qat.experimental.dialect.common.cfg.SuccessorOperandsTrait` rather than
by exposing concrete operation types to the algorithms.
"""
