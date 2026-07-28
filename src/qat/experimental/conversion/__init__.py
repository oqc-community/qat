# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Inter-dialect conversion passes for the QAT experimental IR stack.

Each sub-package implements a conversion from one dialect to another. A conversion
pass takes IR expressed in dialect A and rewrites it into IR expressed in dialect B.
Neither dialect owns the conversion logic, so it does not belong under either
``dialect/`` tree. This mirrors the convention established by MLIR's
``mlir/lib/Conversion`` namespace.

Intra-dialect transforms (canonicalisation, optimisation, or analysis passes that
operate entirely within one dialect) live under
``dialect/<name>/transforms/`` and must not be placed here. Non-IR concerns such
as hardware configuration, system data, and frontend parsing are equally out of
scope for this package.
"""
