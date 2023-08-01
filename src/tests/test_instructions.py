# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import pytest
from qat.purr.backends.echo import EchoEngine, get_default_echo_hardware
from qat.purr.compiler.devices import (
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    PulseShapeType,
)
from qat.purr.compiler.execution import SweepIterator
from qat.purr.compiler.instructions import Pulse, Sweep, SweepValue
from qat.purr.compiler.runtime import get_builder


class TestInstruction:
    def test_name_assignment(self):
        builder = get_builder(get_default_echo_hardware())
        label1 = builder.create_label()
        label2 = builder.create_label()
        assert label1.name != label2.name
        assert label1.name in builder.existing_names
        assert label2.name in builder.existing_names

    def test_nested_sweep_iterator(self):
        sweep_iter = SweepIterator(
            Sweep(SweepValue("dave", [1, 2, 3, 4, 5])),
            SweepIterator(
                Sweep(SweepValue("dave", [1, 2, 3])),
                SweepIterator(Sweep(SweepValue("dave", [1, 2, 3, 4, 5, 6, 7, 8])))
            )
        )
        incrementor = 0
        while not sweep_iter.is_finished():
            sweep_iter.do_sweep([])
            incrementor += 1

        # Test that actual cycles are both equal the accumulated values, as well as the
        # length
        assert incrementor == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == 120

    def test_sweep_iterator(self):
        sweep_iter = SweepIterator(Sweep(SweepValue("dave", [1, 2, 3, 4, 5])))
        incrementor = 0
        while not sweep_iter.is_finished():
            sweep_iter.do_sweep([])
            incrementor += 1

        # Test that actual cycles are both equal the accumulated values, as well as the
        # length
        assert incrementor == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == 5

    def test_instruction_limit(self):
        qie = EchoEngine()
        with pytest.raises(ValueError):
            qie.validate([
                Pulse(
                    PulseChannel("", PhysicalChannel("", 1, PhysicalBaseband("", 1))),
                    PulseShapeType.SQUARE,
                    0
                ) for _ in range(201000)
            ])
