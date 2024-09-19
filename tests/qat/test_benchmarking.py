# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.runtime import execute_instructions
from qat.purr.utils.benchmarking import randomized_benchmarking


class TestBenchmarking:
    def test_randomized_benchmarking(self):
        model = get_default_echo_hardware()
        results, sequence_lengths = randomized_benchmarking(model, 2)
        benchmarking_results = [
            execute_instructions(model, inst)
            for seed_list in results.values()
            for inst in seed_list
        ]
        assert len(benchmarking_results) == 6

    def test_randomized_benchmarking_lengths(self):
        model = get_default_echo_hardware()
        results, sequence_lengths = randomized_benchmarking(model, 2, lengths=[1, 2])
        benchmarking_results = [
            execute_instructions(model, inst)
            for seed_list in results.values()
            for inst in seed_list
        ]
        assert len(benchmarking_results) == 4
