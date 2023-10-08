# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from os.path import abspath, dirname, join
from sys import __loader__

import pytest
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.config import (
    CompilerConfig,
    InlineResultsProcessing,
    MetricsType,
    OptimizationConfig,
    Qasm2Optimizations,
    QiskitOptimizations,
    QuantumResultsFormat,
    ResultsFormatting,
    TketOptimizations,
)
from qat.purr.compiler.instructions import Delay
from qat.qat import execute_with_metrics

from tests.qasm_utils import TestFileType, get_test_file_path


def _get_json_path(file_name):
    return join(
        abspath(join(dirname(__file__), "serialised_compiler_config_templates", file_name))
    )


def _get_contents(file_path):
    """ Get Json from a file. """
    with open(_get_json_path(file_path)) as ifile:
        return ifile.read()


class TestConfigGeneral:
    def test_config_opt_contains(self):
        opt = Qasm2Optimizations()
        assert TketOptimizations.DefaultMappingPass in opt
        assert QiskitOptimizations.Empty in opt

    def test_default_config(self):
        first_conf = CompilerConfig()
        serialized_data = first_conf.to_json()
        second_conf = CompilerConfig.create_from_json(serialized_data)

        assert first_conf.results_format.format == second_conf.results_format.format
        assert first_conf.results_format.transforms == second_conf.results_format.transforms

        conf1_dict = dict(vars(first_conf))
        del conf1_dict['results_format']
        conf2_dict = dict(vars(second_conf))
        del conf2_dict['results_format']

        assert conf1_dict == conf2_dict

    def test_specific_config_optimizations(self):
        first_conf = CompilerConfig()
        first_conf.optimizations = Qasm2Optimizations()
        serialized_data = first_conf.to_json()
        second_conf = CompilerConfig.create_from_json(serialized_data)
        assert first_conf.optimizations.tket_optimizations == \
               second_conf.optimizations.tket_optimizations
        assert first_conf.optimizations.qiskit_optimizations == \
               second_conf.optimizations.qiskit_optimizations

    def test_all_config_optimizations(self):
        def get_subclasses(object):
            subclasses = []

            def find_subclasses(obj):
                for subclass in obj.__subclasses__():
                    subclasses.append(subclass)
                    find_subclasses(subclass)

            find_subclasses(object)

            return subclasses

        optimizations = get_subclasses(OptimizationConfig)

        first_conf = CompilerConfig()

        for optimization in optimizations:
            first_conf.optimizations = optimization()
            serialized_data = first_conf.to_json()
            second_conf = CompilerConfig.create_from_json(serialized_data)

            assert vars(first_conf.optimizations) == vars(second_conf.optimizations)

    def test_config_repeats(self):
        first_conf = CompilerConfig()
        first_conf.repeats = 1000
        first_conf.repetition_period = 10
        serialized_data = first_conf.to_json()
        second_conf = CompilerConfig.create_from_json(serialized_data)

        assert first_conf.repeats == \
               second_conf.repeats
        assert first_conf.repetition_period == \
               second_conf.repetition_period

    def test_config_metrics(self):
        first_conf = CompilerConfig()

        for value in MetricsType:
            first_conf.metrics = value
            serialized_data = first_conf.to_json()
            second_conf = CompilerConfig.create_from_json(serialized_data)

            assert first_conf.metrics == \
                   second_conf.metrics

    def test_config_quantum_results_format(self):
        first_conf = CompilerConfig()

        for format in InlineResultsProcessing:
            for transform in ResultsFormatting:
                first_conf.results_format = QuantumResultsFormat()
                first_conf.results_format.format = format
                first_conf.results_format.transforms = transform
                serialized_data = first_conf.to_json()
                second_conf = CompilerConfig.create_from_json(serialized_data)

                assert first_conf.results_format == \
                       second_conf.results_format

    def test_config_serialisation_raises_error(self):
        class A:
            pass

        first_conf = CompilerConfig()
        first_conf.optimizations = A()  # A is not an allowed type

        with pytest.raises(ValueError):
            first_conf.to_json()

        first_conf.optimizations = Delay  # Not an allowed custom type in project

        with pytest.raises(ValueError):
            first_conf.to_json()

        first_conf.optimizations = __loader__  # Not an allowed type from system module

        with pytest.raises(ValueError):
            first_conf.to_json()

    def test_config_deserialization_raises_error(self):
        serialized_data = str({
            "$type": "<class 'scc.compiler.config.FakeClass'>",
            "$data": {
                "repeats": 1000, "repetition_period": 1000
            }
        })
        with pytest.raises(ValueError):
            CompilerConfig.create_from_json(serialized_data)

    def test_json_version_compatibility(self):
        serialised_data = _get_contents("serialised_default_compiler_config_v2.json")
        deserialised_conf = CompilerConfig.create_from_json(serialised_data)
        assert deserialised_conf.metrics == MetricsType.Default
        assert deserialised_conf.results_format == QuantumResultsFormat()

        serialised_data = _get_contents("serialised_full_compiler_config_v2.json")
        deserialised_conf = CompilerConfig.create_from_json(serialised_data)
        assert deserialised_conf.repeats == 1000
        assert deserialised_conf.repetition_period == 10
        assert deserialised_conf.metrics == MetricsType.OptimizedInstructionCount
        assert deserialised_conf.results_format.format == InlineResultsProcessing.Binary
        assert deserialised_conf.results_format.transforms == ResultsFormatting.DynamicStructureReturn
        assert deserialised_conf.optimizations.qiskit_optimizations == QiskitOptimizations.Empty
        assert deserialised_conf.optimizations.tket_optimizations == TketOptimizations.One

    def test_runs_successfully_with_config(self):
        program = get_test_file_path(TestFileType.QASM2, "ghz.qasm")
        hardware = get_default_echo_hardware()
        serialised_data = _get_contents("serialised_full_compiler_config_v1.json")
        deserialised_conf = CompilerConfig.create_from_json(
            serialised_data
        )  # Test full compiler config v1
        results, metrics = execute_with_metrics(program, hardware, deserialised_conf)
        assert results is not None
        assert metrics is not None
