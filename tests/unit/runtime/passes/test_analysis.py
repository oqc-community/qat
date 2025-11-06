# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.core.result_base import ResultManager
from qat.executables import AcquireData, Executable
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.measure import AcquireMode
from qat.model.loaders.lucy import LucyModelLoader
from qat.runtime.passes.analysis import IndexMappingAnalysis, IndexMappingResult


class TestIndexMappingAnalysis:
    @pytest.fixture(scope="class")
    def model(self):
        return LucyModelLoader(4).load()

    @pytest.fixture(scope="class")
    def ir(self, model):
        builder = QuantumInstructionBuilder(hardware_model=model)
        qubit1 = model.qubit_with_index(0)
        qubit2 = model.qubit_with_index(1)
        builder.acquire(qubit1, output_variable="test0", delay=0.0)
        builder.acquire(qubit1, output_variable="test1", delay=0.0)
        builder.acquire(qubit2, output_variable="test2", delay=0.0)
        return builder

    @pytest.fixture(scope="class")
    def executable(self, model):
        acquires = {
            f"test{i}": AcquireData(
                mode=AcquireMode.INTEGRATOR,
                shape=(1000,),
                physical_channel=model.qubit_with_index(i).resonator.physical_channel.uuid,
            )
            for i in range(3)
        }

        return Executable(programs=[], acquires=acquires)

    def test_var_to_physical_channel_executable(self, model, executable):
        mapping = IndexMappingAnalysis.var_to_physical_channel_executable(executable)
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
            assert (
                mapping[f"test{i}"]
                == model.qubit_with_index(i).resonator.physical_channel.uuid
            )

    def test_var_to_physical_channel_qat_ir(self, model, ir):
        phys_ch1_id = model.qubit_with_index(0).resonator.physical_channel.uuid
        phys_ch2_id = model.qubit_with_index(1).resonator.physical_channel.uuid

        mapping = IndexMappingAnalysis(model).var_to_physical_channel_qat_ir(ir)
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert mapping["test0"] == phys_ch1_id
        assert mapping["test1"] == phys_ch1_id
        assert mapping["test2"] == phys_ch2_id

    def test_var_to_qubit_map(self, model, ir):
        _pass = IndexMappingAnalysis(model)
        mapping = _pass.var_to_qubit_map(_pass.var_to_physical_channel_qat_ir(ir))
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert mapping["test0"] == 0
        assert mapping["test1"] == 0
        assert mapping["test2"] == 1

    def test_full_pass_ir(self, model, ir):
        _pass = IndexMappingAnalysis(model)
        res_mgr = ResultManager()
        _pass.run(None, res_mgr, package=ir)
        mapping = res_mgr.lookup_by_type(IndexMappingResult).mapping
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert mapping["test0"] == 0
        assert mapping["test1"] == 0
        assert mapping["test2"] == 1

    def test_full_pass_executable(self, model, executable):
        _pass = IndexMappingAnalysis(model)
        res_mgr = ResultManager()
        _pass.run(None, res_mgr, package=executable)
        mapping = res_mgr.lookup_by_type(IndexMappingResult).mapping
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
            assert mapping[f"test{i}"] == i
