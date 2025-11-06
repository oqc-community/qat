# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.result_base import ResultManager
from qat.executables import AcquireData, Executable
from qat.ir.measure import AcquireMode
from qat.purr.backends.echo import get_default_echo_hardware
from qat.runtime.passes.purr.analysis import IndexMappingAnalysis, IndexMappingResult


class TestIndexMappingAnalysis:
    model = get_default_echo_hardware(3)

    def create_qat_ir(self):
        model = self.model
        builder = model.create_builder()
        chan1 = model.get_qubit(0).get_acquire_channel()
        chan2 = model.get_qubit(1).get_acquire_channel()
        builder.acquire(chan1, output_variable="test0", delay=0.0)
        builder.acquire(chan1, output_variable="test1", delay=0.0)
        builder.acquire(chan2, output_variable="test2", delay=0.0)
        return builder

    def create_executable(self):
        acquires = {
            f"test{i}": AcquireData(
                mode=AcquireMode.INTEGRATOR,
                shape=(1000,),
                physical_channel=self.model.qubits[i].measure_device.physical_channel.id,
            )
            for i in range(3)
        }

        return Executable(programs=[], acquires=acquires)

    def test_var_to_physical_channel_executable(self):
        package = self.create_executable()
        print(package.acquires)
        mapping = IndexMappingAnalysis.var_to_physical_channel_executable(package)
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
            assert (
                mapping[f"test{i}"]
                == self.model.get_qubit(i).measure_device.physical_channel.id
            )

    def test_var_to_physical_channel_qat_ir(self):
        ir = self.create_qat_ir()
        chan1 = self.model.get_qubit(0).get_acquire_channel()
        chan2 = self.model.get_qubit(1).get_acquire_channel()

        mapping = IndexMappingAnalysis.var_to_physical_channel_qat_ir(ir)
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert mapping["test0"] == chan1.physical_channel_id
        assert mapping["test1"] == chan1.physical_channel_id
        assert mapping["test2"] == chan2.physical_channel_id

    def test_var_to_qubit_map(self):
        ir = self.create_qat_ir()

        _pass = IndexMappingAnalysis(self.model)
        mapping = _pass.var_to_qubit_map(_pass.var_to_physical_channel_qat_ir(ir))
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert mapping["test0"] == 0
        assert mapping["test1"] == 0
        assert mapping["test2"] == 1

    def test_full_pass_ir(self):
        ir = self.create_qat_ir()
        _pass = IndexMappingAnalysis(self.model)
        res_mgr = ResultManager()
        _pass.run(None, res_mgr, package=ir)
        mapping = res_mgr.lookup_by_type(IndexMappingResult).mapping
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert mapping["test0"] == 0
        assert mapping["test1"] == 0
        assert mapping["test2"] == 1

    def test_full_pass_executable(self):
        package = self.create_executable()
        _pass = IndexMappingAnalysis(self.model)
        res_mgr = ResultManager()
        _pass.run(None, res_mgr, package=package)
        mapping = res_mgr.lookup_by_type(IndexMappingResult).mapping
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
            assert mapping[f"test{i}"] == i
