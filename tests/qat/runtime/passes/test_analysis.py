# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.result_base import ResultManager
from qat.ir.measure import AcquireMode
from qat.purr.backends.echo import get_default_echo_hardware
from qat.runtime.executables import AcquireDataStruct, ChannelData, Executable
from qat.runtime.passes.analysis import IndexMappingAnalysis, IndexMappingResult


class TestIndexMappingAnalysis:

    def create_qat_ir(self):
        model = get_default_echo_hardware(2)
        builder = model.create_builder()
        chan1 = model.get_qubit(0).get_acquire_channel()
        chan2 = model.get_qubit(1).get_acquire_channel()
        builder.acquire(chan1, output_variable="test0", delay=0.0)
        builder.acquire(chan1, output_variable="test1", delay=0.0)
        builder.acquire(chan2, output_variable="test2", delay=0.0)
        return model, builder

    def create_executable(self):
        model = get_default_echo_hardware(2)
        acquires = [
            AcquireDataStruct(
                length=1,
                position=0,
                mode=AcquireMode.INTEGRATOR,
                output_variable=f"test{i}",
            )
            for i in range(3)
        ]

        channel1 = ChannelData(acquires=[acquires[0], acquires[1]])
        channel2 = ChannelData(acquires=[acquires[2]])
        channel_ids = [
            model.get_qubit(i).measure_device.physical_channel.full_id() for i in range(2)
        ]
        package = Executable(
            channel_data={channel_ids[0]: channel1, channel_ids[1]: channel2}
        )
        return model, package

    def test_var_to_physical_channel_executable(self):
        model, package = self.create_executable()
        mapping = IndexMappingAnalysis.var_to_physical_channel_executable(package)
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert (
            mapping["test0"] == model.get_qubit(0).get_acquire_channel().physical_channel_id
        )
        assert (
            mapping["test1"] == model.get_qubit(0).get_acquire_channel().physical_channel_id
        )
        assert (
            mapping["test2"] == model.get_qubit(1).get_acquire_channel().physical_channel_id
        )

    def test_var_to_physical_channel_qat_ir(self):
        model, ir = self.create_qat_ir()
        chan1 = model.get_qubit(0).get_acquire_channel()
        chan2 = model.get_qubit(1).get_acquire_channel()

        mapping = IndexMappingAnalysis.var_to_physical_channel_qat_ir(ir)
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert mapping["test0"] == chan1.physical_channel_id
        assert mapping["test1"] == chan1.physical_channel_id
        assert mapping["test2"] == chan2.physical_channel_id

    def test_var_to_qubit_map(self):
        model, ir = self.create_qat_ir()

        _pass = IndexMappingAnalysis(model)
        mapping = _pass.var_to_qubit_map(_pass.var_to_physical_channel_qat_ir(ir))
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert mapping["test0"] == 0
        assert mapping["test1"] == 0
        assert mapping["test2"] == 1

    def test_full_pass_ir(self):
        model, ir = self.create_qat_ir()
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

    def test_full_pass_executable(self):
        model, package = self.create_executable()
        _pass = IndexMappingAnalysis(model)
        res_mgr = ResultManager()
        _pass.run(None, res_mgr, package=package)
        mapping = res_mgr.lookup_by_type(IndexMappingResult).mapping
        assert len(mapping) == 3
        for i in range(3):
            assert f"test{i}" in mapping
        assert mapping["test0"] == 0
        assert mapping["test1"] == 0
        assert mapping["test2"] == 1
