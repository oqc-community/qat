from qat.purr.backends.echo import get_default_echo_hardware


class TestCachedProperties:
    def create_echo_hw_model(self, num_qubits):
        return get_default_echo_hardware(
            num_qubits, connectivity=[(q, q + 1) for q in range(num_qubits - 1)]
        )

    def test_pulse_channel_cached_id(self):
        """
        Check that deleting the cached properties of a pulse channel works.
        """
        hw = self.create_echo_hw_model(4)
        pid = list(hw.pulse_channels.keys())[0]
        pulse_channel = hw.pulse_channels[pid]
        pulse_channel.id = "test"
        pulse_channel._delete_cached_full_id()
        assert pulse_channel.full_id() == pulse_channel.physical_channel_id + ".test"

    def test_change_qubit_ids(self):
        """
        Check that the qubit IDs update correctly.
        """
        hw = self.create_echo_hw_model(4)
        for i in range(4):
            hw.change_qubit_id(f"Q{i}", f"P{i}")
            assert hw.quantum_devices[f"P{i}"].full_id() == f"P{i}"

    def test_change_qubit_ids_pulses(self):
        """
        Check that the pulse IDs update correctly when the qubit IDs are changed.
        """
        hw = self.create_echo_hw_model(4)
        hw_check = self.create_echo_hw_model(4)
        for i in range(4):
            hw.change_qubit_id(f"Q{i}", f"P{i}")

        for key, val in hw_check.pulse_channels.items():
            new_key = key.replace("Q", "P")
            assert hw.pulse_channels[new_key].full_id() == val.full_id().replace("Q", "P")

    def test_change_qubit_ids_pulses_qubit(self):
        """
        Check that the pulse IDs updated correctly with the qubit when qubit IDs are changed.
        """
        hw = self.create_echo_hw_model(4)
        hw_check = self.create_echo_hw_model(4)
        for i in range(4):
            hw.change_qubit_id(f"Q{i}", f"P{i}")

        for qubit in hw_check.qubits:
            qubit_new = qubit.id.replace("Q", "P")
            for key, val in qubit.pulse_channels.items():
                new_key = key.replace("Q", "P")
                assert hw.quantum_devices[qubit_new].pulse_channels[
                    new_key
                ].pulse_channel.full_id() == val.pulse_channel.full_id().replace("Q", "P")

    def test_change_resonator_ids(self):
        """
        Check that the resonator IDs update correctly.
        """
        hw = self.create_echo_hw_model(4)
        for i in range(4):
            hw.change_resonator_id(f"R{i}", f"S{i}")
            assert hw.quantum_devices[f"S{i}"].full_id() == f"S{i}"

    def test_change_resonator_ids_qubits(self):
        """
        Check that the resonator IDs update correctly within qubits.
        """
        hw = self.create_echo_hw_model(4)
        for i in range(4):
            hw.change_resonator_id(f"R{i}", f"S{i}")

        for qubit in hw.qubits:
            assert qubit.measure_device.full_id() in hw.quantum_devices

    def test_change_physical_channel_ids(self):
        """
        Check the physical channel IDs update correctly.
        """
        hw = self.create_echo_hw_model(4)
        channel_ids = [key for key in hw.physical_channels.keys()]
        for id in channel_ids:
            new_id = id.replace("CH", "CHA")
            hw.change_physical_channel_id(id, new_id)
            assert hw.physical_channels[new_id].full_id() == new_id

    def test_change_physical_channel_ids_devices(self):
        """
        Check that the physical channel IDs correctly update in quantum devices.
        """
        hw = self.create_echo_hw_model(4)
        channel_ids = [key for key in hw.physical_channels.keys()]
        for id in channel_ids:
            new_id = id.replace("CH", "CHA")
            hw.change_physical_channel_id(id, new_id)

        for device in hw.quantum_devices.values():
            assert device.physical_channel.full_id() in hw.physical_channels

    def test_change_physical_channel_ids_pulses(self):
        """
        Check that the physical channel IDs correctly update the pulse channel
        IDs.
        """
        hw = self.create_echo_hw_model(4)
        hw_check = self.create_echo_hw_model(4)
        channel_ids = [key for key in hw.physical_channels.keys()]
        for id in channel_ids:
            new_id = id.replace("CH", "CHA")
            hw.change_physical_channel_id(id, new_id)

        for key in hw_check.pulse_channels.keys():
            new_id = key.replace("CH", "CHA")
            assert new_id in hw.pulse_channels
            assert hw.pulse_channels[new_id].full_id() == new_id
            assert hw.pulse_channels[new_id].physical_channel_id in hw.physical_channels

    def test_change_physical_channel_ids_device_pulses(self):
        """
        Check that the physical channel IDs correctly update the pulse channel
        IDs within quantum devices.
        """
        hw = self.create_echo_hw_model(4)
        channel_ids = [key for key in hw.physical_channels.keys()]
        for id in channel_ids:
            new_id = id.replace("CH", "CHA")
            hw.change_physical_channel_id(id, new_id)

        for device in hw.quantum_devices.values():
            for pc in device.pulse_channels.values():
                assert pc.full_id() in hw.pulse_channels
                assert pc.physical_channel_id in hw.physical_channels
