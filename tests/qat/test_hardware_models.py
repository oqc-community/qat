from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.devices import Qubit
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class TestCachedProperties:
    def create_echo_hw_model(self, num_qubits):
        return get_default_echo_hardware(
            num_qubits, connectivity=[(q, q + 1) for q in range(num_qubits - 1)]
        )

    def change_quantum_device_id(self, hw: QuantumHardwareModel, id: str, new_id: str):
        """
        Change the ID of a quantum device in the hardware model. Updates the IDs
        of dependant pulse channels.
        """
        if id not in hw.quantum_devices:
            raise ValueError(
                f"Tried to change the ID for a physical channel ({str(id)}) that doesn't exist."
            )
        if new_id in hw.quantum_devices:
            raise ValueError(
                f"The new ID {str(new_id)} is already assigned to a quantum device."
            )

        # update qubit
        hw.quantum_devices[new_id] = hw.quantum_devices.pop(id)
        hw.quantum_devices[new_id].id = new_id

        # update pulse channels
        for device in hw.quantum_devices.values():
            # update the pulse channels in each device
            old_ids = []
            new_ids = []
            pcs = device.pulse_channels
            for pid, pchan in pcs.items():
                if (
                    hw.quantum_devices[new_id] in pchan.auxiliary_devices
                    or device == hw.quantum_devices[new_id]
                ):
                    # find the new partial id
                    old_full_id = pchan.pulse_channel.full_id()
                    pchan.pulse_channel.id = device._create_pulse_channel_id(
                        pchan.channel_type, [device] + pchan.auxiliary_devices
                    )
                    key = device._create_pulse_channel_id(
                        pchan.channel_type, pchan.auxiliary_devices
                    )
                    old_ids.append(pid)
                    new_ids.append(key)

                    # update the pulse channel dict with the full id
                    hw.pulse_channels[pchan.pulse_channel.create_full_id()] = (
                        hw.pulse_channels.pop(old_full_id)
                    )
            for i in range(len(old_ids)):
                pcs[new_ids[i]] = pcs.pop(old_ids[i])

            # if the qubit is coupled to another qubit, update the id
            if isinstance(device, Qubit) and id in device.pulse_hw_zx_pi_4:
                device.pulse_hw_zx_pi_4[new_id] = device.pulse_hw_zx_pi_4.pop(id)

        hw.delete_cache()

    def change_physical_channel_id(self, hw: QuantumHardwareModel, id: str, new_id: str):
        """
        Change the ID of a physical channel in the hardware model. Updates the IDs
        of dependant pulse channels.
        """
        # Verify channel exists
        if id not in hw.physical_channels:
            raise ValueError(
                f"Tried to change the ID for a physical channel ({str(id)}) that doesn't exist."
            )

        # The new id
        if new_id in hw.physical_channels:
            raise ValueError(
                f"The new ID {str(new_id)} is already assigned to a physical channel."
            )

        # Update the channel in the hardware model
        hw.physical_channels[new_id] = hw.physical_channels.pop(id)
        chan = hw.physical_channels[new_id]
        chan.id = new_id

        # Update the full_ids of pulse channels
        for device in hw.quantum_devices.values():
            pcs = device.pulse_channels
            for _, pchan in pcs.items():
                if pchan.physical_channel == chan:
                    # we only need to update hw.physical_channels
                    old_full_id = id + "." + pchan.partial_id()
                    pchan.pulse_channel.id = device._create_pulse_channel_id(
                        pchan.channel_type, [device] + pchan.auxiliary_devices
                    )
                    new_full_id = pchan.pulse_channel.create_full_id()
                    hw.pulse_channels[new_full_id] = hw.pulse_channels.pop(old_full_id)

        hw.delete_cache()

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
            self.change_quantum_device_id(hw, f"Q{i}", f"P{i}")
            assert hw.quantum_devices[f"P{i}"].full_id() == f"P{i}"

    def test_change_qubit_ids_pulses(self):
        """
        Check that the pulse IDs update correctly when the qubit IDs are changed.
        """
        hw = self.create_echo_hw_model(4)
        hw_check = self.create_echo_hw_model(4)
        for i in range(4):
            self.change_quantum_device_id(hw, f"Q{i}", f"P{i}")

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
            self.change_quantum_device_id(hw, f"Q{i}", f"P{i}")

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
            self.change_quantum_device_id(hw, f"R{i}", f"S{i}")
            assert hw.quantum_devices[f"S{i}"].full_id() == f"S{i}"

    def test_change_resonator_ids_qubits(self):
        """
        Check that the resonator IDs update correctly within qubits.
        """
        hw = self.create_echo_hw_model(4)
        for i in range(4):
            self.change_quantum_device_id(hw, f"R{i}", f"S{i}")

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
            self.change_physical_channel_id(hw, id, new_id)
            assert hw.physical_channels[new_id].full_id() == new_id

    def test_change_physical_channel_ids_devices(self):
        """
        Check that the physical channel IDs correctly update in quantum devices.
        """
        hw = self.create_echo_hw_model(4)
        channel_ids = [key for key in hw.physical_channels.keys()]
        for id in channel_ids:
            new_id = id.replace("CH", "CHA")
            self.change_physical_channel_id(hw, id, new_id)

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
            self.change_physical_channel_id(hw, id, new_id)

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
            self.change_physical_channel_id(hw, id, new_id)

        for device in hw.quantum_devices.values():
            for pc in device.pulse_channels.values():
                assert pc.full_id() in hw.pulse_channels
                assert pc.physical_channel_id in hw.physical_channels
