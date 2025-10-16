# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from dataclasses import fields

import pytest

from qat.integrations.features import VERSION, OpenPulseFeatures
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.loaders.purr.echo import EchoModelLoader


class TestOpenPulseFeatures:
    hw_purr = EchoModelLoader(qubit_count=4).load()
    hw_pyd = convert_purr_echo_hw_to_pydantic(hw_purr)

    @pytest.mark.parametrize("invalid_hw", [None, 123, "invalid", [], hw_purr])
    def test_from_hardware_invalid(self, invalid_hw):
        with pytest.raises(ValueError, match="PhysicalHardwareModel"):
            OpenPulseFeatures.from_hardware(invalid_hw)

    def test_from_hardware(self):
        features = OpenPulseFeatures.from_hardware(self.hw_pyd)
        assert isinstance(features, OpenPulseFeatures)

    def test_to_json_dict(self):
        features = OpenPulseFeatures.from_hardware(self.hw_pyd)

        json_dict = features.to_json_dict()
        assert isinstance(json_dict, dict)
        assert "open_pulse" in json_dict

        features = json_dict["open_pulse"]
        assert "name" in features
        assert features["name"] == "OpenPulse"

        assert "description" in features
        assert features["description"] == "Features for OpenPulse integration."

        assert "version" in features
        assert features["version"] == VERSION

        assert "enabled" in features
        assert features["enabled"]

        assert "ports" in features
        assert len(features["ports"]) == self.hw_pyd.number_of_qubits * 2

        assert "frames" in features
        number_of_pulse_channels = (
            sum(
                len(qubit.all_qubit_and_resonator_pulse_channels)
                for qubit in self.hw_pyd.qubits.values()
            )
            - 2 * self.hw_pyd.number_of_qubits
        )  # Freq shift and second state pulse channels are ignored for OpenPulse features.
        assert len(features["frames"]) == number_of_pulse_channels

        assert "waveforms" in features
        assert "constraints" in features

    def test_parity_with_purr_openpulse_features(self):
        from qat.purr.integrations.features import (
            OpenPulseFeatures as PurrOpenPulseFeatures,
        )

        purr_features = PurrOpenPulseFeatures()
        purr_features.for_hardware(self.hw_purr)
        pyd_features = OpenPulseFeatures.from_hardware(self.hw_pyd)

        # Compare the JSON dicts
        purr_dict = purr_features.to_json_dict()["open_pulse"]
        pyd_dict = pyd_features.to_json_dict()["open_pulse"]

        assert set(purr_dict.keys()).issubset(
            pyd_dict.keys()
        )  # The pyd features contain more info.

        for key in ["ports", "frames"]:
            assert set(purr_dict[key].keys()) == set(pyd_dict[key].keys())
            for name, component_purr in purr_dict[key].items():
                assert component_purr == pyd_dict[key][name]

        assert set(purr_dict["waveforms"].keys()).issubset(
            set(pyd_dict["waveforms"].keys())
        )

        for field in fields(purr_dict["constraints"]):
            assert field.name in pyd_dict["constraints"]
