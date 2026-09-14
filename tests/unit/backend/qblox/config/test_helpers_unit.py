# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.backend.qblox.config.helpers import (
    QcmRfConfigHelper,
    QrcConfigHelper,
    QrmRfConfigHelper,
)
from qat.backend.qblox.config.specification import ModuleConfig


@pytest.mark.parametrize(
    ("config_name", "method_name", "driver_name"),
    [
        pytest.param("fir", "configure_fir", "fir", id="fir"),
        pytest.param("exp0", "configure_exp0", "exp0", id="exponential-0"),
        pytest.param("exp1", "configure_exp1", "exp1", id="exponential-1"),
        pytest.param("exp2", "configure_exp2", "exp2", id="exponential-2"),
        pytest.param("exp3", "configure_exp3", "exp3", id="exponential-3"),
    ],
)
def test_qrc_configures_all_filter_paths(
    mocker,
    config_name,
    method_name,
    driver_name,
):
    values = {
        "out0": "bypassed",
        "out1": "delay_comp",
        "out2": "bypassed",
        "out3": "delay_comp",
        "out4": "bypassed",
        "out5": "delay_comp",
        "marker0": "bypassed",
    }
    module = mocker.Mock()
    helper = QrcConfigHelper(ModuleConfig(**{config_name: values}))

    getattr(helper, method_name)(module)

    getattr(module, f"out0_{driver_name}_config").assert_called_once_with("bypassed")
    getattr(module, f"out1_{driver_name}_config").assert_called_once_with("delay_comp")
    getattr(module, f"out2_{driver_name}_config").assert_called_once_with("bypassed")
    getattr(module, f"out3_{driver_name}_config").assert_called_once_with("delay_comp")
    getattr(module, f"out4_{driver_name}_config").assert_called_once_with("bypassed")
    getattr(module, f"out5_{driver_name}_config").assert_called_once_with("delay_comp")
    getattr(module, f"marker0_{driver_name}_config").assert_called_once_with("bypassed")


def test_qrc_configures_all_scope_acquisition_paths(mocker):
    module = mocker.Mock()
    helper = QrcConfigHelper(
        ModuleConfig(
            scope_acq={
                "sequencer_select": 2,
                "avg_mode_en_path0": True,
                "avg_mode_en_path1": True,
                "avg_mode_en_path2": True,
                "avg_mode_en_path3": True,
                "trigger_mode_path0": "sequencer",
                "trigger_mode_path1": "level",
                "trigger_mode_path2": "sequencer",
                "trigger_mode_path3": "level",
                "trigger_level_path0": 0.1,
                "trigger_level_path1": 0.2,
                "trigger_level_path2": 0.3,
                "trigger_level_path3": 0.4,
            }
        )
    )

    helper.configure_scope_acq(module)

    module.scope_acq_sequencer_select.assert_called_once_with(2)
    module.scope_acq_avg_mode_en_path0.assert_called_once_with(True)
    module.scope_acq_avg_mode_en_path1.assert_called_once_with(True)
    module.scope_acq_avg_mode_en_path2.assert_called_once_with(True)
    module.scope_acq_avg_mode_en_path3.assert_called_once_with(True)
    module.scope_acq_trigger_mode_path0.assert_called_once_with("sequencer")
    module.scope_acq_trigger_mode_path1.assert_called_once_with("level")
    module.scope_acq_trigger_mode_path2.assert_called_once_with("sequencer")
    module.scope_acq_trigger_mode_path3.assert_called_once_with("level")
    module.scope_acq_trigger_level_path0.assert_called_once_with(0.1)
    module.scope_acq_trigger_level_path1.assert_called_once_with(0.2)
    module.scope_acq_trigger_level_path2.assert_called_once_with(0.3)
    module.scope_acq_trigger_level_path3.assert_called_once_with(0.4)


def test_qcm_rf_configures_complete_module(mocker):
    module = mocker.Mock()
    helper = QcmRfConfigHelper(
        ModuleConfig(
            lo={
                "out0_en": True,
                "out0_freq": 4.0e9,
                "out1_en": True,
                "out1_freq": 5.0e9,
            },
            attenuation={"out0": 10.0, "out1": 12.0},
            offset={
                "out0_path0": 0.1,
                "out0_path1": 0.2,
                "out1_path0": 0.3,
                "out1_path1": 0.4,
            },
        )
    )

    helper.configure_module(module)

    module.out0_lo_en.assert_called_once_with(True)
    module.out0_lo_freq.assert_called_once_with(4.0e9)
    module.out1_lo_en.assert_called_once_with(True)
    module.out1_lo_freq.assert_called_once_with(5.0e9)
    module.out0_att.assert_called_once_with(10)
    module.out1_att.assert_called_once_with(12)
    module.out0_offset_path0.assert_called_once_with(0.1)
    module.out0_offset_path1.assert_called_once_with(0.2)
    module.out1_offset_path0.assert_called_once_with(0.3)
    module.out1_offset_path1.assert_called_once_with(0.4)


def test_qrm_rf_configures_complete_module(mocker):
    module = mocker.Mock()
    helper = QrmRfConfigHelper(
        ModuleConfig(
            lo={"out0_in0_en": True, "out0_in0_freq": 4.0e9},
            attenuation={"out0": 10.0, "in0": 12.0},
            offset={
                "out0_path0": 0.1,
                "out0_path1": 0.2,
                "in0_path0": 0.3,
                "in0_path1": 0.4,
            },
            scope_acq={
                "sequencer_select": 1,
                "avg_mode_en_path0": True,
                "avg_mode_en_path1": True,
                "trigger_mode_path0": "sequencer",
                "trigger_mode_path1": "level",
                "trigger_level_path0": 0.1,
                "trigger_level_path1": 0.2,
            },
        )
    )

    helper.configure_module(module)

    module.out0_in0_lo_en.assert_called_once_with(True)
    module.out0_in0_lo_freq.assert_called_once_with(4.0e9)
    module.out0_att.assert_called_once_with(10)
    module.in0_att.assert_called_once_with(12)
    module.out0_offset_path0.assert_called_once_with(0.1)
    module.out0_offset_path1.assert_called_once_with(0.2)
    module.in0_offset_path0.assert_called_once_with(0.3)
    module.in0_offset_path1.assert_called_once_with(0.4)
    module.scope_acq_sequencer_select.assert_called_once_with(1)
    module.scope_acq_avg_mode_en_path0.assert_called_once_with(True)
    module.scope_acq_avg_mode_en_path1.assert_called_once_with(True)
    module.scope_acq_trigger_mode_path0.assert_called_once_with("sequencer")
    module.scope_acq_trigger_mode_path1.assert_called_once_with("level")
    module.scope_acq_trigger_level_path0.assert_called_once_with(0.1)
    module.scope_acq_trigger_level_path1.assert_called_once_with(0.2)


@pytest.mark.parametrize("helper_type", [QcmRfConfigHelper, QrmRfConfigHelper])
def test_rf_default_module_configuration_makes_no_driver_calls(mocker, helper_type):
    module = mocker.Mock()

    helper_type().configure_module(module)

    assert module.mock_calls == []


@pytest.mark.parametrize(
    ("helper_type", "field", "value", "expected"),
    [
        pytest.param(QcmRfConfigHelper, "out0", 1.5, "integer", id="qcm-non-integer"),
        pytest.param(QcmRfConfigHelper, "out0", 3.0, "multiples of 2", id="qcm-out0-odd"),
        pytest.param(QcmRfConfigHelper, "out1", 1.5, "integer", id="qcm-out1-fractional"),
        pytest.param(QcmRfConfigHelper, "out1", 3.0, "multiples of 2", id="qcm-odd"),
        pytest.param(QrmRfConfigHelper, "out0", 1.5, "integer", id="qrm-out"),
        pytest.param(QrmRfConfigHelper, "out0", 3.0, "multiples of 2", id="qrm-out-odd"),
        pytest.param(QrmRfConfigHelper, "in0", 1.5, "integer", id="qrm-in-fractional"),
        pytest.param(QrmRfConfigHelper, "in0", 3.0, "multiples of 2", id="qrm-in"),
    ],
)
def test_rf_attenuation_rejects_unsupported_values(
    mocker,
    helper_type,
    field,
    value,
    expected,
):
    module = mocker.Mock()
    helper = helper_type(ModuleConfig(attenuation={field: value}))

    with pytest.raises(ValueError, match=expected):
        helper.configure_attenuation(module)


@pytest.mark.parametrize(
    ("helper_type", "calibration_method", "expected"),
    [
        pytest.param(
            QcmRfConfigHelper,
            "out0_lo_cal",
            {
                "out0_path0": 0.1,
                "out0_path1": 0.2,
                "out1_path0": 0.3,
                "out1_path1": 0.4,
            },
            id="qcm-rf",
        ),
        pytest.param(
            QrmRfConfigHelper,
            "out0_in0_lo_cal",
            {
                "out0_path0": 0.1,
                "out0_path1": 0.2,
                "in0_path0": 0.3,
                "in0_path1": 0.4,
            },
            id="qrm-rf",
        ),
    ],
)
def test_rf_lo_calibration_returns_measured_offsets(
    mocker,
    helper_type,
    calibration_method,
    expected,
):
    module = mocker.Mock()
    module.out0_offset_path0.return_value = 0.1
    module.out0_offset_path1.return_value = 0.2
    module.out1_offset_path0.return_value = 0.3
    module.out1_offset_path1.return_value = 0.4
    module.in0_offset_path0.return_value = 0.3
    module.in0_offset_path1.return_value = 0.4

    offsets = helper_type().calibrate_lo_leakage(module)

    getattr(module, calibration_method).assert_called_once_with()
    assert offsets.model_dump(exclude_none=True) == expected


def test_qrc_lo_calibration_is_explicitly_unsupported(mocker):
    module = mocker.Mock()

    assert QrcConfigHelper().calibrate_lo_leakage(module) is None
