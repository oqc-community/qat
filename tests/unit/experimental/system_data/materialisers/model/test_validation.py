# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for canonical system data validation (model/validation.py)."""

import pytest

from qat.experimental.system_data.canonical.schema import (
    AcquireDefinitionData,
    CanonicalSystemData,
    ChannelData,
    LinearMapToRealMethodData,
    MaxLikelihoodDiscriminateParams,
    MaxLikelihoodMethodData,
    ModeData,
    OscillatorData,
    PortData,
    ProbabilityEntry,
    QubitCouplingData,
    QubitData,
    ReadoutProbabilityData,
    TwoQubitGateFidelityData,
    WaveformData,
)
from qat.experimental.system_data.materialisers.errors import (
    MaterialisationConsistencyError,
    MaterialisationValidationError,
)
from qat.experimental.system_data.materialisers.model.validation import (
    _is_finite_number,
    validate,
)


def _minimal() -> CanonicalSystemData:
    """Return the absolute minimum valid CanonicalSystemData."""
    return CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )


def _with_mode(
    mode: ModeData,
) -> CanonicalSystemData:
    """Minimal model with one qubit carrying the supplied mode."""
    return CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0, modes=(mode,)),),
    )


def test_valid_minimal_passes():
    validate(_minimal())  # must not raise


def test_no_qubits_raises():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(),
    )
    with pytest.raises(MaterialisationConsistencyError, match="no qubits"):
        validate(model)


def test_no_ports_raises():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationConsistencyError, match="no ports"):
        validate(model)


def test_no_channels_raises():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationConsistencyError, match="no channels"):
        validate(model)


def test_no_oscillators_raises():
    model = CanonicalSystemData(
        oscillators=(),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationConsistencyError, match="no oscillators"):
        validate(model)


@pytest.mark.parametrize("limit", [-1, 1, 100, 1024])
def test_acquire_limit_valid(limit):
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
        acquire_limit=limit,
    )
    validate(model)  # must not raise


@pytest.mark.parametrize("limit", [0, -2, -100])
def test_acquire_limit_invalid_raises(limit):
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
        acquire_limit=limit,
    )
    with pytest.raises(MaterialisationValidationError, match="acquire_limit"):
        validate(model)


@pytest.mark.parametrize("sample_time", [0, -1, -1000])
def test_port_sample_time_non_positive_raises(sample_time):
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=sample_time),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationValidationError, match="sample_time"):
        validate(model)


@pytest.mark.parametrize("block_size", [0, -1])
def test_port_block_size_less_than_one_raises(block_size):
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000, block_size=block_size),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationValidationError, match="block_size"):
        validate(model)


@pytest.mark.parametrize("min_blocks", [0, -1])
def test_port_min_blocks_less_than_one_raises(min_blocks):
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000, min_blocks=min_blocks),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationValidationError, match="min_blocks"):
        validate(model)


@pytest.mark.parametrize("max_blocks", [0, -2])
def test_port_max_blocks_invalid_raises(max_blocks):
    """max_blocks must be -1 or >= 1; 0 and other negatives are rejected."""
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000, max_blocks=max_blocks),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationValidationError, match="max_blocks"):
        validate(model)


def test_port_max_blocks_negative_one_is_valid():
    """max_blocks == -1 means unbounded — always valid."""
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000, min_blocks=5, max_blocks=-1),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    validate(model)  # must not raise


def test_port_min_blocks_greater_than_max_blocks_raises():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000, min_blocks=5, max_blocks=3),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(
        MaterialisationValidationError, match="min_blocks.*max_blocks|min_blocks"
    ):
        validate(model)


def test_port_min_blocks_equal_to_max_blocks_is_valid():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000, min_blocks=3, max_blocks=3),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    validate(model)  # must not raise


@pytest.mark.parametrize("freq", [0, -1, -5_000_000_000])
def test_oscillator_non_positive_frequency_raises(freq):
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=freq),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationValidationError, match="[Oo]scillator frequency"):
        validate(model)


def test_channel_negative_frequency_raises():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=-1),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationValidationError, match="[Cc]hannel frequency"):
        validate(model)


def test_channel_zero_frequency_is_valid():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=0),),
        qubits=(QubitData(id="q0", index=0),),
    )
    validate(model)  # 0 is non-negative, must not raise


def test_channel_unknown_port_raises():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="unknown_port", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationConsistencyError, match="[Uu]nknown port|port"):
        validate(model)


def test_channel_unknown_oscillator_reference_raises():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(
            ChannelData(
                id="ch0",
                port_id="p0",
                frequency=5_000_000_000,
                oscillator_reference="no_such_osc",
            ),
        ),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(
        MaterialisationConsistencyError, match="[Uu]nknown oscillator|oscillator"
    ):
        validate(model)


def test_channel_known_oscillator_reference_is_valid():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(
            ChannelData(
                id="ch0",
                port_id="p0",
                frequency=5_000_000_000,
                oscillator_reference="osc0",
            ),
        ),
        qubits=(QubitData(id="q0", index=0),),
    )
    validate(model)  # osc0 exists


def test_mode_unknown_channel_raises():
    mode = ModeData(id="drive_q0", channel_id="no_such_channel")
    with pytest.raises(MaterialisationConsistencyError, match="[Uu]nknown channel|channel"):
        validate(_with_mode(mode))


def test_mode_known_channel_passes():
    mode = ModeData(id="drive_q0", channel_id="ch0")
    validate(_with_mode(mode))  # ch0 exists in _with_mode


def test_waveform_negative_width_raises():
    mode = ModeData(
        id="drive_q0",
        channel_id="ch0",
        waveform_definitions=(WaveformData(id="w0", width=-1),),
    )
    with pytest.raises(MaterialisationValidationError, match="[Ww]aveform width"):
        validate(_with_mode(mode))


def test_waveform_zero_width_passes():
    mode = ModeData(
        id="drive_q0",
        channel_id="ch0",
        waveform_definitions=(WaveformData(id="w0", width=0),),
    )
    validate(_with_mode(mode))


def test_is_finite_number_bool_returns_false():
    assert _is_finite_number(True) is False
    assert _is_finite_number(False) is False


def test_is_finite_number_finite_complex_returns_true():
    assert _is_finite_number(1 + 2j) is True


def test_is_finite_number_non_numeric_returns_false():
    assert _is_finite_number("hello") is False
    assert _is_finite_number(None) is False


@pytest.mark.parametrize("bad_rise", [float("inf"), float("-inf"), float("nan"), -1.0])
def test_waveform_invalid_rise_raises(bad_rise):
    mode = ModeData(
        id="drive_q0",
        channel_id="ch0",
        waveform_definitions=(WaveformData(id="w0", rise=bad_rise),),
    )
    with pytest.raises(MaterialisationValidationError, match="[Ww]aveform rise"):
        validate(_with_mode(mode))


def test_waveform_zero_rise_passes():
    mode = ModeData(
        id="drive_q0",
        channel_id="ch0",
        waveform_definitions=(WaveformData(id="w0", rise=0.0),),
    )
    validate(_with_mode(mode))


@pytest.mark.parametrize("field", ["amp", "drag", "phase", "amp_setup"])
@pytest.mark.parametrize(
    "bad_value",
    [float("inf"), float("-inf"), float("nan")],
)
def test_waveform_non_finite_numeric_field_raises(field, bad_value):
    mode = ModeData(
        id="drive_q0",
        channel_id="ch0",
        waveform_definitions=(WaveformData(id="w0", **{field: bad_value}),),
    )
    with pytest.raises(MaterialisationValidationError):
        validate(_with_mode(mode))


@pytest.mark.parametrize("field", ["amp", "drag", "phase", "amp_setup"])
def test_waveform_none_numeric_field_passes(field):
    """None values for optional waveform numeric fields are acceptable."""
    mode = ModeData(
        id="drive_q0",
        channel_id="ch0",
        waveform_definitions=(WaveformData(id="w0", **{field: None}),),
    )
    validate(_with_mode(mode))


@pytest.mark.parametrize("bad_delay", [-1, -100])
def test_acquire_negative_delay_raises(bad_delay):
    mode = ModeData(
        id="acquire_q0",
        channel_id="ch0",
        acquire_definitions=(AcquireDefinitionData(id="acq0", delay=bad_delay),),
    )
    with pytest.raises(MaterialisationValidationError, match="[Aa]cquire delay"):
        validate(_with_mode(mode))


def test_acquire_zero_delay_passes():
    mode = ModeData(
        id="acquire_q0",
        channel_id="ch0",
        acquire_definitions=(AcquireDefinitionData(id="acq0", delay=0),),
    )
    validate(_with_mode(mode))


@pytest.mark.parametrize("bad_width", [-1, -100])
def test_acquire_negative_width_raises(bad_width):
    mode = ModeData(
        id="acquire_q0",
        channel_id="ch0",
        acquire_definitions=(AcquireDefinitionData(id="acq0", width=bad_width),),
    )
    with pytest.raises(MaterialisationValidationError, match="[Aa]cquire width"):
        validate(_with_mode(mode))


def test_acquire_zero_width_passes():
    mode = ModeData(
        id="acquire_q0",
        channel_id="ch0",
        acquire_definitions=(AcquireDefinitionData(id="acq0", width=0),),
    )
    validate(_with_mode(mode))


@pytest.mark.parametrize(
    "weights",
    [
        (float("inf"),),
        (float("nan"),),
        (1.0 + float("inf") * 1j,),
        (complex(float("nan"), 0.0),),
    ],
)
def test_acquire_non_finite_weights_raises(weights):
    mode = ModeData(
        id="acquire_q0",
        channel_id="ch0",
        acquire_definitions=(AcquireDefinitionData(id="acq0", weights=weights),),
    )
    with pytest.raises(MaterialisationValidationError, match="[Ww]eight"):
        validate(_with_mode(mode))


def test_acquire_valid_complex_weights_passes():
    mode = ModeData(
        id="acquire_q0",
        channel_id="ch0",
        acquire_definitions=(AcquireDefinitionData(id="acq0", weights=(1.0 + 0.5j, 0.0)),),
    )
    validate(_with_mode(mode))


def test_linear_map_valid_args_passes():
    mode = ModeData(
        id="acquire_q0",
        channel_id="ch0",
        post_process_method=LinearMapToRealMethodData(mean_z_map_args=(1 + 0j, 0j)),
    )
    validate(_with_mode(mode))


@pytest.mark.parametrize(
    "args",
    [
        (1 + 0j,),  # only one element
        (1 + 0j, 0j, 1j),  # three elements
        (),  # empty
    ],
)
def test_linear_map_wrong_arg_count_raises(args):
    mode = ModeData(
        id="acquire_q0",
        channel_id="ch0",
        post_process_method=LinearMapToRealMethodData(mean_z_map_args=args),
    )
    with pytest.raises(MaterialisationValidationError, match="mean_z_map_args"):
        validate(_with_mode(mode))


@pytest.mark.parametrize(
    "args",
    [
        (complex(float("inf"), 0), 0j),
        (1 + 0j, complex(float("nan"), 0)),
    ],
)
def test_linear_map_non_finite_args_raises(args):
    mode = ModeData(
        id="acquire_q0",
        channel_id="ch0",
        post_process_method=LinearMapToRealMethodData(mean_z_map_args=args),
    )
    with pytest.raises(MaterialisationValidationError, match="mean_z_map_args"):
        validate(_with_mode(mode))


def _max_likelihood_mode(**overrides) -> ModeData:
    """Return a mode with a valid MaxLikelihoodMethodData, optionally overridden."""
    defaults = {
        "method": "max_likelihood",
        "states": ((0, MaxLikelihoodDiscriminateParams(location=0 + 0j)),),
        "noise_est": 1.0,
        "p_min": 0.0,
        "transform": None,
        "offset": None,
    }
    defaults.update(overrides)
    return ModeData(
        id="acquire_q0",
        channel_id="ch0",
        post_process_method=MaxLikelihoodMethodData(**defaults),
    )


def test_max_likelihood_valid_passes():
    validate(_with_mode(_max_likelihood_mode()))


def test_max_likelihood_empty_states_raises():
    mode = _max_likelihood_mode(states=())
    with pytest.raises(MaterialisationValidationError, match="states"):
        validate(_with_mode(mode))


def test_max_likelihood_non_finite_location_raises():
    mode = _max_likelihood_mode(
        states=((0, MaxLikelihoodDiscriminateParams(location=complex(float("nan"), 0))),)
    )
    with pytest.raises(MaterialisationValidationError, match="location"):
        validate(_with_mode(mode))


@pytest.mark.parametrize("noise_est", [float("inf"), float("-inf"), float("nan")])
def test_max_likelihood_non_finite_noise_est_raises(noise_est):
    mode = _max_likelihood_mode(noise_est=noise_est)
    with pytest.raises(MaterialisationValidationError, match="noise_est"):
        validate(_with_mode(mode))


@pytest.mark.parametrize("p_min", [-0.1, 1.1, float("inf"), float("nan")])
def test_max_likelihood_invalid_p_min_raises(p_min):
    mode = _max_likelihood_mode(p_min=p_min)
    with pytest.raises(MaterialisationValidationError, match="p_min"):
        validate(_with_mode(mode))


@pytest.mark.parametrize("p_min", [0.0, 0.5, 1.0])
def test_max_likelihood_valid_p_min_passes(p_min):
    validate(_with_mode(_max_likelihood_mode(p_min=p_min)))


def test_max_likelihood_non_finite_transform_raises():
    mode = _max_likelihood_mode(
        transform=((float("inf"), 0.0), (0.0, 1.0)),
    )
    with pytest.raises(MaterialisationValidationError, match="transform"):
        validate(_with_mode(mode))


def test_max_likelihood_valid_transform_passes():
    mode = _max_likelihood_mode(transform=((1.0, 0.0), (0.0, 1.0)))
    validate(_with_mode(mode))


def test_max_likelihood_non_finite_offset_raises():
    mode = _max_likelihood_mode(offset=(float("nan"), 0.0))
    with pytest.raises(MaterialisationValidationError, match="offset"):
        validate(_with_mode(mode))


def test_max_likelihood_valid_offset_passes():
    mode = _max_likelihood_mode(offset=(0.1, -0.2))
    validate(_with_mode(mode))


def _model_with_readout_probability(entries: tuple) -> CanonicalSystemData:
    return CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(
            QubitData(
                id="q0",
                index=0,
                readout_probability=ReadoutProbabilityData(probability_entries=entries),
            ),
        ),
    )


def test_readout_probability_valid_passes():
    model = _model_with_readout_probability(
        (
            ProbabilityEntry(prepared_state=0, measured_state=0, probability=0.95),
            ProbabilityEntry(prepared_state=0, measured_state=1, probability=0.05),
            ProbabilityEntry(prepared_state=1, measured_state=0, probability=0.03),
            ProbabilityEntry(prepared_state=1, measured_state=1, probability=0.97),
        )
    )
    validate(model)


@pytest.mark.parametrize("bad_prob", [float("inf"), float("-inf"), float("nan")])
def test_readout_probability_non_finite_raises(bad_prob):
    model = _model_with_readout_probability(
        (ProbabilityEntry(prepared_state=0, measured_state=0, probability=bad_prob),)
    )
    with pytest.raises(MaterialisationValidationError, match="[Rr]eadout probability"):
        validate(model)


@pytest.mark.parametrize("bad_prob", [-0.001, 1.001])
def test_readout_probability_out_of_range_raises(bad_prob):
    model = _model_with_readout_probability(
        (ProbabilityEntry(prepared_state=0, measured_state=0, probability=bad_prob),)
    )
    with pytest.raises(MaterialisationValidationError, match="\\[0.*1\\]|[Rr]eadout"):
        validate(model)


def test_readout_probability_sum_not_one_raises():
    """Probabilities for a given prepared state must sum to 1."""
    model = _model_with_readout_probability(
        (
            ProbabilityEntry(prepared_state=0, measured_state=0, probability=0.6),
            ProbabilityEntry(prepared_state=0, measured_state=1, probability=0.2),
            # sum = 0.8, not 1.0
        )
    )
    with pytest.raises(MaterialisationConsistencyError, match="[Ss]um|[Nn]ormali"):
        validate(model)


def test_readout_probability_sum_near_one_with_tolerance_passes():
    """Sum within PROBABILITY_TOLERANCE (1e-6) of 1.0 must not raise."""
    eps = 5e-7  # half of 1e-6 tolerance
    model = _model_with_readout_probability(
        (
            ProbabilityEntry(prepared_state=0, measured_state=0, probability=1.0 - eps),
            ProbabilityEntry(prepared_state=0, measured_state=1, probability=eps),
        )
    )
    validate(model)


def test_readout_probability_multiple_prepared_states_validated_independently():
    """Each prepared state is normalised independently."""
    model = _model_with_readout_probability(
        (
            ProbabilityEntry(prepared_state=0, measured_state=0, probability=0.9),
            ProbabilityEntry(prepared_state=0, measured_state=1, probability=0.1),
            ProbabilityEntry(prepared_state=1, measured_state=0, probability=0.05),
            ProbabilityEntry(prepared_state=1, measured_state=1, probability=0.95),
        )
    )
    validate(model)


def test_readout_probability_second_prepared_state_bad_sum_raises():
    model = _model_with_readout_probability(
        (
            ProbabilityEntry(prepared_state=0, measured_state=0, probability=1.0),
            ProbabilityEntry(prepared_state=1, measured_state=0, probability=0.5),
            # prepared_state=1 sums to 0.5
        )
    )
    with pytest.raises(MaterialisationConsistencyError):
        validate(model)


def _two_qubit_model(*, extra_couplings=()) -> CanonicalSystemData:
    return CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(
            QubitData(id="q0", index=0),
            QubitData(id="q1", index=1),
        ),
        couplings=extra_couplings,
    )


def test_coupling_valid_qubit_ids_passes():
    model = _two_qubit_model(
        extra_couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
            ),
        )
    )
    validate(model)


def test_coupling_unknown_source_qubit_raises():
    model = _two_qubit_model(
        extra_couplings=(
            QubitCouplingData(
                source_qubit_id="unknown_q",
                target_qubit_id="q1",
                gate_fidelities=(),
            ),
        )
    )
    with pytest.raises(MaterialisationConsistencyError, match="[Uu]nknown qubit|qubit"):
        validate(model)


def test_coupling_unknown_target_qubit_raises():
    model = _two_qubit_model(
        extra_couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="unknown_q",
                gate_fidelities=(),
            ),
        )
    )
    with pytest.raises(MaterialisationConsistencyError, match="[Uu]nknown qubit|qubit"):
        validate(model)


def test_coupling_missing_fidelities_does_not_raise():
    """Missing gate fidelities emit a warning but do not fail validation."""
    model = _two_qubit_model(
        extra_couplings=(
            QubitCouplingData(
                source_qubit_id="q0",
                target_qubit_id="q1",
                gate_fidelities=(),
            ),
        )
    )
    validate(model)  # must not raise


def test_inconsistent_acquire_sample_times_does_not_raise():
    """Heterogeneous sample_time across acquire-capable ports warns but does not fail."""
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(
            PortData(id="p0", sample_time=1000, acquire_allowed=True),
            PortData(id="p1", sample_time=2000, acquire_allowed=True),
        ),
        channels=(
            ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),
            ChannelData(id="ch1", port_id="p1", frequency=5_000_000_000),
        ),
        qubits=(QubitData(id="q0", index=0),),
    )
    validate(model)  # must not raise


def test_inconsistent_drive_sample_times_does_not_raise():
    """Heterogeneous sample_time across drive ports warns but does not fail."""
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(
            PortData(id="p0", sample_time=1000, acquire_allowed=False),
            PortData(id="p1", sample_time=2000, acquire_allowed=False),
        ),
        channels=(
            ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),
            ChannelData(id="ch1", port_id="p1", frequency=5_000_000_000),
        ),
        qubits=(QubitData(id="q0", index=0),),
    )
    validate(model)  # must not raise


def test_validation_error_carries_path():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=0),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationValidationError) as exc_info:
        validate(model)
    assert exc_info.value.path is not None
    assert "sample_time" in exc_info.value.path


def test_consistency_error_carries_path():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=5_000_000_000),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="bad_port", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationConsistencyError) as exc_info:
        validate(model)
    assert exc_info.value.path is not None
    assert "port_id" in exc_info.value.path


def test_validation_error_carries_details():
    model = CanonicalSystemData(
        oscillators=(OscillatorData(id="osc0", frequency=-1),),
        ports=(PortData(id="p0", sample_time=1000),),
        channels=(ChannelData(id="ch0", port_id="p0", frequency=5_000_000_000),),
        qubits=(QubitData(id="q0", index=0),),
    )
    with pytest.raises(MaterialisationValidationError) as exc_info:
        validate(model)
    assert "value" in exc_info.value.details
    assert exc_info.value.details["value"] == -1
