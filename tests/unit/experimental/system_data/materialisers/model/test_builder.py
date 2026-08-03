# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for CanonicalSystemDataBuilder."""

import pytest

from qat.experimental.system_data.canonical.schema import (
    AcquireDefinitionData,
    AttributeEntry,
    CanonicalSystemData,
    DelayOperationStepData,
    LinearMapToRealMethodData,
    MaxLikelihoodDiscriminateParams,
    MaxLikelihoodMethodData,
    ModeData,
    OperationData,
    ProbabilityEntry,
    ReadoutProbabilityData,
    TwoQubitGateFidelityData,
    WaveformData,
)
from qat.experimental.system_data.materialisers.builder import CanonicalSystemDataBuilder


def _base() -> CanonicalSystemDataBuilder:
    """Minimal valid builder used as a base for non-resource field tests."""
    return (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
    )


def test_build_returns_canonical_system_data():
    canonical = _base().build()
    assert isinstance(canonical, CanonicalSystemData)


def test_with_calibration_id():
    canonical = _base().with_calibration_id("cal-abc").build()
    assert canonical.calibration_id == "cal-abc"


def test_with_acquire_limit():
    canonical = _base().with_acquire_limit(512).build()
    assert canonical.acquire_limit == 512


def test_with_acquire_mode_instance():
    canonical = _base().with_acquire_mode("integrator").build()
    assert len(canonical.acquire_modes) == 1
    assert canonical.acquire_modes[0].type == "integrator"


def test_with_acquire_mode_with_attributes():
    entry = AttributeEntry(key="resolution", value=16)
    canonical = _base().with_acquire_mode("raw", attributes=(entry,)).build()
    assert canonical.acquire_modes[0].attributes == (entry,)


def test_with_default_acquire_mode():
    canonical = (
        _base()
        .with_acquire_mode("integrator")
        .with_default_acquire_mode("integrator")
        .build()
    )
    assert canonical.default_acquire_mode == "integrator"


def test_with_reset_method_instance():
    canonical = _base().with_reset_method("passive").build()
    assert len(canonical.reset_methods) == 1
    assert canonical.reset_methods[0].type == "passive"


def test_with_reset_method_with_attributes():
    entry = AttributeEntry(key="duration_ns", value=1000)
    canonical = _base().with_reset_method("passive", attributes=(entry,)).build()
    assert canonical.reset_methods[0].attributes == (entry,)


def test_with_default_reset_method():
    canonical = (
        _base().with_reset_method("passive").with_default_reset_method("passive").build()
    )
    assert canonical.default_reset_method == "passive"


def test_with_oscillator():
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .build()
    )
    assert len(canonical.oscillators) == 1
    assert canonical.oscillators[0].id == "osc0"
    assert canonical.oscillators[0].frequency == 5_000_000_000


def test_with_oscillator_optional_resource():
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000, external_resource_id="r0")
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000, oscillator_reference="osc0")
        .with_qubit("q0", 0)
        .build()
    )
    assert canonical.oscillators[0].external_resource_id == "r0"


def test_with_port():
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .build()
    )
    assert len(canonical.ports) == 1
    assert canonical.ports[0].id == "p0"
    assert canonical.ports[0].sample_time == 1000
    assert canonical.ports[0].acquire_allowed is False


def test_with_port_acquire_allowed():
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000, acquire_allowed=True)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .build()
    )
    assert canonical.ports[0].acquire_allowed is True


def test_with_channel():
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .build()
    )
    assert len(canonical.channels) == 1
    assert canonical.channels[0].id == "ch0"
    assert canonical.channels[0].port_id == "p0"
    assert canonical.channels[0].frequency == 5_000_000_000


def test_with_channel_optional_fields():
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel(
            "ch0",
            "p0",
            5_000_000_000,
            oscillator_reference="osc0",
            scale=0.5 + 0.0j,
            imbalance=0.9,
            phase_offset=0.1,
        )
        .with_qubit("q0", 0)
        .build()
    )
    ch = canonical.channels[0]
    assert ch.oscillator_reference == "osc0"
    assert ch.scale == 0.5 + 0.0j
    assert ch.imbalance == 0.9
    assert ch.phase_offset == 0.1


def test_with_qubit():
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .build()
    )
    assert len(canonical.qubits) == 1
    assert canonical.qubits[0].id == "q0"
    assert canonical.qubits[0].index == 0
    assert canonical.qubits[0].modes == ()


def test_with_qubit_with_modes():
    mode = ModeData(id="drive_q0", channel_id="ch0")
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0, modes=(mode,))
        .build()
    )
    assert canonical.qubits[0].modes == (mode,)


def test_with_coupling():
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .with_qubit("q1", 1)
        .with_coupling("q0", "q1")
        .build()
    )
    assert len(canonical.couplings) == 1
    c = canonical.couplings[0]
    assert c.source_qubit_id == "q0"
    assert c.target_qubit_id == "q1"
    assert c.gate_fidelities == ()


def test_with_coupling_with_fidelities():
    fidelity = TwoQubitGateFidelityData(gate="cx", fidelity=0.99)
    canonical = (
        CanonicalSystemDataBuilder()
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .with_qubit("q1", 1)
        .with_coupling("q0", "q1", gate_fidelities=(fidelity,))
        .build()
    )
    assert canonical.couplings[0].gate_fidelities == (fidelity,)


def test_with_external_resource():
    canonical = _base().with_external_resource("r0", object_type="ClusterModule").build()
    assert len(canonical.external_resources) == 1
    assert canonical.external_resources[0].id == "r0"
    assert canonical.external_resources[0].object_type == "ClusterModule"


def test_with_metadata_entry_instance():
    entry = AttributeEntry(key="lab", value="oxford")
    canonical = _base().with_metadata(entry).build()
    assert canonical.metadata == (entry,)


def test_with_metadata_key_value_shorthand():
    canonical = _base().with_metadata("env", "ci").build()
    assert len(canonical.metadata) == 1
    assert canonical.metadata[0].key == "env"
    assert canonical.metadata[0].value == "ci"


def test_with_metadata_raises_if_entry_and_value_given():
    with pytest.raises(TypeError, match="do not pass a separate value"):
        CanonicalSystemDataBuilder().with_metadata(AttributeEntry(key="k", value="v"), "v2")


def test_with_metadata_raises_if_string_key_without_value():
    with pytest.raises(TypeError, match="a value must be supplied"):
        CanonicalSystemDataBuilder().with_metadata("k")


def test_chaining_returns_same_builder_instance():
    builder = CanonicalSystemDataBuilder()
    result = builder.with_calibration_id("x").with_acquire_limit(10)
    assert result is builder


def test_build_is_non_destructive():
    """Calling build() twice on the same builder should return equal instances."""
    builder = (
        CanonicalSystemDataBuilder()
        .with_calibration_id("x")
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
    )
    first = builder.build()
    second = builder.build()
    assert first == second


def test_multiple_items_accumulate():
    canonical = (
        CanonicalSystemDataBuilder()
        .with_acquire_mode("integrator")
        .with_acquire_mode("raw")
        .with_oscillator("osc0", 5_000_000_000)
        .with_oscillator("osc1", 6_000_000_000)
        .with_port("p0", 1000)
        .with_port("p1", 2000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_channel("ch1", "p1", 6_000_000_000)
        .with_qubit("q0", 0)
        .with_qubit("q1", 1)
        .build()
    )
    assert len(canonical.acquire_modes) == 2
    assert len(canonical.ports) == 2
    assert len(canonical.qubits) == 2


def test_build_payload_keys_mirror_canonical_fields():
    import dataclasses

    from qat.experimental.system_data.canonical.schema import CanonicalSystemData

    payload = _base().with_calibration_id("payload-test").build_payload()
    assert isinstance(payload, dict)
    assert CanonicalSystemDataBuilder.data_field in payload

    model = payload[CanonicalSystemDataBuilder.data_field]
    model.pop(CanonicalSystemDataBuilder.versioning_key)
    expected_keys = {f.name for f in dataclasses.fields(CanonicalSystemData)}

    assert set(model.keys()) == expected_keys
    assert model["calibration_id"] == "payload-test"


def test_build_payload_values_equal_build():
    import dataclasses

    builder = _base().with_calibration_id("eq-test")
    payload = builder.build_payload()
    canonical = builder.build()

    model = payload[CanonicalSystemDataBuilder.data_field]
    for f in dataclasses.fields(canonical):
        assert model[f.name] == getattr(canonical, f.name)


def test_full_chain_round_trips_all_fields():
    """All builder paths exercised; output fields should match what was added."""
    canonical = (
        CanonicalSystemDataBuilder()
        .with_calibration_id("full-test")
        .with_acquire_limit(100)
        .with_acquire_mode("integrator")
        .with_default_acquire_mode("integrator")
        .with_reset_method("passive")
        .with_default_reset_method("passive")
        .with_oscillator("osc0", 6_000_000_000)
        .with_port("p0", 1000, acquire_allowed=True)
        .with_channel("ch0", "p0", 6_000_000_000)
        .with_qubit("q0", 0)
        .with_external_resource("ext0")
        .with_metadata("source", "echo")
        .build()
    )

    assert canonical.calibration_id == "full-test"
    assert canonical.acquire_limit == 100
    assert canonical.acquire_modes[0].type == "integrator"
    assert canonical.default_acquire_mode == "integrator"
    assert canonical.reset_methods[0].type == "passive"
    assert canonical.default_reset_method == "passive"
    assert canonical.oscillators[0].id == "osc0"
    assert canonical.oscillators[0].frequency == 6_000_000_000
    assert canonical.ports[0].id == "p0"
    assert canonical.ports[0].acquire_allowed is True
    assert canonical.channels[0].id == "ch0"
    assert canonical.qubits[0].id == "q0"
    assert canonical.external_resources[0].id == "ext0"
    assert canonical.metadata[0].key == "source"
    assert canonical.metadata[0].value == "echo"


def test_with_qubit_mode_fields():
    """ModeData with all optional fields is preserved via with_qubit."""
    waveform = WaveformData(id="pi_pulse", shape="gaussian", width=100, amp=0.5)
    acquire = AcquireDefinitionData(id="acq0", delay=100, sync=True)
    method = LinearMapToRealMethodData(mean_z_map_args=(1 + 0j, 0j))
    mode = ModeData(
        id="measure_q0",
        channel_id="ch0",
        waveform_definitions=(waveform,),
        acquire_definitions=(acquire,),
        post_process_method=method,
        preselect_disallowed_states=frozenset({2}),
    )
    canonical = _base().with_qubit("q1", 1, modes=(mode,)).build()
    result_mode = canonical.qubits[1].modes[0]
    assert result_mode.waveform_definitions == (waveform,)
    assert result_mode.acquire_definitions == (acquire,)
    assert result_mode.post_process_method == method
    assert result_mode.preselect_disallowed_states == frozenset({2})


def test_with_qubit_mode_max_likelihood_post_process():
    method = MaxLikelihoodMethodData(
        states=(
            (0, MaxLikelihoodDiscriminateParams(location=1.0 + 0j)),
            (1, MaxLikelihoodDiscriminateParams(location=-1.0 + 0j)),
        )
    )
    mode = ModeData(id="measure_q0", channel_id="ch0", post_process_method=method)
    canonical = _base().with_qubit("q1", 1, modes=(mode,)).build()
    assert canonical.qubits[1].modes[0].post_process_method == method


def test_with_qubit_with_operations():
    step = DelayOperationStepData(mode_id="drive_q0", duration=500)
    op = OperationData(id="delay_op", operation_steps=(step,))
    canonical = _base().with_qubit("q1", 1, operations=(op,)).build()
    assert len(canonical.qubits[1].operations) == 1
    assert canonical.qubits[1].operations[0].id == "delay_op"
    assert canonical.qubits[1].operations[0].operation_steps == (step,)


def test_with_qubit_with_readout_probability():
    entries = (
        ProbabilityEntry(prepared_state=0, measured_state=0, probability=0.95),
        ProbabilityEntry(prepared_state=0, measured_state=1, probability=0.05),
        ProbabilityEntry(prepared_state=1, measured_state=0, probability=0.08),
        ProbabilityEntry(prepared_state=1, measured_state=1, probability=0.92),
    )
    canonical = (
        _base()
        .with_qubit(
            "q1", 1, readout_probability=ReadoutProbabilityData(probability_entries=entries)
        )
        .build()
    )
    assert canonical.qubits[1].readout_probability is not None
    assert canonical.qubits[1].readout_probability.probability_entries == entries
