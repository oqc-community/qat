# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import inspect
from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

import qat.experimental.system_data.canonical.schema as canonical_schema
from qat.experimental.system_data.canonical.schema import (
    AcquireDefinitionData,
    AcquireModeData,
    AcquireOperationStepData,
    AttributeEntry,
    CanonicalSystemData,
    ChannelData,
    DelayOperationStepData,
    ExternalResourceData,
    LinearMapToRealMethodData,
    MaxLikelihoodDiscriminateParams,
    MaxLikelihoodMethodData,
    ModeData,
    OperationData,
    OscillatorData,
    PortData,
    ProbabilityEntry,
    PulseOperationStepData,
    QubitCouplingData,
    QubitData,
    ReadoutProbabilityData,
    ResetData,
    SyncOperationStepData,
    TwoQubitGateFidelityData,
    WaveformData,
)

SCHEMA_CASES = [
    (AttributeEntry, {"key": "k", "value": 1}),
    (ExternalResourceData, {"id": "res0"}),
    (OscillatorData, {"id": "osc0", "frequency": 5_000_000_000}),
    (PortData, {"id": "p0", "sample_time": 1_000}),
    (ChannelData, {"id": "ch0", "port_id": "p0", "frequency": 5_000_000_000}),
    (WaveformData, {"id": "wf0"}),
    (
        ReadoutProbabilityData,
        {
            "probability_entries": (
                ProbabilityEntry(prepared_state=0, measured_state=0, probability=0.99),
                ProbabilityEntry(prepared_state=0, measured_state=1, probability=0.01),
            )
        },
    ),
    (AcquireDefinitionData, {"id": "acq0"}),
    (ModeData, {"id": "mode0", "channel_id": "ch0"}),
    (
        AcquireOperationStepData,
        {"mode_id": "mode0", "acquire_definition": "acq0"},
    ),
    (DelayOperationStepData, {"mode_id": "mode0", "duration": 1_000}),
    (
        PulseOperationStepData,
        {"mode_id": "mode0", "waveform_definition": "wf0"},
    ),
    (SyncOperationStepData, {"mode_ids": frozenset({"mode0"})}),
    (OperationData, {"id": "op0"}),
    (
        LinearMapToRealMethodData,
        {"method": "linear_map_complex_to_real"},
    ),
    (
        MaxLikelihoodDiscriminateParams,
        {"location": 1 + 0j},
    ),
    (
        MaxLikelihoodMethodData,
        {
            "states": (
                (0, MaxLikelihoodDiscriminateParams(location=1 + 0j)),
                (1, MaxLikelihoodDiscriminateParams(label="1", location=-1 + 0j)),
            )
        },
    ),
    (QubitData, {"id": "q0", "index": 0}),
    (
        QubitCouplingData,
        {
            "source_qubit_id": "q0",
            "target_qubit_id": "q1",
            "gate_fidelities": (TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
        },
    ),
    (ResetData, {"type": "passive"}),
    (AcquireModeData, {"type": "integrator"}),
    (CanonicalSystemData, {"calibration_id": "cal0", "acquire_limit": 100}),
]


def assert_kw_only_signature(cls):
    """Assert that the dataclass constructor accepts keyword-only parameters."""

    for name, param in inspect.signature(cls).parameters.items():
        if name == "self":
            continue
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def assert_dataclass_contract(cls):
    """Assert the schema dataclass contract (frozen, slotted, keyword-only)."""

    assert cls.__dataclass_params__.frozen
    assert hasattr(cls, "__slots__")
    assert all(field_info.kw_only for field_info in fields(cls))
    assert_kw_only_signature(cls)


def _schema_dataclass_types():
    """Return dataclass types defined directly in the canonical schema module."""

    return [
        obj
        for _, obj in inspect.getmembers(canonical_schema, inspect.isclass)
        if obj.__module__ == canonical_schema.__name__ and is_dataclass(obj)
    ]


@pytest.mark.parametrize("cls", _schema_dataclass_types(), ids=lambda cls: cls.__name__)
def test_schema_dataclass_contract_via_inspection(cls):
    """Validate all schema dataclasses retain their structural contract."""

    assert_dataclass_contract(cls)


@pytest.mark.parametrize("cls, kwargs", SCHEMA_CASES)
def test_schema_dataclass_runtime_immutability_and_slots(cls, kwargs):
    """Verify dataclass instances are immutable and reject undeclared slot fields."""

    instance = cls(**kwargs)

    with pytest.raises(FrozenInstanceError, match=".*"):
        first_field_name = fields(cls)[0].name
        setattr(instance, first_field_name, object())

    with pytest.raises(AttributeError, match=".*"):
        object.__setattr__(instance, "not_a_field", 1)


@pytest.mark.parametrize("cls, kwargs", SCHEMA_CASES)
def test_schema_dataclasses_reject_positional_arguments(cls, kwargs):
    """Ensure schema construction is keyword-only for all covered dataclasses."""

    with pytest.raises(TypeError, match=".*"):
        positional_args = tuple(kwargs.values())
        cls(*positional_args)


def test_canonical_system_data_accepts_nested_records():
    """Build a representative nested schema object and verify record wiring."""

    metadata = (AttributeEntry(key="release", value="2026.06"),)
    external_resource = ExternalResourceData(id="res0", object_type="Cluster")
    oscillator = OscillatorData(
        id="osc0",
        frequency=5_500_000_000,
        external_resource_id="res0",
    )
    port = PortData(
        id="p0",
        sample_time=1_000,
        external_resource_id="res0",
        acquire_allowed=True,
    )
    channel = ChannelData(
        id="ch0",
        port_id="p0",
        frequency=6_000_000_000,
        oscillator_reference="osc0",
    )
    waveform = WaveformData(id="wf0", shape="gaussian", width=40_000)
    acquire = AcquireDefinitionData(id="acq0", delay=10_000, sync=True)
    post_process = MaxLikelihoodMethodData(
        states=(
            (0, MaxLikelihoodDiscriminateParams(location=1 + 0j)),
            (1, MaxLikelihoodDiscriminateParams(label="1", location=-1 + 0j)),
        ),
        p_min=0.05,
    )
    mode = ModeData(
        id="mode0",
        channel_id="ch0",
        waveform_definitions=(waveform,),
        acquire_definitions=(acquire,),
        post_process_method=post_process,
        preselect_disallowed_states=frozenset({1}),
    )
    operation = OperationData(
        id="op0",
        operation_steps=(
            PulseOperationStepData(mode_id="mode0", waveform_definition="wf0"),
            AcquireOperationStepData(mode_id="mode0", acquire_definition="acq0"),
            DelayOperationStepData(mode_id="mode0", duration=20_000),
            SyncOperationStepData(mode_ids=frozenset({"mode0"})),
        ),
    )
    qubit = QubitData(
        id="q0",
        index=0,
        modes=(mode,),
        operations=(operation,),
    )
    coupling = QubitCouplingData(
        source_qubit_id="q0",
        target_qubit_id="q1",
        gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
    )
    acquire_mode = AcquireModeData(type="integrator")
    reset = ResetData(type="passive")

    system_data = CanonicalSystemData(
        calibration_id="cal0",
        acquire_limit=100,
        acquire_modes=(acquire_mode,),
        default_acquire_mode="integrator",
        reset_methods=(reset,),
        default_reset_method="passive",
        oscillators=(oscillator,),
        ports=(port,),
        channels=(channel,),
        qubits=(qubit,),
        couplings=(coupling,),
        external_resources=(external_resource,),
        metadata=metadata,
    )

    assert system_data.calibration_id == "cal0"
    assert system_data.acquire_modes[0].type == "integrator"
    assert system_data.default_acquire_mode == "integrator"
    assert system_data.reset_methods[0].type == "passive"
    assert system_data.default_reset_method == "passive"
    assert system_data.oscillators[0].id == "osc0"
    assert system_data.ports[0].acquire_allowed
    assert system_data.channels[0].port_id == "p0"
    assert system_data.channels[0].oscillator_reference == "osc0"
    assert system_data.qubits[0].modes[0].waveform_definitions[0].id == "wf0"
    assert system_data.qubits[0].operations[0].operation_steps[0].mode_id == "mode0"
    assert system_data.qubits[0].modes[0].post_process_method is not None
    assert system_data.qubits[0].modes[0].post_process_method.method == "max_likelihood"
    assert system_data.qubits[0].modes[0].preselect_disallowed_states == frozenset({1})
    assert system_data.couplings[0].source_qubit_id == "q0"
    assert system_data.couplings[0].gate_fidelities[0].gate == "cx"
    assert system_data.couplings[0].gate_fidelities[0].fidelity == pytest.approx(0.99)
    assert system_data.external_resources[0].id == "res0"
    assert system_data.metadata[0].key == "release"


def test_readout_probability_data_construction():
    """Confirm readout confusion probabilities can be constructed with multiple entries."""

    confusion = ReadoutProbabilityData(
        probability_entries=(
            ProbabilityEntry(prepared_state=0, measured_state=0, probability=0.95),
            ProbabilityEntry(prepared_state=0, measured_state=1, probability=0.05),
            ProbabilityEntry(prepared_state=1, measured_state=0, probability=0.07),
            ProbabilityEntry(prepared_state=1, measured_state=1, probability=0.93),
        )
    )

    assert len(confusion.probability_entries) == 4
    assert confusion.probability_entries[0].prepared_state == 0
    assert confusion.probability_entries[0].measured_state == 0
    assert confusion.probability_entries[3].probability == 0.93


@pytest.mark.parametrize(
    "weights",
    (
        None,
        (1.0, -0.5, 0.0),
        (1.0 + 0.0j, -0.25 + 0.75j),
    ),
)
def test_acquire_definition_data_accepts_weight_tuples(weights):
    """Ensure acquisition weights support optional float/complex coefficient tuples."""

    acquire = AcquireDefinitionData(id="acq0", weights=weights)

    assert acquire.weights == weights


@pytest.mark.parametrize(
    ("transform", "offset"),
    (
        (None, None),
        (((1.0, 0.0), (0.0, 1.0)), (0.0, 0.0)),
        (((0.5, -0.25), (0.25, 0.5)), (0.1, -0.2)),
    ),
)
def test_max_likelihood_method_data_accepts_affine_transform_and_offset(transform, offset):
    """Ensure canonical ML method accepts optional typed affine IQ transform data."""

    method = MaxLikelihoodMethodData(
        states=((0, MaxLikelihoodDiscriminateParams(location=1 + 0j)),),
        transform=transform,
        offset=offset,
    )

    assert method.transform == transform
    assert method.offset == offset
