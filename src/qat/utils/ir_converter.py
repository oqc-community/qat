import copy

from qat.ir.instruction_list import InstructionList, find_all_instructions
from qat.ir.instructions import Instruction as PydanticInstruction
from qat.ir.instructions import Variable as PydanticVariable
from qat.ir.waveforms import AbstractWaveform as PydanticAbstractWaveform
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Instruction as LegacyInstruction
from qat.purr.compiler.instructions import Variable as LegacyVariable
from qat.purr.compiler.waveforms import AbstractWaveform as LegacyAbstractWaveform


class IRConverter:

    def __init__(self, model: QuantumHardwareModel = None):
        self.model = model
        self._target_cache = {}
        if self.model:
            self._build_target_dict()

        # create a mapping of instructions
        self.pydantic_instructions = {
            inst.__name__: inst
            for inst in find_all_instructions(
                [PydanticInstruction, PydanticAbstractWaveform]
            )
        }
        self.legacy_instructions = {
            inst.__name__: inst
            for inst in find_all_instructions([LegacyInstruction, LegacyAbstractWaveform])
        }

    def legacy_to_pydantic_instructions(self, instructions: list[LegacyInstruction]):
        """
        Takes a legacy instruction list and converts all instructions to Pydantic.

        Pydantic instructions are derived from Pydantic's BaseModel and have no references
        to the details of a hardware model other than the full_id of quantum components.
        """

        return InstructionList(
            instruction_list=[
                self._legacy_to_pydantic_instruction(instruction)
                for instruction in instructions
            ]
        )

    def pydantic_to_legacy_instructions(self, instructions: InstructionList):
        """
        Takes a Pydantic InstructionList and converts all instructions to legacy equivalents.

        Throughout the conversion, we have to retrieve QuantumComponents from the hardware
        model. To keep performance high, we use a cache that maps id -> component.
        """
        return [
            self._pydantic_to_legacy_instruction(instruction)
            for instruction in instructions.instruction_list
        ]

    def _legacy_to_pydantic_instruction(self, instruction: LegacyInstruction):
        """
        Converts a legacy instruction to a pydantic one. Replaces any references to
        QuantumComponents with their full_id.
        """

        pydantic_type = self.pydantic_instructions[type(instruction).__name__]

        # any serious performance problems by using copy?
        args = copy.copy(instruction.__dict__)

        for key, val in args.items():
            if isinstance(val, LegacyVariable):
                args[key] = PydanticVariable(
                    name=val.name, var_type=val.var_type, value=val.value
                )

        # The legacy instructions has inconsistencies with the name of how members
        # are saved and how they are provided in __init__. In these instances, we have
        # to manually define how to deal with these cases.
        if isinstance(instruction, self.legacy_instructions["PhaseShift"]):
            return pydantic_type(args["quantum_targets"][0], args["phase"])

        elif isinstance(instruction, self.legacy_instructions["FrequencyShift"]):
            return pydantic_type(args["quantum_targets"][0], args["frequency"])

        elif isinstance(instruction, self.legacy_instructions["ResultsProcessing"]):
            return pydantic_type(args["variable"], args["results_processing"])

        elif isinstance(instruction, self.legacy_instructions["PostProcessing"]):
            acquire_legacy = args.pop("quantum_targets")
            args["acquire"] = self._legacy_to_pydantic_instruction(acquire_legacy[0])

        elif isinstance(instruction, self.legacy_instructions["Assign"]):
            if isinstance(args["value"], LegacyVariable):
                val = args["value"]
                args["value"] = PydanticVariable(
                    name=val.name, var_type=val.var_type, value=val.value
                )
            elif isinstance(args["value"], list):
                itms = []
                for i, val in enumerate(args["value"]):
                    if isinstance(val, LegacyVariable):
                        itms.append(
                            PydanticVariable(
                                name=val.name, var_type=val.var_type, value=val.value
                            )
                        )
                    else:
                        itms.append(val)
                    args["value"] = itms
        return pydantic_type(**args)

    def _pydantic_to_legacy_instruction(self, instruction: PydanticInstruction):
        """
        Converts a pydantic instruction to a legacy one: requires us to repopulate
        any QuantumComponents from a hardware model which are acquired using the
        full_id.
        """

        if not self.model:
            raise (
                ValueError(
                    "IRConverter must be given a valid hardware model to deserialize."
                )
            )

        legacy_type = self.legacy_instructions[type(instruction).__name__]

        # any serious performance problems by using copy?
        args = copy.copy(instruction.__dict__)
        args.pop("inst")

        for key, val in args.items():
            if isinstance(val, PydanticVariable):
                args[key] = LegacyVariable(val.name, val.var_type, val.value)

        # pydantic instructions save quantum targets by id: repopulate them with
        # the correct pulse channel views.
        if "quantum_targets" in args:
            qts_str = args.pop("quantum_targets")
            if isinstance(qts_str, str):
                qts = self._fetch_quantum_target(qts_str)
            else:
                qts = [self._fetch_quantum_target(qt) for qt in qts_str]

        # there are a number of special cases that need a unique treatment
        if isinstance(instruction, self.pydantic_instructions["Jump"]):
            inst = legacy_type(args["target"], args["condition"])

        elif isinstance(instruction, self.pydantic_instructions["AbstractWaveform"]):
            inst = legacy_type(
                qts, args["width"], args["amp"], args["ignore_channel_scale"]
            )

        elif isinstance(instruction, self.pydantic_instructions["Pulse"]):
            inst = legacy_type(qts, **args)

        elif isinstance(instruction, self.pydantic_instructions["Acquire"]):
            args.pop("suffix_incrementor")
            inst = legacy_type(qts, **args)

        elif isinstance(instruction, self.pydantic_instructions["PostProcessing"]):
            acquire_pydantic = args.pop("acquire")
            acquire = self._pydantic_to_legacy_instruction(acquire_pydantic)
            result_needed = args.pop("result_needed")
            inst = legacy_type(acquire, **args)
            inst.result_needed = result_needed

        elif isinstance(instruction, self.pydantic_instructions["CustomPulse"]):
            inst = legacy_type(qts, args["samples"], args["ignore_channel_scale"])

        elif isinstance(instruction, self.pydantic_instructions["Assign"]):
            if isinstance(args["value"], PydanticVariable):
                val = args["value"]
                args["value"] = LegacyVariable(val.name, val.var_type, val.value)
            elif isinstance(args["value"], list):
                for i, val in enumerate(args["value"]):
                    if isinstance(val, PydanticVariable):
                        args["value"][i] = LegacyVariable(val.name, val.var_type, val.value)
                print([type(arg) for arg in args["value"]])
            inst = legacy_type(**args)

        elif isinstance(instruction, self.pydantic_instructions["QuantumInstruction"]):
            inst = legacy_type(qts, **args)

        else:
            inst = legacy_type(**args)

        return inst

    def _fetch_quantum_target(self, target: str):
        """
        Retrieves a QuantumDevice or a PulseChannel with a given full_id from a
        hardware model. A dictionary cache can be provided to store results to
        reduce the cost of repeatedly calculating full_id().
        """

        if len(self._target_cache) == 0:
            self._build_target_dict()

        channel = self._target_cache.get(target, None)
        if channel:
            return channel

        raise (
            ValueError(
                f"A quantum component with id {target} could not be found in "
                "the hardware model."
            )
        )

    def _build_target_dict(self):
        """
        Builds a map of IDs to quantum components ahead of time.
        """
        for target in self.model.quantum_devices.keys():
            self._target_cache[target] = self.model.quantum_devices[target]

        # the targets of instructions are PulseChannelViews, so we have to find
        # the pulse channel by searching through each quantum device.
        for device in self.model.quantum_devices.values():
            for pc in device.pulse_channels.values():
                self._target_cache[pc.full_id()] = pc
