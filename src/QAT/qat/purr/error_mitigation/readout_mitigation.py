import abc
from collections import defaultdict

import numpy as np
import regex as re
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Instruction, ResultsProcessing


class ApplyReadoutMitigation:
    name = "readout_base_class"

    @abc.abstractmethod
    def apply_error_mitigation(
        self, results: dict, qfile: QatFile, model: QuantumHardwareModel
    ):
        pass

    def get_mapping_and_qubits(self, qfile):
        """
        Result processing of form <creg_name>[<creg_index>]_<qubit_index>

        This dict is creg -> qubit number
        """
        mapping = {}
        qubits = []
        for instruction in qfile.meta_instructions:
            if isinstance(instruction, ResultsProcessing):
                var = instruction.variable
                creg, qubit = var.split("_", 1)
                qubit = int(qubit)
                qubits.append(qubit)
                creg_index = re.search("[a-zA-Z0-9]\[(.*)\]", creg).group(1)
                mapping[creg_index] = qubit
        return mapping, len(qubits)

    def process_results(self, results):
        if isinstance(new_results := list(results.values())[0], dict):
            results = new_results
        if list(results.values())[0] > 1:
            shots = sum(list(results.values()))
            results = {key: value / shots for key, value in results.items()}
        return results


class ApplyPostProcReadoutMitigation(ApplyReadoutMitigation):
    name = "post-processing_readout_base_class"

    def apply_error_mitigation(
        self, results: dict, qfile: QatFile, model: QuantumHardwareModel
    ):
        raise NotImplementedError()


class ApplyHybridReadoutMitigation(ApplyReadoutMitigation):
    name = "hybrid_readout_base_class"

    def apply_error_mitigation(
        self, results: dict, qfile: QatFile, model: QuantumHardwareModel
    ):
        raise NotImplementedError()


class ApplyLinearReadoutMitigation(ApplyPostProcReadoutMitigation):
    name = "linear_readout_mitigation"
    """
    {
        <qubit_number>: {
            "0|0": 1,
            "1|0": 1,
            "0|1": 1,
            "1|1": 1,
        }
    }
    """

    def generate_mitigated_probabilites(
        self,
        bitstring_value,
        bitstring,
        creg_index,
        qubit_count,
        mapping,
        linear_mitigation_mappings,
        result_probability,
        mitigated_results,
    ):
        other_value = "1" if bitstring_value == "0" else "0"
        new_sting = "".join(
            [
                i if index_ != creg_index else other_value
                for index_, i in enumerate(bitstring)
            ]
        )

        qubit_number = mapping[str(creg_index)]
        lin_mit = linear_mitigation_mappings[qubit_number]
        change_factor = lin_mit[f"{other_value}|{bitstring_value}"] / qubit_count
        unchange_factor = lin_mit[f"{bitstring_value}|{bitstring_value}"] / qubit_count

        mitigated_results[new_sting] += change_factor * result_probability
        mitigated_results[bitstring] += unchange_factor * result_probability
        return mitigated_results

    def apply_error_mitigation(
        self, results: dict, qfile: QatFile, model: QuantumHardwareModel
    ):
        results = self.process_results(results)
        linear_mitigation_mappings = model.error_mitigation.readout_mitigation.linear
        mitigated_results = defaultdict(lambda: 0)
        mapping, qubit_count = self.get_mapping_and_qubits(qfile)
        for bitstring, result_probability in results.items():
            for creg_index, bitstring_value in enumerate(bitstring):
                mitigated_results = self.generate_mitigated_probabilites(
                    bitstring_value,
                    bitstring,
                    creg_index,
                    qubit_count,
                    mapping,
                    linear_mitigation_mappings,
                    result_probability,
                    mitigated_results,
                )
        return mitigated_results

    def __repr__(self):
        return "Linear readout mitigation"


class ApplyMatrixReadoutMitigation(ApplyPostProcReadoutMitigation):
    name = "matrix_readout_mitigation"

    def remap_result(self, results, mapping, output_length):
        output = {}
        for bitstring, result in results.items():
            tmp_bit_string = ["0" for _ in range(output_length)]
            for i, bit in enumerate(bitstring):
                tmp_bit_string[mapping[str(i)]] = bit
            output["".join(tmp_bit_string)] = result
        return output

    def create_array_from_dict(self, algo_dict, n):
        tmp_array = {
            bin(i)[2:].zfill(n): algo_dict[bin(i)[2:].zfill(n)]
            if algo_dict.get(bin(i)[2:].zfill(n))
            else 0
            for i in range(2**n)
        }
        keys = sorted(tmp_array.keys(), key=lambda x: int(x, 2))
        output = []
        for key in keys:
            output.append(tmp_array[key])
        return np.array(output)

    def apply_error_mitigation(
        self, results: dict, qfile: QatFile, model: QuantumHardwareModel
    ):
        results = self.process_results(results)
        mapping, n = self.get_mapping_and_qubits(qfile)

        algo_data = self.remap_result(results, mapping, n)
        algo_data_array = self.create_array_from_dict(algo_data, n)

        data = np.matmul(
            model.error_mitigation.readout_mitigation.matrix,
            np.transpose(algo_data_array),
        )
        mitigated_data = {bin(i)[2:].zfill(n): data[i] for i in range(2**n)}
        inverted_map = {str(value): int(key) for key, value in mapping.items()}

        return self.remap_result(mitigated_data, inverted_map, n)

    def __repr__(self):
        return "Matrix readout mitigation"


class ReadoutMitigation(Instruction):
    pass


class PostProcessingReadoutMitigation(ReadoutMitigation):
    pass


class LinearReadoutMitigation(PostProcessingReadoutMitigation):
    pass


class MatrixReadoutMitigation(PostProcessingReadoutMitigation):
    pass


def get_readout_mitigation(instruction: ReadoutMitigation):
    if isinstance(instruction, LinearReadoutMitigation):
        return ApplyLinearReadoutMitigation()
    if isinstance(instruction, MatrixReadoutMitigation):
        return ApplyMatrixReadoutMitigation()
