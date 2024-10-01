import abc
from typing import Dict

import numpy as np
from compiler_config.config import ErrorMitigationConfig

from qat.purr.compiler.hardware_models import QuantumHardwareModel


class ApplyReadoutMitigation:
    name = "readout_base_class"

    @abc.abstractmethod
    def apply_error_mitigation(
        self, results: dict, mapping: Dict, model: QuantumHardwareModel
    ):
        pass

    def process_results(self, results):
        if isinstance(new_results := list(results.values())[0], dict):
            results = new_results
        if list(results.values())[0] > 1:
            shots = sum(list(results.values()))
            results = {key: value / shots for key, value in results.items()}
        return results


class ApplyPostProcReadoutMitigation(ApplyReadoutMitigation):
    name = "post_processing_readout_base_class"

    def apply_error_mitigation(
        self, results: dict, mapping: Dict, model: QuantumHardwareModel
    ):
        raise NotImplementedError()


class ApplyHybridReadoutMitigation(ApplyReadoutMitigation):
    name = "hybrid_readout_base_class"

    def apply_error_mitigation(
        self, results: dict, mapping: Dict, model: QuantumHardwareModel
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

    def apply_error_mitigation(
        self, results: dict, mapping: Dict, model: QuantumHardwareModel
    ):
        results = self.process_results(results)
        qubit_count = len(mapping)
        for i in range(qubit_count):
            qubit = mapping[str(i)]
            noise_map = model.error_mitigation.readout_mitigation.linear[str(qubit)]
            results = self.apply_correction_qubit(results, i, noise_map)
        return results

    def apply_correction_qubit(self, results, index, noise_map):
        noise_matrix = np.zeros((2, 2))
        for input_bit in [0, 1]:
            for output_bit in [0, 1]:
                noise_matrix[output_bit, input_bit] = noise_map[f"{output_bit}|{input_bit}"]
        inv_noise_matrix = np.linalg.inv(noise_matrix)
        corrected_results = {}
        for bitstring, probability in results.items():
            corrected_results.setdefault(bitstring, 0.0)
            bit_value = int(bitstring[index])
            other_bit_value = 1 - bit_value
            corrected_results[bitstring] += (
                probability * inv_noise_matrix[bit_value, bit_value]
            )
            other_bitstring = "".join(
                [
                    bitstring[i] if i != index else str(other_bit_value)
                    for i in range(len(bitstring))
                ]
            )
            if other_bitstring in results:
                corrected_results.setdefault(other_bitstring, 0.0)
                corrected_results[other_bitstring] += (
                    probability * inv_noise_matrix[other_bit_value, bit_value]
                )

        # TODO - check validity of ignoring negative probabilities
        corrected_results = {
            key: probability
            for key, probability in corrected_results.items()
            if probability > 0
        }

        return corrected_results

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
            bin(i)[2:].zfill(n): (
                algo_dict[bin(i)[2:].zfill(n)] if algo_dict.get(bin(i)[2:].zfill(n)) else 0
            )
            for i in range(2**n)
        }
        keys = sorted(tmp_array.keys(), key=lambda x: int(x, 2))
        output = []
        for key in keys:
            output.append(tmp_array[key])
        return np.array(output)

    def apply_error_mitigation(self, results: dict, mapping, model: QuantumHardwareModel):
        n = len(mapping)
        results = self.process_results(results)
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


def get_readout_mitigation(mitigation_config: ErrorMitigationConfig):
    mitigators = []
    if ErrorMitigationConfig.LinearMitigation in mitigation_config:
        mitigators.append(ApplyLinearReadoutMitigation())
    if ErrorMitigationConfig.MatrixMitigation in mitigation_config:
        mitigators.append(ApplyMatrixReadoutMitigation())
    return mitigators
