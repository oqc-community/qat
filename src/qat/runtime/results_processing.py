# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from numbers import Number
from typing import Iterable

import numpy as np

# TODO: these functions might benefit from a little attention, and also add tests.
# they are just extracted from various `purr` packages


def binary_average(results_list):
    """
    Extracted from `purr.compiler.execution`.

    Averages all repeat results and returns a definitive 1/0 for each qubit measurement.
    """
    # If we have many sweeps/repeats loop through all of them and sum.
    if all([isinstance(val, list) for val in results_list]):
        binary_results = [binary_average(nested) for nested in results_list]
    else:
        binary_results = binary(results_list)

    return 1 if sum(binary_results) >= (len(binary_results) / 2) else 0


def binary(results_list):
    """
    Extracted from `purr.compiler.execution`.

    Changes all measurements to binary format.
    """
    if not isinstance(results_list, Iterable):
        return [results_list]

    results = []
    for item in results_list:
        if isinstance(item, Iterable):
            results.append(binary(item))
        elif isinstance(item, complex):  # If we're a flat register, just append.
            results.append(complex_to_binary(item))
        elif isinstance(item, float):
            results.append(0 if item > 0 else 1)
        else:
            results.append(item)
    return results


def complex_to_binary(number: complex):
    """
    Extracted from `purr.compiler.execution`.

    Base calculation for changing a complex measurement to binary form.
    """
    return 0 if number.real > 0 else 1


def numpy_array_to_list(array):
    """
    Extracted from `purr.compiler.execution`.

    Transform numpy arrays to a normal list.
    """
    if isinstance(array, np.ndarray):
        numpy_list = array.tolist()
        if len(numpy_list) == 1:
            return numpy_list[0]
        return numpy_list
    elif isinstance(array, list):
        list_list = [numpy_array_to_list(val) for val in array]
        if len(list_list) == 1:
            return list_list[0]
        return list_list
    else:
        return array


def binary_count(results_list, repeats):
    """
    Extracted from `qat.purr.compiler.runtime`.

    Returns a dictionary of binary number: count. So for a two qubit register it'll return
        the various counts for ``00``, ``01``, ``10`` and ``11``.
    """

    def flatten(res):
        """
        Combine binary result from the QPU into composite key result.
        Aka '0110' or '0001'
        """
        if isinstance(res, Iterable):
            return "".join([flatten(val) for val in res])
        else:
            return str(res)

    def get_tuple(res, index):
        return [val[index] if isinstance(val, (list, np.ndarray)) else val for val in res]

    binary_results = binary(results_list)

    # If our results are a single qubit then pretend to be a register of one.
    if (
        isinstance(next(iter(binary_results), None), Number)
        and len(binary_results) == repeats
    ):
        binary_results = [binary_results]

    result_count = dict()
    for qubit_result in [list(get_tuple(binary_results, i)) for i in range(repeats)]:
        key = flatten(qubit_result)
        value = result_count.get(key, 0)
        result_count[key] = value + 1

    return result_count
