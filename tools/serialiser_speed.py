from sys import getsizeof
import time

import matplotlib.pyplot as plt

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.logical import convert_to_other, get_default_logical_hardware
from qat.purr.compiler.builders import InstructionBuilder, SerialiserBackend
from qat.purr.compiler.runtime import get_builder

base_hw = get_default_echo_hardware(32)

def get_test_builder(qubits=32, layers=200, logical_hardware=False):
    """
    Make some random large builder to be serialised.
    """
    hardware = get_default_logical_hardware(qubit_count=qubits) if logical_hardware else get_default_echo_hardware(qubits)

    builder = get_builder(hardware)
    for layer in range(layers):
        for single_q_layer in range(qubits):
            builder.X(builder.model.get_qubit(single_q_layer))
        for two_q_layer in range(qubits - 1):
            builder.ECR(
                builder.model.get_qubit(two_q_layer),
                builder.model.get_qubit(two_q_layer + 1),
            )
    for qubit in range(qubits):
        builder.measure(builder.model.get_qubit(qubit))
    return builder


def serialisation_time(builder, backend: SerialiserBackend=None, remove_hw=False, logical=False):
    """
    Time isn't the best way to do this, but it's simple for the purposes of this test.
    """
    start = time.time()
    hw = builder.model
    if remove_hw:
        builder.model = None
    if logical:
        string = builder.serialise()
        size = getsizeof(string)
    else:
        string = builder.serialize(backend=backend)
        size = getsizeof(string)
    if logical:
        convert_to_other(string, base_hw)
    else:
        reconstitued = InstructionBuilder.deserialize(string, backend=backend)
    if remove_hw:
        reconstitued.model = hw
    end = time.time()
    return (end - start), size


if __name__ == "__main__":
    layers = [10, 50, 100, 200, 300, 500, 1000]
    json_times = []
    ujson_times = []
    logical_times = []
    json_sizes = []
    ujson_sizes = []
    logical_sizes = []

    for layer in layers:
        builder = get_test_builder(layers=layer)
        json_time, json_size =(
            serialisation_time(builder=builder, backend=SerialiserBackend.json)
        )
        json_times.append(json_time)
        json_sizes.append(json_size)
        ujson_time, ujson_size = (
            serialisation_time(builder=builder, backend=SerialiserBackend.ujson)
        )
        ujson_sizes.append(ujson_size)
        ujson_times.append(ujson_time)
        logical_time, logical_size = (
            serialisation_time(get_test_builder(layers=layer, logical_hardware=True), logical=True)
        )
        logical_times.append(logical_time)
        logical_sizes.append(logical_size)

    plt.plot(layers, json_times, label="json")
    plt.plot(layers, ujson_times, label="ujson")
    plt.plot(layers, logical_times, label="Logical hardware")
    plt.legend(loc="upper left")
    plt.xlabel("Layers")
    plt.ylabel("Time (s)")
    plt.show()

    plt.plot(layers, json_sizes, label="json")
    plt.plot(layers, ujson_sizes, label="ujson")
    plt.plot(layers, logical_sizes, label="Logical hardware")
    plt.legend(loc="upper left")
    plt.xlabel("Layers")
    plt.ylabel("Size (bytes)")
    plt.show()
