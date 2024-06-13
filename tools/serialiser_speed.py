import time

import matplotlib.pyplot as plt

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder, SerialiserBackend
from qat.purr.compiler.runtime import get_builder


def get_test_builder(qubits=32, layers=200):
    """
    Make some random large builder to be serialised.
    """
    hardware = get_default_echo_hardware(qubits)

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


def serialisation_time(builder, backend: SerialiserBackend):
    """
    Time isn't the best way to do this, but it's simple for the purposes of this test.
    """
    start = time.time()
    string = builder.serialize(backend=backend)
    reconstitued = InstructionBuilder.deserialize(string, backend=backend)
    end = time.time()
    return end - start


if __name__ == "__main__":
    layers = [10, 50, 100, 200, 300, 500, 1000]
    json_times = []
    ujson_times = []

    for layer in layers:
        builder = get_test_builder(layers=layer)
        json_times.append(
            serialisation_time(builder=builder, backend=SerialiserBackend.json)
        )
        ujson_times.append(
            serialisation_time(builder=builder, backend=SerialiserBackend.ujson)
        )

    plt.plot(layers, json_times, label="json")
    plt.plot(layers, ujson_times, label="ujson")
    plt.legend(loc="upper left")
    plt.show()
