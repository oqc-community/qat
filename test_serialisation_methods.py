
from ctypes import sizeof
import sys
from qat.purr.backends.echo import Connectivity, get_default_echo_hardware
from qat.purr.backends.logical import LogicalBuilder, convert_to_other, get_default_logical_hardware
from qat.purr.compiler.config import CompilerConfig, Qasm2Optimizations, QuantumResultsFormat, TketOptimizations
from qat.purr.compiler.frontends import QASMFrontend
from qat.purr.compiler.runtime import get_builder
from qat.purr.integrations.qasm import Qasm2Parser, Qasm3Parser



echo = get_default_echo_hardware(qubit_count=8, connectivity=Connectivity.Ring)
logical = get_default_logical_hardware(qubit_count=8)

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg b[4];
h q[0];
cx q[0], q[1];
cx q[0], q[2];
cx q[0], q[3];
measure q -> b;"""

def logical_parse():
    parser = Qasm2Parser()
    frontend = QASMFrontend()
    builder = LogicalBuilder(logical)

    opts = Qasm2Optimizations()
    opts.tket_optimizations = TketOptimizations.Two
    config = CompilerConfig(
        repeats=300,
        repetition_period=1e-4,
        optimizations=opts,
        results_format=QuantumResultsFormat().binary_count(),
    )

    builder = parser.parse(
        builder,
        qasm,
    )
    builder, _ = frontend.parse(qasm, logical, config)
    instructions = builder.instructions
    for instruction in instructions:
        print(instruction)
    output = builder.serialise()
    print(output)

    converted_builder = convert_to_other(output, echo)
    for instuct in converted_builder.instructions:
        if instuct:
            print(instuct)
            
    print(sys.getsizeof(output), sys.getsizeof(converted_builder.serialize()))


logical_parse()