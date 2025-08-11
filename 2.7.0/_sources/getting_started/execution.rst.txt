.. _execution:

Execution 
------------------------

QAT also provides utility for executing quantum programs and interpreting the results. 
In the pipelines, we saw that programs could be executed using :code:`QAT().execute(pkg)`. 
Here we break this down into the lower-level details. In particular, execution in QAT is 
composed of two elements: the **Engine** and the **Runtime**.

Below is a diagram that explains how execution is done in QAT, in particular, for the 
:class:`SimpleRuntime <qat.runtime.simple.SimpleRuntime>`. The native code is fed to the
runtime, which passes it to the engine. The engine communicates with the target to 
execute the program and fetch the results. These results then enter the post-processing 
pipeline - a series of passes that each mutate the results according the program. The 
mutated readout results are returned to the user.

.. image:: images/runtime.png
    :width: 400
    :align: center

.. contents::


Engines 
***********************

The :class:`NativeEngine <qat.engines.native.NativeEngine>` is the base class used to 
implement an engine. The engine is expected to uphold a contract with the Runtime: 

* Packages can be executed through the method 
  :meth:`NativeEngine.execute <qat.engines.native.NativeEngine.execute>`, which expects to 
  receive an :class:`Executable <qat.runtime.executables.Executable>` (the native code) as
  an argument.
* In return, the engine returns the results to the runtime in an expected
  format, which is a dictionary of acquisition results (one result per acquisition). The
  result is an array of readout acquisitions, whose shape will depend on the acquisition
  mode. The key for the acquisition in the dictionary is the :attr:`output_variable` stored
  in the :class:`AcquireData <qat.runtime.executables.AcquireData>`.
* The number of shots to execute is stored in the attribute :attr:`compiled_shots`. Note
  that while the total number of shots in a program might be larger than the
  :attr:`compiled_shots`, sometimes the target cannot support the required amount of shots.
  When this is the case, shots will be batched.

Engines available in QAT 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following engines are available to use within QAT. Engines written for proprietary OQC 
hardware is not available here.

* :class:`EchoEngine <qat.engines.waveform_v1.echo.EchoEngine>`: an engine compatible only
  with the :class:`WaveformV1Backend <qat.backend.waveform_v1.codegen.WaveformV1Backend>`
  that simply "echos" back the readout pulses, primarily used for testing purposes.
* :class:`ZeroEngine <qat.engines.zero.ZeroEngine>`: returns all readout responses as
  zeroes, again used for testing purposes.
* :class:`QiskitEngine <qat.purr.backends.qiskit_simulator.QiskitEngine>`: a legacy engine 
  that simulates quantum circuits using Qiskit's AerSimulator. To be refactored to make full 
  use of the pipelines API.
* :class:`RealtimeChipSimEngine <qat.purr.backends.realtime_chip_simulator.RealtimeChipSimEngine>`:
  OQC's home-made simulator for accurate and realistic simulation of superconducting qubits.
  Also a legacy engine and needs to be refactored to make full use of the pipelines API.

Echo engine example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example, let us use the :class:`EchoEngine <qat.engines.waveform_v1.echo.EchoEngine>`
to execute a QASM2 program. For simplicity, we will make use of a pipeline to compile the
program, but then use to engine independently to execute the program.

.. code-block:: python 
    :linenos:
    
    from qat import QAT
    from qat.pipelines.echo import echo8
    from qat.engines.waveform_v1 import EchoEngine
    from compiler_config.config import CompilerConfig, Tket

    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    config = CompilerConfig(repeats=10, optimizations=Tket().disable())

    core = QAT()
    core.pipelines.add(echo8, default=True)
    pkg, _ = core.compile(qasm_str, config)
    results = EchoEngine().execute(pkg)

The results returned as a dictionary: the keys correspond to output variables assigned 
to the readouts at compilation, in this case, it has the format :code:`c[{clbit}]_{qubit}`,
where :code:`clbit` corresponds to the bit specified in the QASM program, and the
:code:`qubit` denotes the qubit that is read out (note this may differ to what is 
specified in the QASM program if optimizations are used). Since the
:attr:`AcquireMode.INTEGRATOR <qat.purr.compiler.instructions.AcquireMode>` is used by 
default for readout acquisitions, the values in the dictionary are arrays with one readout 
per shot. For this example, the results are:


.. code-block:: python 

    results = {
        'c[0]_0': array([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j]),
        'c[1]_1': array([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j])
    }

Connection handling with engines 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes an engine requires a connection to be made with the target. Connection
capabilities can be specified by mixing in a
:class:`ConnectionMixin <qat.engines.native.ConnectionMixin>`.
To demonstrate how connection handling can be specified, see the following example, which 
adds a mock connection to the :class:`ZeroEngine <qat.engines.zero.ZeroEngine>`.

.. code-block:: python 
    :linenos:

    from qat.engines import ConnectionMixin
    from qat.engines.zero import ZeroEngine

    class NewEngine(ZeroEngine, ConnectionMixin):
        is_connected: bool = False 

        def connect(self):
            self.is_connected = True
            print("Engine has connected.")
            return self.is_connected
        
        def disconnect(self):
            self.is_connected = False
            print("Engine has disconnected.")
            return self.is_connected


Runtimes 
***********************

The Runtime is the object that is used to fully execute a program. When provided with a 
package, it makes calls to the engine to execute the "quantum parts" of the program, and
then runs the results it receives through a post-processing pipeline to execute the 
"classical parts". See :mod:`qat.runtime.passes` for a full list of post-processing passes 
available. The standard runtime to use is the
:class:`SimpleRuntime <qat.runtime.simple.SimpleRuntime>`, which simply calls the 
engine (possibly multiple times if the shots are batched) and then processes the results.
In the future, there may be more complex runtimes such as hybrid runtimes that allow for a
more comprehensive interplay of classical and quantum computation.

For engines where a connection is required, the Runtime can be provided a
:class:`ConnectionMode <qat.runtime.connection.ConnectionMode>` flag that instructs the
runtime on how the connection should be handled. For example, if a connection should always
be maintained for the entire lifetime of a runtime, we can use the flag 
:attr:`ConnectionMode.ALWAYS <qat.runtime.connection.ConnectionMode>`. Alternatively, if 
we want to delegate the responsibility of connection to the user, we can use the 
:attr:`ConnectionMode.MANUAL <qat.runtime.connection.ConnectionMode>` flag.

Simple runtime 
^^^^^^^^^^^^^^^^^^^^^^^^

The following example shows how to use the
:class:`SimpleRuntime <qat.runtime.simple.SimpleRuntime>` with a 
:class:`ZeroEngine <qat.engines.zero.ZeroEngine>` and a custom pipeline. For completeness,
it also shows how to add a connection flag, although it will be of no use here as the 
:class:`ZeroEngine <qat.engines.zero.ZeroEngine>` does not require a connection!

.. code-block:: python 
    :linenos:

    from qat import QAT
    from qat.pipelines.echo import echo8
    from qat.engines.zero import ZeroEngine
    from qat.runtime import SimpleRuntime
    from qat.runtime.connection import ConnectionMode
    from qat.passes.pass_base import PassManager
    from compiler_config.config import CompilerConfig, QuantumResultsFormat
    from qat.runtime.transform_passes import (
        AssignResultsTransform,
        InlineResultsProcessingTransform,
        PostProcessingTransform,
        ResultTransform
    )

    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    config = CompilerConfig(repeats=10, results_format=QuantumResultsFormat().binary_count())

    core = QAT()
    core.pipelines.add(echo8, default=True)
    pkg, _ = core.compile(qasm_str, config)

    pipeline = (
        PassManager()
        | PostProcessingTransform()
        | InlineResultsProcessingTransform()
        | AssignResultsTransform()
        | ResultTransform()
    )

    runtime = SimpleRuntime(ZeroEngine(), pipeline, ConnectionMode.ALWAYS)
    results = runtime.execute(pkg, compiler_config=config)

Since the Runtime takes care of post-processing responsibilities, the results returned look
quite a bit different to what was returned from the engine:

.. code-block:: python 

    results = {'c': {'11': 10}}

Legacy runtime 
^^^^^^^^^^^^^^^^^^^^^^^^

QAT pipelines also have support for legacy engines through the
:class:`LegacyRuntime <qat.runtime.legacy.LegacyRuntime>`. For example, we can
define a runtime for the RTCS:

.. code-block:: python 
    :linenos:

    from qat.runtime import LegacyRuntime
    from qat.model.loaders.legacy import RTCSModelLoader
    from qat.purr.backends.realtime_chip_simulator import RealtimeChipSimEngine

    model = RTCSModelLoader().load()
    runtime = LegacyRuntime(RealtimeChipSimEngine(model))


.. warning:: 

    Legacy engines can vary in the post-processing responsibilities that they carry out.
    An appropriate post-processing pipeline must be picked to match the legacy engine.