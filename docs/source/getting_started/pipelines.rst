Pipelines
---------------------

The primary way to access QAT's compilation and execution capabilities is through its
pipelines API. Pipelines allow us to define the compilation and execution process in a 
configurable and modular way.

.. contents::

Using QAT pipelines
***********************

QAT has a number of pipelines that are pre-defined and ready to use. We start by creating a
a :class:`QAT <qat.core.qat.QAT>` object.

.. code-block:: python

    from qat import QAT 
    core = QAT()

We can now add a pipeline to it. Let's add a pipeline that uses the 
:class:`EchoEngine <qat.engines.waveform_v1.echo.EchoEngine>`:

.. code-block:: python 

    from qat.pipelines.echo import echo8
    core.pipeline.add(echo8, default=True)

This will add the "echo8" pipeline to :code:`core`, which can be used to compile and execute
programs using a simulator that simply returns all readouts as zeroes. The "8" in "echo8"
specifies that the simulator has 8 qubits available. The pipeline already has its 
compilation modules chosen so that it is compatible with the echo engine: we'll cover this 
in more detail in :ref:`compilation`. Now we can use this to execute a QASM program.

.. code-block:: python 
    :linenos:

    from qat import QAT 
    from qat.pipelines.echo import echo8
    from compiler_config.config import CompilerConfig, QuantumResultsFormat

    core = QAT()
    core.pipelines.add(echo8, default=True)

    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    config = CompilerConfig(results_format=QuantumResultsFormat().binary_count())
    results, metrics = core.run(qasm_str, config, pipeline="echo8")

There's a couple of things to unpack here.

* The program :code:`qasm_str` describes a simple QASM2 program to create a bell state on
  two qubits.
* Finally, you will note that :code:`metrics` is returned. This is an object that contains
  various metrics regarding compilation, such as circuits after they have been optimized, or
  the total number of pulse-level instructions. Since the echo pipeline just returns zeros
  for readouts, the result returned here is :code:`results = {'c': {'00': 1000}}`.

We could also achieve the same workflow by compiling and executing separately:

.. code-block:: python 
    
    pkg, metrics = core.compile(qasm_str, config, pipeline="echo8")
    results, metrics = core.execute(pkg, config, pipeline="echo8")

The package :code:`pkg` contains the native instructions to be sent to the target (in this
case, the echo simulator).

Default pipelines that are available in QAT 
*********************************************

There are a number of pipelines in QAT that are available to use off-the-shelf.

* :mod:`qat.pipelines.echo`: Pipelines that execute using the
  :class:`EchoEngine <qat.engines.waveform_v1.echo.EchoEngine>`. The pipelines available
  by default are :attr:`echo8`, :attr:`echo16`, :attr:`echo32`. For a custom amount of
  qubits, the method :meth:`get_pipeline <qat.pipelines.echo.get_pipeline>` can be used.

There are also pipelines that use legacy hardware and engines, but wrapped in the new 
pipeline API:

* :mod:`qat.pipelines.legacy.echo`: Pipelines that execute using the legacy
  :class:`EchoEngine <qat.purr.backends.echo.EchoEngine>`. The pipelines available
  by default are :attr:`legacy_echo8`, :attr:`legacy_echo16`, :attr:`lgeacy_echo32`. For a
  custom amount of qubits, the method
  :meth:`get_pipeline <qat.pipelines.legacy.echo.get_pipeline>` can be used.
* :mod:`qat.pipelines.legacy.rtcs`: Pipelines that execute using the legacy
  :class:`RealtimeChipSimEngine <qat.purr.backends.realtime_chip_simulator.RealtimeChipSimEngine>`.
  The only available pipeline is for two qubits, :attr:`legacy_rtcs2`.
* :mod:`qat.pipelines.legacy.qiskit`: Pipelines that execute using the legacy
  :class:`QiskitEngine <qat.purr.backends.qiskit_simulator.QiskitEngine>`. The available 
  pipelines are :attr:`legacy_qiskit8`, :attr:`legacy_qiskit16` and :attr:`legacy_qiskit32`.
  For a custom amount of qubits, the method
  :meth:`get_pipeline <qat.pipelines.legacy.qiskit.get_pipeline>` can be used.



Defining custom pipelines
*************************

Pipelines in QAT are highly customisable to allow for diverse compilation behaviour for a 
range of targets, such as live hardware or custom simulators. Compilation is broken down
into three parts: the **frontend**, the **middleend** and the **backend**. We will not go
into the details of each module here, but they will be covered in :ref:`compilation`.
Similarly,the execution part of the pipeline is defined by two objects: the **engine** and
the **runtime**. The engine acts as an adapter to the target, and deals with communicating
the instructions and results from the runtime to the target. The runtime handles the the
engine, and deals with software post-processing of the results. See :ref:`execution` for 
more details.

Let us quickly show how to define a custom pipeline by recreating the "echo8" pipeline.

.. code-block:: python
    :linenos:

    from qat import QAT 
    from qat.core.pipeline import Pipeline
    from qat.frontend.frontends import DefaultFrontend
    from qat.middleend.middleends import DefaultMiddleend
    from qat.backend.waveform_v1 import WaveformV1Backend
    from qat.engines.waveform_v1 import EchoEngine
    from qat.runtime import SimpleRuntime
    from qat.model.loaders.legacy import EchoModelLoader
    from compiler_config.config import CompilerConfig, QuantumResultsFormat

    model = EchoModelLoader(8).load()
    new_echo8 = Pipeline(
        name="new_echo8",
        frontend=DefaultFrontend(model),
        middleend=DefaultMiddleend(model),
        backend=WaveformV1Backend(model),
        runtime=SimpleRuntime(EchoEngine()),
        model=model
    )

    core = QAT()
    core.pipelines.add(new_echo8)

    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    config = CompilerConfig(results_format=QuantumResultsFormat().binary_count())
    results, metrics = core.run(qasm_str, config, pipeline="new_echo8")

Notice that the :class:`EchoEngine <qat.engines.waveform_v1.echo>` and the
:class:`WaveformV1Backend <qat.backend.waveform_v1.codegen.WaveformV1Backend>` are both
contained in a :code:`waveform_v1` package. This is not by coincidence: the engine has to be
appropriately picked to match the code generated from the backend. There will be more
details on the responsibilities of backends and engines in later sections.


Defining pipelines using a configuration file 
***********************************************

So far we have manually imported pipelines and added them to a
:class:`QAT <qat.core.qat.QAT>` to use the pipeline API :code:`QAT.compile` and
:code:`QAT.execute`. However, we can specify some default pipelines to use via a
configuration file.

.. code-block:: yaml
    :linenos:

    MAX_REPEATS_LIMIT: 1000
    PIPELINES:
    - name: echo8-alt
      pipeline: qat.pipelines.echo.echo8
      default: false
    - name: echo16-alt
      pipeline: qat.pipelines.echo.echo16
      default: true
    - name: echo32-alt
      pipeline: qat.pipelines.echo.echo32
      default: false
    - name: echo6-alt
      pipeline: qat.pipelines.echo.get_pipeline
      hardware_loader: echo6loader
      default: false

    HARDWARE:
    - name: echo6loader
      loader: qat.model.loaders.legacy.EchoModelLoader
      init:
        qubit_count: 6

This file currently allows us to specify the maximum number of shots that can be done for 
each job, and a number of pipelines.  Notice that the first three pipelines just point to 
an already defined pipeline. The fourth points to a function that lets us provide our own
hardware model, which is specified under :code:`HARDWARE`. 

To use this within QAT, we can simply use the directory of the file to instantiate the 
:class:`QAT <qat.core.qat.QAT>` object.

.. code-block:: python 
    :linenos:

    from qat import QAT 
    qat = QAT(qatconfig="path_to_file.yaml")

    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    results = qat.run(qasm_str, pipeline="echo6-alt")