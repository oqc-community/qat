Pipelines
---------------------

The primary way to access QAT's compilation and execution capabilities is through its
pipelines API. Pipelines allow us to define the compilation and execution process in a 
configurable and modular way.

.. contents::

Using QAT pipelines
***********************

QAT has a number of pipelines that are pre-defined and ready to use. We start by creating
a :class:`QAT <qat.core.qat.QAT>` object.

.. code-block:: python

    from qat import QAT 
    core = QAT()

We can now add a pipeline to it. Let's add a pipeline that uses the 
:class:`EchoEngine <qat.engines.waveform_v1.echo.EchoEngine>`:

.. code-block:: python 

    from qat.pipelines.waveform import echo8
    core.pipelines.add(echo8, default=True)

This will add the "echo8" pipeline to :code:`core`, which can be used to compile and execute
programs using a simulator that simply returns all readouts as zeroes. The "8" in "echo8"
specifies that the simulator has 8 qubits available. The pipeline already has its 
compilation modules chosen so that it is compatible with the echo engine: we'll cover this 
in more detail in :ref:`compilation`. Now we can use this to execute a QASM program.

.. code-block:: python 
    :linenos:

    from qat import QAT 
    from qat.pipelines.waveform import echo8
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


Note that :code:`metrics` is returned. This is an object that contains
various metrics regarding compilation, such as circuits after they have been optimized, or
the total number of pulse-level instructions. Since the echo pipeline just returns zeros
for readouts, the result returned here is :code:`results = {'c': {'00': 1000}}`.

We could also achieve the same workflow by compiling and executing separately:

.. code-block:: python 
    
    pkg, metrics = core.compile(qasm_str, config, pipeline="echo8")
    results, metrics = core.execute(pkg, config, pipeline="echo8")

The package :code:`pkg` contains the native instructions to be sent to the target (in this
case, the echo simulator).


Updateable Pipelines
*********************************************

The pipeline from the previous example was a default pipeline instance that you can import
from QAT. But more likely, you would want to configure your own pipeline using your own 
hardware model and target data for a given type of hardware.
Updateable pipelines provide a prescribed way to create pipelines that offer utility to 
rebuild pipelines using new hardware models or target data. Let's demonstrate with the
Waveform example and the echo engine from the previous section, but by configuring our own.

.. code-block:: python 
  :linenos: 

  from qat.pipelines.waveform import EchoPipeline, PipelineConfig
  from qat.model.loaders.lucy import LucyModelLoader

  loader = LucyModelLoader(qubit_count=8)
  config = PipelineConfig(name="echo_pipeline")
  pipeline = EchoPipeline(loader=loader, config=config)
  pipeline.update(reload_model=True)

There's a few things to notice here. First, the updateable pipeline actually takes
ownership of the pipeline instance it creates, and in the way, can be used in-place of the 
pipeline instance. Secondly, we instantiated it using the model loader, allowing us to 
reload the model directly from the loader. However, updateable pipelines can be configured 
using a hardware model directly, and similarly, they can be updated by providing a new model
directly. The target data can also be provided at instantiation, and updated to using 
:code:`pipeline.update(target_data=target_data)`. Finally, each updateable pipeline is
paired with a :class:`PipelineConfig <qat.pipelines.updateable.PipelineConfig>` object that
stores configuration data for the pipeline, such as its name, and additional compiler 
settings.






Compile and Execute (updateable) pipelines 
*********************************************

The example seen previously uses a "full pipeline" that is capable of both compiling and 
executing a program. However, we can also express pipelines that can only compile
:class:`CompilePipeline <qat.pipelines.pipeline.CompilePipeline>` or only execute
:class:`ExecutePipeline <qat.pipelines.pipeline.ExecutePipeline>`. The benefits of this are that

* We can clearly separate out the compilation and execution steps over distributed systems.
* We can mix-and-match compilation and execution pipelines. For example, we could compile
  a program for a specific hardware target, but execute it on a simulator. On the contrary,
  we could also define multiple compile pipelines that expose different compiler features,
  but execute them all on the same hardware target.

We can compile and execute against particular pipelines by using :code:`QAT.compile` and
:code:`QAT.execute`, specifying the pipeline to use.

.. code-block:: 
  :linenos:

  from qat.pipelines.waveform import WaveformCompilePipeline, EchoExecutePipeline, PipelineConfig
  from qat.model.loaders.lucy import LucyModelLoader
  from qat import QAT

  # Define pipelines 
  model = LucyModelLoader(qubit_count=16).load()
  compile_pipeline = WaveformCompilePipeline(config=PipelineConfig(name="compile"), model=model)
  execute_pipeline = EchoExecutePipeline(config=PipelineConfig(name="execute"), model=model)  

  # Register pipelines
  core = QAT()
  core.pipelines.add(compile_pipeline, default=True)
  core.pipelines.add(execute_pipeline, default=True)

  # Execute against pipelines
  qasm_str = """
  OPENQASM 2.0;
  include "qelib1.inc";
  qreg q[2];
  creg c[2];
  h q[0];
  cx q[0], q[1];
  measure q -> c;
  """
  executable, compile_metrics = core.compile(qasm_str, pipeline="compile")
  results, execute_metrics = core.execute(executable, pipeline="execute")


Default pipelines that are available in QAT 
*********************************************

There are a number of pipelines in QAT that are available to use off-the-shelf.

* :mod:`qat.pipelines.waveform`: Pipelines that execute using the
  :class:`EchoEngine <qat.engines.waveform.echo.EchoEngine>`. The pipelines available
  by default are :attr:`echo8`, :attr:`echo16`, :attr:`echo32`. The updateable pipelines are
  available through :class:`EchoPipeline <qat.pipelines.waveform.EchoPipeline>`, 
  :class:`WaveformCompilePipeline <qat.pipelines.waveform.WaveformCompilePipeline>` and
  :class:`EchoExecutePipeline <qat.pipelines.waveform.EchoExecutePipeline>`.

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
Similarly, the execution part of the pipeline is defined by two objects: the **engine** and
the **runtime**. The engine acts as an adapter to the target, and deals with communicating
the instructions and results from the runtime to the target. The runtime handles the
engine, and deals with software post-processing of the results. See :ref:`execution` for 
more details.

See :doc:`../notebooks/tutorials/custom_pipeline` for a working example of defining a custom
pipeline and updateable pipeline.