.. _compiler_config:

Compiler Config
---------------------

There are many settings that can impact the way QAT will compile the source code into
native code. Some of these settings cannot always be specified in the source code, such as
the number of shots, or the format that results are returned in. Others are settings that
are used during optimization that might result in a different outcome, such as compiler
optimization settings. The :class:`CompilerConfig` allows these settings to be specified and
tailored to the job.

.. contents::

Shots 
*******************

The compiler config can be used to specify the number of shots for a program (the number of 
times a program should be executed on the hardware) through the attribute :code:`repeats`.
Furthermore, if the qubits in the QPU reset to their ground state passively in between each 
shot, the compiler config can be used to specify the time each shot will take: a padding is
added to the end of each shot to allow the state of the qubits to reset, and the total time
of the shot (including the padding) is :code:`repetition_time`. 

.. code-block:: python 

    from compiler_config.config import CompilerConfig
    config = CompilerConfig(repeats=1000, repetition_time=1e-3)


Results formatting 
*******************

The engines for target hardware or simulators will return results in a format that depends 
on the :class:`AcquireMode <qat.purr.compiler.instructions.AcquireMode>`, which will not 
always match the user-expectation. The compiler config can be used to specify the results
format using the :class:`QuantumResultsFormat`. There are a number of options available:

* :code:`QuantumResultsFormat.raw()`: Returns results in the format returned by the engine:
  no post-processing is done.
* :code:`QuantumResultsFormat.binary()`: The most common result for each measurement is 
  returned. For example, for two qubit measurements the results might look like
  :code:`{'c': [0, 0]}`.
* :code:`QuantumResultsFormat.binary_count()`: Results are returned as a dictionary of 
  bit strings, with the values containing the number of times each bit string was measured.
  For example, for two qubit measurements, :code:`{'c': {'00': 502, '11': 498}}`.
* :code:`QuantumResultsFormat.squash_binary_result_arrays()`: Like :code:`.binary()`, but 
  the result is formatted as a bit string. For example, :code:`{'c': '00'}`.

Within the compiler config, the :class:`QuantumResultsFormat` can be used as follows:

.. code-block:: python 

    from compiler_config.config import CompilerConfig
    config = CompilerConfig(results_format=QuantumResultsFormat().binary_count())


Optimizations 
*****************

Sometimes a quantum program that is given is not always immediately compatible or optimal 
for the target hardware. QAT has a number of optimization methods to overcome this issue,
which can be specified through the :class:`OptimizationConfig`.

A large amount of the optimization methods for quantum circuits used in QAT is done via
`pytket <https://docs.quantinuum.com/tket/api-docs/>`_. The different optimizations can be 
chosen using the :class:`TketOptimizations` flag. There are many passes available, here we 
will just specify some pre-defined composite options:

* :code:`TketOptimizations.One`: Maps the quantum circuit onto the target hardware's
  topology and resolves the direction of 2Q gates to match the hardware.
* :code:`TketOptimizations.Two`: :code:`TketOptimizations.One` and manipulates the circuit 
  to reduce the gate count without changing the meaning of the circuit.

These options can be used in the Tket config, which comes with some options:

* :code:`Tket().disable()`: Disables Tket optimizations.
* :code:`Tket().minimum()`: Maps the quantum circuit onto the target hardware's topology.
* :code:`Tket().default()`: Uses :code:`TketOptimizations.One`.
* Custom optimization flags, e.g.,  :code:`Tket(TketOptimizations.Two)`.

Within the compiler config, the :class:`Tket` config can be used as follows:

.. code-block:: python 

    from compiler_config.config import CompilerConfig, Tket
    config = CompilerConfig(optimizations=Tket().default())

Error mitigation
*****************

If available in the hardware model, the compiler config can be used to specify readout 
error mitigation strategies to be used. The available options are 

* :code:`ErrorMitigationConfig.LinearMitigation`: Applies error mitigation to each qubit 
  individually using the readout errors in the hardware model.
* :code:`ErrorMitigationConfig.MatrixMitigation`: Applies error mitigation to qubits
  collectively, considering the collective state of all qubits.

.. code-block:: python 

    from compiler_config.config import ErrorMitigationConfig
    config = CompilerConfig(error_mitigiation=ErrorMitigationConfig.LinearMitigation)

.. warning::

    Error mitigation is an experimental feature and might not always work as expected.