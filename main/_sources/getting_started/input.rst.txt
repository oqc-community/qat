Supported input to the compiler
---------------------

Quantum programs can be provided as input to the compiler through various formats, and compiler settings can be configured through the :class:`CompilerConfig`.

.. contents::

QASM2
==========================

Simple quantum programs represented as a circuit model can be written in QASM2. These files can be provided as a string to the compiler. By default, the compiler supports the standard gate libraries `qelib1.inc` and `stdgates.inc`. Details on using QASM2 can be found `here <https://arxiv.org/pdf/1707.03429>`_. Qubit routing and circuit-level optimization can be configured for QASM2 programs using the :class:`CompilerConfig` (see below).


QASM3 and OpenPulse
==========================

Similarly, high-level circuit model programs can be provided in QASM3 format. QASM3 also supports pulse-level control through the OpenPulse specification. QASM3 programs can be provided as a string to the compiler. Details on using QASM3 can be found `here <https://arxiv.org/pdf/2104.14722>`_ and through its `specification <https://openqasm.com/>`_. 

.. warning::

    Circuit-level optimisations and qubit routing are not currently supported for QASM3 programs. Also, not all of the QASM3 and OpenPulse specification is implemented in QAT yet; this depends on the hardware being targeted. This includes classical control flow, which currently has no support in QAT.

Quantum Intermediate Representation (QIR)
==========================

QAT also accepts Quantum Intermediate Representation (QIR) programs as input to the compiler. QIR is an intermediate representation based on LLVM IR, designed to be a common interface between high-level quantum programming languages and low-level quantum hardware. QIR programs can be provided as bytecode to the compiler, or in `.ll` format. Details on using QIR can be found `here <https://quantum.microsoft.com/en-us/insights/blogs/qir/introducing-quantum-intermediate-representation-qir>`_. 

QAT treats qubits specified in QIR as logical qubits, and can handle qubit placement. Routing and circuit-level optimisations are supported through the :class:`CompilerConfig` (see below).

.. warning::

    Not all classical instructions in QIR are currently supported in QAT; this includes classical control flow.

Instruction Builder 
==========================

QAT has its own :class:`QuantumInstructionBuilder <qat.ir.instruction_builder.QuantumInstructionBuilder>` that can be used to assemble programs directly. It is instantiated with a hardware model, :code:`builder = QuantumInstructionBuilder(hardware_model)`, and has methods to assemble quantum programs directly in a pulse-level representation. Please refer to the API reference for more details.


Circuit-level (qubit) operations
*******************

The instruction builder allows us to do a number of gate-level operations, such as fetching qubits, applying unitary gates, and measurements. For example, to create a simple Bell state circuit:

.. code-block:: python 
  :linenos:

  from qat.model.loaders.lucy import LucyModelLoader
  from qat.ir.instruction_builder import QuantumInstructionBuilder

  model = LucyModelLoader(qubit_count=8).load()
  builder = QuantumInstructionBuilder(model)

  # Fetch the qubits by logical index
  qubit_0 = builder.get_logical_qubit(0)
  qubit_1 = builder.get_logical_qubit(1)

  # Apply gates and measurements for a bell state
  builder.had(qubit_0)
  builder.cnot(qubit_0, qubit_1)
  builder.measure(qubit_0)
  builder.measure(qubit_1)

Notice that we requested a logical qubit from the builder. The qubits are ordered by index in the hardware model, but there may be missing qubit indicies. This might be because the indexing starts at one, or because a qubit is not available. To fetch qubits by their physical indexing, we can use :code:`get_physical_qubit(index)` instead.

Pulse-level operations 
**********************

The hardware model contains a description of pulse channels for each qubit. When the builder is instantiated, it will create an IR representation for each pulse channel described in the hardware model. These pulse channels can be fetched using :code:`get_pulse_channel(channel_name)`. But also, we can create a new pulse channel with the desired properties using :code:`create_pulse_channel(frequency, physical_channel_id, ...)` where the `physical_channel_id` specifies which physical channel the pulse channel will be mapped to at compilation (and thus the qubit it will target). See below for fetching / creating pulse channels, and applying pulse-level operations such as phase shifts and pulses.

.. code-block:: python 
  :linenos:

  from qat.model.loaders.lucy import LucyModelLoader
  from qat.ir.instruction_builder import QuantumInstructionBuilder
  from qat.ir.waveforms import SquareWaveform

  model = LucyModelLoader(qubit_count=8).load()
  builder = QuantumInstructionBuilder(model)

  # Fetch & create the pulse channels
  drive_channel_id = model.qubit_with_index(0).drive_pulse_channel.uuid
  physical_channel_id = model.qubit_with_index(1).physical_channel.uuid 
  drive_channel = builder.get_pulse_channel(drive_channel_id)
  custom_channel = builder.create_pulse_channel(
      frequency=5.6e9,
      physical_channel=physical_channel_id,
      scale=0.05,
      uuid="custom_channel",
  )

  # Apply a phase shift and a pulse
  builder.phase_shift(drive_channel, 0.254)
  waveform = SquareWaveform(width=80e-9, amp=0.1)
  builder.pulse(target=custom_channel.uuid, waveform=waveform)

.. warning::

    High-level iteration support, such as sweeps and device assigns aren't currently supported in the version of the instruction builder.


PuRR builders 
***********************

A similar builder exists in the legacy PuRR module, :class:`QuantumInstructionBuilder <qat.purr.compiler.builders.QuantumInstructionBuilder>`. This builder has similar functionality to the new instruction builder, but uses the legacy PuRR IR representation.

Compiler Config 
========================

There are many settings that can impact the way QAT will compile the source code into
native code. Some of these settings cannot always be specified in the source code, such as
the number of shots, or the format that results are returned in. Others are settings that
are used during optimization that might result in a different outcome, such as compiler
optimization settings. The :class:`CompilerConfig` allows these settings to be specified and
tailored to the job.

Shots 
*******************

The compiler config can be used to specify the number of shots for a program (the number of 
times a program should be executed on the hardware) through the attribute :code:`repeats`.
Furthermore, if the qubits in the QPU reset to their ground state passively in between each 
shot, the compiler config can be used to specify the passive reset time used at the end of 
shot through the :code:`passive_reset_time`. Alternatively, you can use the :code:`repetition_time`
which pad the shot with a passive reset up to the :code:`repetition_time`, although this is
considered legacy and will not be supported in the future.

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