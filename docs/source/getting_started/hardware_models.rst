Hardware models & target data
------------------------

The description of the quantum processing unit (QPU) and the control hardware used to manipulate is configured through the *hardware model* and *target data*.

Hardware Models 
*****************************

Hardware models are a high-level description of quantum processing units (QPUs). They contain information about a QPU and the control hardware that is required for compilation, and can change on a regular basis (e.g. during a daily calibration).

* **Topology**: How qubits on the QPU are coupled, and the quality of those couplings.
* **Qubit properties**: Properties of the qubits that are required for pulse generation,
  such as the driving frequency.
* **Resonators**: Properties of the resonators that are required for readout. 
* **Calibrated pulses**: The pulses used to manipulate the state of qubits need to be chosen
  precisely to have the desired outcome. Calibration routines are run frequently to optimize 
  these values: the hardware model contains these calibration results, such as pulse shape 
  and duration.
* **Error mitigation**: Contains the readout benchmarks which can be used in error
  mitigation strategies.

See :class:`LogicalHardwareModel <qat.model.hardware_model.LogicalHardwareModel>` and 
:class:`PhysicalHardwareModel <qat.model.hardware_model.PhysicalHardwareModel>` for more 
details on the properties contained in hardware models.


Hardware model loaders 
***********************

QAT has a number of lightweight "loaders" for importing hardware models from external 
calibration files and loading in some default models used for simulators and testing. 

Echo models 
^^^^^^^^^^^^^^^^^^^^^^^

Let's get started with loading in a simple model of four qubits on a ring with
nearest-neighbour connectivity.

.. code-block:: python 

    from qat.model.loaders.lucy import LucyModelLoader

    model = LucyModelLoader(qubit_count=4).load()

This is just a mock-up of OQC's Lucy hardware, and doesn't contain any practical calibration data.

File model loaders 
^^^^^^^^^^^^^^^^^^^^^^^

The :class:`FileModelLoader <qat.model.loaders.file.FileModelLoader>` can be used to load
hardware models from their respective "calibration" JSON files.

.. code-block:: python 
  
    from qat.model.loaders.file import FileModelLoader

    model = FileModelLoader("calibration.json").load()

There is an equivalent legacy (PuRR) :class:`FileModelLoader <qat.model.loaders.purr.file.FileModelLoader>` loader that loads in a :class:`QuantumHardwareModel <qat.purr.compiler.hardware_models.QuantumHardwareModel>`, which can be found at :class:`FileModelLoader <qat.model.loaders.purr.FileModelLoader>`.


Legacy models 
^^^^^^^^^^^^^^^^^^^^^^^

For PuRR models that do not have a designated loader method, we can manually convert from 
a legacy to Pydantic hardware model by using the
:meth:`convert_purr_echo_hw_to_pydantic <qat.model.convert_purr.convert_purr_echo_hw_to_pydantic>`
method.


Building hardware models
*************************

The :class:`PhysicalHardwareModelBuilder <qat.model.builder.PhysicalHardwareModelBuilder>`
can be used for creating hardware models with a custom topology. The couplings of a QPU 
are known as the "physical connectivity": in theory, these might be bidirectional. 
The "logical connectivity" refers to the connectivity that will be used in compilation.
This is useful for a few reasons. Firstly, while couplings are bidirectional in principle,
in practice we find that a particular QPU might have a preference for just one direction
(that is, we are able to calibrate it better for the given direction). Secondly, some QPUs
might have faulty couplings that are known not to perform well. When this is the case, we
might want to disable the coupling without throwing away knowledge of the existence of the
coupling.

As a simple example, let us build a model that has qubits on a ring with nearest-neighbour 
couplings, but with random coupling directions.

.. code-block:: python 
    :linenos:

    from qat.model.builder import PhysicalHardwareModelBuilder

    physical_connectivity = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {2, 0}}
    logical_connectivity = {0: {1}, 2: {1, 3}, 3: {0}}
    builder = PhysicalHardwareModelBuilder(physical_connectivity, logical_connectivity)
    model = builder.model

.. note:: 

    Once built, the topology of hardware models (including both the physical and logical 
    topology) is frozen and cannot be changed directly. However, the calibratable
    properties, such as pulses can be modified, are validated to ensure they are sensible.


Target Data 
****************

Unlike the hardware model, the target data contains configurational information about the hardware that is static or information that changes infrequently. This includes:

* Information surrounding the number of shots that can be achieved in a single execution.
* The passive reset time used to return qubits to their ground state between executions.
* Constants of the control hardware, such as the sampling rates and clock speeds.
* Limits on allowed quantities, such as frequencies and pulse durations.

The standard target data can be found at :class:`TargetData <qat.model.target_data.TargetData>`. that can be customized through :class:`CustomTargetData <qat.model.target_data.CustomTargetData>`.
Each type of control hardware has its own target data, which might contain information unique to that set up. An instance of each target data is expected to exist for each QPU.