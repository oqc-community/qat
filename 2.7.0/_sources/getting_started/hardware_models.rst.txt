Hardware models 
------------------------

Hardware models are a high-level description of quantum processing units (QPUs). They
contain all the information about a QPU that is needed at compile time and runtime,
including:

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

    from qat.model.loaders.converted import EchoModelLoader

    connectivity = [(0, 1), (1, 2), (2, 3), (3, 0)]
    model = EchoModelLoader(qubit_count=4, connectivity=connectivity).load()

The connectivity here is used to express which qubits are coupled. While we write it down
here explicitly for demonstration purposes, this is the default connectivity for the 
:class:`EchoModelLoader <qat.model.loaders.converted.EchoModelLoader>`.

Legacy models 
^^^^^^^^^^^^^^^^^^^^^^^

Models from legacy QAT can easily be imported, either as a
:class:`QuantumHardwareModel <qat.purr.compiler.hardware_models.QuantumHardwareModel>` (the 
legacy version of hardware models), or as a Pydantic
:class:`PhysicalHardwareModel <qat.model.hardware_model.PhysicalHardwareModel>`. 
The import type will depend on its use case. The later was achieved in the previous example,
where the :class:`EchoModelLoader <qat.model.loaders.converted.EchoModelLoader>` creates a
legacy Echo model and converts it to a Pydantic model.

To load the models as a
:class:`QuantumHardwareModel <qat.purr.compiler.hardware_models.QuantumHardwareModel>`,
:mod:`qat.model.loaders` has a :mod:`legacy <qat.model.loaders.legacy>` package, which
includes the following loaders:

* :class:`EchoModelLoader <qat.model.loaders.legacy.EchoModelLoader>`: a model
  traditionally used with "echo engines".
* :class:`LucyModelLoader <qat.model.loaders.legacy.LucyModelLoader>`: a mock-up hardware 
  model of the legacy OQC Lucy hardware.
* :class:`QiskitModelLoader <qat.model.loaders.legacy.QiskitModelLoader>`: a model 
  with specific support for Qiskit's AerSimulator as a target.
* :class:`RTCSModelLoader <qat.model.loaders.legacy.RTCSModelLoader>`: a model 
  with specific support for OQC's real time chip simulator (RTCS).
* :class:`FileModelLoader <qat.model.loaders.legacy.FileModelLoader>`: used to load 
  legacy hardware models from file.

For legacy models that do not have a designated loader method, we can manually convert from 
a legacy to Pydantic hardware model by using the
:meth:`convert_legacy_echo_hw_to_pydantic <qat.model.convert_legacy.convert_legacy_echo_hw_to_pydantic>`
method.





Loading models from external files 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hardware models can be serialized and saved as a JSON file, allowing them to be shared and 
stored elsewhere after calibration routines are complete. QAT has some helper methods to 
build hardware models from calibration files, Let's start by generating a calibration file 
that we can use as an example.

.. code-block:: python 
    :linenos:

    from qat.model.loaders.converted import EchoModelLoader

    connectivity = [(0, 1), (1, 2), (2, 3), (3, 0)]
    model = EchoModelLoader(qubit_count=4, connectivity=connectivity).load()
    blob = model.model_dump_json()
    with open("temp.json", "w") as f:
        f.write(blob)
    
This file can be loaded using the
:class:`FileModelLoader <qat.model.loaders.file.FileModelLoader>`.

.. code-block:: python 
    :linenos:

    from qat.model.loaders.file import FileModelLoader

    model = FileModelLoader("temp.json").load()


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
