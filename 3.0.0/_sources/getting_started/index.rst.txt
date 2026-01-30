Getting started with QAT
==========================

.. image:: images/pipeline.png
    :align: center


The Quantum Assembly Toolkit (QAT) is OQC's open source package that primarily aids with two
aspects:

#. **Compile**: Compiling quantum source programs into native code that can be executed on
   actual quantum hardware.
#. **Execute**: Runtime of quantum programs to assist with execution, interpretation and
   results post-processing.

It also contains utility for simulation (typically used for testing purposes), such as 
an adapter to Qiskit's AerSimulator and our realistic (but limited) real time chip simulator.



.. toctree::
    :maxdepth: 4
   
    hardware_models
    compiler_config
    pipelines
    compilation 
    execution 