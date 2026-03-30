Executing your first program
------------------------

.. include:: ../notebooks/getting_started.ipynb
   :parser: myst_nb.docutils_

Setting up using QAT config 
=============================

QAT also provides a way to conveniently set up compilation and execution pipelines as a configurational YAML file. If we write a file called :code:`qatconfig.yaml` with the following contents:

.. code-block:: yaml
    :linenos:

    HARDWARE:
    - name: lucy
      type: qat.model.loaders.lucy.LucyModelLoader 
      config:
        qubit_count: 8

    PIPELINES:
    - name: echo_pipeline
      pipeline: qat.pipelines.waveform.EchoPipeline 
      hardware_loader: lucy

We can then load it in to QAT and execute as before:

.. code-block:: python
    :linenos:

    from qat import QAT 
    from compiler_config.config import CompilerConfig, QuantumResultsFormat

    # Set up compile and execute pipelines
    core = QAT(qatconfig="qatconfig.yaml")

    # Set up the input program 
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    config = CompilerConfig(repeats=1000, results_format=QuantumResultsFormat().binary_count())

    # Execute using QAT 
    results, metrics = core.run(qasm_str, compiler_config=config, pipeline="echo_pipeline")

