QAT Config 
=========================

The QAT Config is a configuration YAML file that can be used to configure compilation and execution through QAT. It allows us to specify the source of a hardware model, engines that execute the compiled code against hardware or simulators, and complete compilation and execution pipelines.

QAT Configs are imported with the :class:`QAT <qat.core.qat.QAT>` object, by providing the path to the configuration file as the argument:

.. code-block:: python

    from qat import QAT

    core = QAT("path_to_config.yaml")

.. contents::

Global QAT settings
----------------------
There are a few settings that can be specified globally for QAT through the configuration file. These include:

* **MAX_REPEATS_LIMIT**: The maximum number of shots that can be specified for a job. This is to prevent accidental submission of jobs with an extremely high number of shots.
* **INSTRUCTION_VALIDATION**:
    * **NO_MID_CIRCUIT_MEASUREMENTS**: A boolean flag that indicates whether mid-circuit measurements are allowed in the input program. If set to true, any mid-circuit measurements will raise an error during compilation.
    * **MAX_INSTRUCTION_LENGTH**: A boolean flag that indicates whether to validate the length of instructions in the compiled program against the maximum allowed by the target data. If set to true, any instructions that exceed the maximum length will raise an error during compilation.
    * **PULSE_DURATION_LIMITS**: A boolean flag that indicates whether to validate the duration of pulses in the compiled program against the limits allowed by the target data. If set to true, any pulses that exceed the duration limits will raise an error during compilation.

These can optionally be set in the YAML file. For example,

.. code-block:: YAML

    MAX_REPEATS_LIMIT: 5000
    INSTRUCTION_VALIDATION:
        NO_MID_CIRCUIT_MEASUREMENTS: false
        MAX_INSTRUCTION_LENGTH: true
        PULSE_DURATION_LIMITS: true


Hardware
-----------------------------

The hardware loaders allow us to load in a hardware model. They can be specified in the QAT config under a **HARDWARE** section, like so:

.. code-block:: YAML 

    HARDWARE:
    - name: lucy8
      type: qat.model.loaders.lucy.LucyModelLoader
      config:
        qubit_count: 8
    - name: lucy16
      type: qat.model.loaders.lucy.LucyModelLoader
      config:
        qubit_count: 16 
    - name: file_model
      type: qat.model.loaders.file.FileModelLoader
      config:
        path: path_to_calibration.json

Notice each hardware is given a unique name, which can be referenced later when defining pipelines. They also contain a config with arguments that correspond to the :code:`__init__` method of each loader. 


Engines 
--------------------------------------

Engines are the components of execution pipelines that communicate with the hardware or a simulator. Since we might want to define multiple execution pipelines that target the same hardware, it is convenient to define a single engine. This is even more so if the hardware / simulator only allows a single engine to maintain a connection. We can define engines as such:

.. code-block:: YAML

    ENGINES:
    - name: zero
      type: qat.engines.zero.ZeroEngine
    - name: echo 
      type: qat.engines.waveform.echo.EchoEngine

Should the engines require it, the hardware can also be provided with a `hardware` attribute.


Pipelines 
--------------------------------------

Pipelines can be configured in the QAT config. They can be provided as explicit compile and execute pipelines using the `COMPILE` and `EXECUTE` sections, or as a combined pipeline using the `PIPELINES` section. 

.. warning:: 

    While full pipelines can be configured, it is preferred to define `COMPILE` and `EXECUTE` pipelines instead. Full pipelines might not be supported in future versions.

See the example below for defining pipelines 

.. code-block:: YAML

    COMPILE:
    - name: echo8-compile
      pipeline: qat.pipelines.waveform.WaveformCompilePipeline
      hardware_loader: lucy8
      target_data:
        type: qat.model.target_data.CustomTargetData
        config:
          passive_reset_time: 5e-4
    - name: echo16-compile
      pipeline: qat.pipelines.waveform.WaveformCompilePipeline
      hardware_loader: lucy16
      default: true

    EXECUTE:
    - name: echo8-execute
      pipeline: qat.pipelines.waveform.EchoExecutePipeline 
      hardware_loader: lucy8
    - name: echo16-execute
      pipeline: qat.pipelines.waveform.EchoExecutePipeline
      hardware_loader: lucy16
      default: true
    - name: zero8-execute
      runtime: qat.runtime.SimpleRuntime
      engine: zero
      hardware_loader: lucy8

    PIPELINES:
    - name: echo8
      pipeline: qat.pipelines.waveform.EchoPipeline
      hardware_loader: lucy8
    - name: echo32
      pipeline: qat.pipelines.echo.echo32
      default: false

Notice that in `echo8-compile` we also provide the target data. That doesn't yet have its own section in QAT config, but that will likely change in the future.  We also defined a `zero8-execute` pipeline that didn't use an :class:`UpdateablePipeline <qat.pipelines.updateable.UpdateablePipeline>`, but instead was provided its own runtime and used the zero engine. This demonstrates how pipelines can be defined in a more granular fashion. We also set a default compile and execute pipeline. Alternatively, we could have set a default full pipeline in the `PIPELINES` section.

A complete example
-------------------------------

Putting it all together, we can define a complete QAT config as such:

.. code-block:: YAML
    :linenos:

    MAX_REPEATS_LIMIT: 5000
    INSTRUCTION_VALIDATION:
        NO_MID_CIRCUIT_MEASUREMENTS: false

    HARDWARE:
    - name: lucy8
      type: qat.model.loaders.lucy.LucyModelLoader
      config:
        qubit_count: 8

    ENGINES:
    - name: echo 
      type: qat.engines.waveform.echo.EchoEngine

    COMPILE:
    - name: echo8-compile
      pipeline: qat.pipelines.waveform.WaveformCompilePipeline
      hardware_loader: lucy8
      target_data:
        type: qat.model.target_data.CustomTargetData
        config:
          passive_reset_time: 5e-4
      default: true

    EXECUTE:
    - name: echo8-execute
      pipeline: qat.pipelines.waveform.EchoExecutePipeline 
      hardware_loader: lucy8
      default: true