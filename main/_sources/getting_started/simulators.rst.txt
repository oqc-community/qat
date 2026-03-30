Using QAT with simulators 
-------------------------------------

There are a number of routes for testing and executing quantum programs within QAT without using actual hardware.

.. contents::

Zero Engine
********************************

The simplest of such is the :class:`ZeroEngine <qat.engines.zero.ZeroEngine>`, which simply returns all readout results as zeroes. This is primarily useful for testing purposes.

Echo Engine 
********************************

The next level --- only compatible with programs compiled with the :class:`WaveformBackend <qat.backend.waveform.codegen.WaveformBackend>` --- is the :class:`EchoEngine <qat.engines.waveform.echo.EchoEngine>`. This engine "echos" back the readout pulses that would have been sent to the hardware. This is again primarily useful for testing purposes, especially when developing compilation pipelines. It's particularly useful for testing timing of pulse instructions.

Real-time Chip Simulator 
********************************

This is an engine that aims to simulate the physics behind Transmon physics, using the properties of the hardware model. It is currently only accessible through the PuRR module. However, we have provided pipeline wrappers that expose this simulator through the pipelines API. See :class:`LegacyRTCSPipeline <qat.pipelines.legacy.rtcs.LegacyRTCSPipeline>` for the pipeline, and :class:`RTCSModelLoader <qat.model.loaders.purr.rtcs.RTCSModelLoader>` for loading in hardware models compatible with this simulator.

See :doc:`../tutorials/rtcs` for an example of using the Real-time Chip Simulator with QAT.

.. note::

    The Real-time Chip Simulator is computationally intensive, and may take a while to run depending on the program and hardware model used. It also runs using the legacy PuRR module, and there are currently no plans to implement it with the new ways of working with QAT. This may change in the future.

Qiskit Aer Simulator
********************************    

To simulate quantum circuits, we make use of Qiskit's Aer Simulator through the legacy PuRR module. This is particularly useful to exactly simulate quantum circuits (at the circuit model level) to understand what exact results to expect. There is a pipeline wrapper that allows us to use this simulator through the pipelines API. See :class:`LegacyQiskitPipeline <qat.pipelines.legacy.qiskit.LegacyQiskitPipeline>` for the pipeline, and :class:`QiskitModelLoader <qat.model.loaders.purr.qiskit.QiskitModelLoader>` for loading in hardware models compatible with this simulator.

See :doc:`../tutorials/qiskit` for an example of using the Qiskit Aer Simulator with QAT.

.. note:: 

    This runs using the legacy PuRR module We will likely refactor this to make full use of the new pipelines API in the future.