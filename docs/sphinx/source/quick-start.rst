Getting Started
===============================

At its most basic level QAT requires three things: a valid hardware model, some IR as input such as QASM or QIR, and a configuration object. It also exposes API's at the top level that consume these objects.

So you can use it simply with:

.. code-block:: Python

    hardware_model = get_default_echo_hardware()
    qasm_file = ...
    config = CompilerConfig()
    results = execute(qasm_file, hardware_model, config)

This will run a QASM file of your choice against our debug echo backend. The echo does nothing except echo nonsense back at you, but it's still useful for testing everything up until that point.

We have hardware models that are configured already for hardware OQC have available, most of them also have a 'get_default' method which requires no further configuration.

The compiler config is used for all high-level process settings: optimizers, shot count, result format, things of that ilk.

If you don't need to extend any functionality you should be able to do everything you want by using the options available in the compiler config and using a pre-build hardware model from one of our backends.

If you *do* want to override various pieces of functionality start by reading about QATs architecture and work from there.

In both instances our test suite can help give you a guide on how to do things.