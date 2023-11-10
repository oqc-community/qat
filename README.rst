.. image:: https://github.com/oqc-community/qat/blob/main/qat-logo.png
  :width: 400
  :alt: QAT

.. readme_text_start_label

|

**QAT** (Quantum Assembly Toolkit/Toolchain) is a low-level quantum compiler and runtime which facilitates executing quantum IRs
such as `QASM <https://openqasm.com/>`_, `OpenPulse <https://openqasm.com/language/openpulse.html>`_ and
`QIR <https://devblogs.microsoft.com/qsharp/introducing-quantum-intermediate-representation-qir/>`_ against QPU drivers.
It facilitates the execution of largely-optimised code, converted into abstract pulse-level and hardware-level instructions,
which are then transformed and delivered to an appropriate driver.

For the official QAT documentation, please see `QAT <https://oqc-community.github.io/qat>`_.

|

----------------------
Installation
----------------------

QAT can be installed from `PyPI <https://pypi.org/project/qat-compiler/>`_ via:
:code:`pip install qat-compiler`

|

----------------------
Building from Source
----------------------

We use `poetry <https://python-poetry.org/>`_ for dependency management and run on
`Python 3.8+ <https://www.python.org/downloads/>`_.
Once both of these are installed run this in the root folder to install all the dependencies that you need:
:code:`poetry install`

.. note::
   If you are contributing to the project we recommend that you also run

   :code:`poetry run pre-commit install`
   to enable pre-commit checks.

|

----------------------
Roadmap
----------------------

Here's a list of what we're currently working on, if you want to get involved contact the person or group linked.

In-development:

- Classical-quantum hybrid computation. `John Dumbell <jdumbell@oxfordquantumcircuits.com>`_
- Runtime-embedded QEC. `Jamie Friel <jfriel@oxfordquantumcircuits.com>`_

Designing / verifying suitability:

- Distributed QPU execution. `John Dumbell <jdumbell@oxfordquantumcircuits.com>`_

To-do:

- Full QASM v3 support. Currently waiting for available parsers.

|

----------------------
Contributing
----------------------

To take the first steps towards contributing to QAT, visit our
`contribution <https://github.com/oqc-community/qat/blob/main/CONTRIBUTING.rst>`_ documents, which provides details about our
process.
We also encourage new contributors to familiarise themselves with the
`code of conduct <https://github.com/oqc-community/qat/blob/main/CODE_OF_CONDUCT.rst>`_ and to adhere to these
expectations.

|

----------------------
Where to get help
----------------------

For support, please reach out in the `Discussions <https://github.com/oqc-community/qat/discussions>`_ tab of this repository or file an `issue <https://github.com/oqc-community/qat/issues>`_.

|

----------------------
Licence
----------------------

This code in this repository is licensed under the BSD 3-Clause Licence.
Please see `LICENSE <https://github.com/oqc-community/qat/blob/main/LICENSE>`_ for more information.

|

----------------------
Feedback
----------------------

Please let us know your feedback and any suggestions by reaching out in `Discussions <https://github.com/oqc-community/qat/discussions>`_.
Additionally, to report any concerns or
`code of conduct <https://github.com/oqc-community/qat/blob/main/CODE_OF_CONDUCT.rst>`_ violations please use this
`form <https://docs.google.com/forms/d/e/1FAIpQLSeyEX_txP3JDF3RQrI3R7ilPHV9JcZIyHPwLLlF6Pz7iGnocw/viewform?usp=sf_link>`_.

|

----------------------
FAQ
----------------------
    Why is this in Python?

Mixture of reasons. Primary one is that v1.0 was an early prototype and since the majority of the quantum community
know Python it was the fastest way to build a system which the most people could contribute to building. The API's would
always stick around anyway, but as time goes on the majority of its internals has been, is being, or will be moved to Rust/C++.

    Where do I get started?

Our tests are a good place to start as they will show you the various ways to run QAT. Running and then stepping
through how it functions is the best way to learn.

We have what's known as an echo model and engine which is used to test QATs functionality when not attached to a QPU.
You'll see these used almost exclusively in the tests, but you can also use this model to see how QAT functions on
larger and more novel architectures.

High-level architectural documents are incoming and will help explain its various concepts at a glance, but
right now aren't complete.

    What OS's does QAT run on?

Windows and Linux are its primary development environments. Most of its code is OS-agnostic but we can't
guarantee it won't have bugs on untried ones. Dependencies are usually where you'll have problems, not the core
QAT code itself.

If you need to make changes to get your OS running feel free to PR them to get them included.

    I don't see anything related to OQC's hardware here!

Certain parts of how we run our QPU have to stay propriety and for our initial release we did not have time to
properly unpick this from things we can happily release. We want to release as much as possible and as you're
reading this are likely busy doing just that.

    Do you have your own simulator?

We have a real-time chip simulator that is used to help test potential changes and their ramifications to hardware.
It focuses on accuracy and testing small-scale changes so should not be considered a general simulator. 3/4 qubit
simulations is its maximum without runtime being prohibitive.
