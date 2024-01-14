![QAT Logo](qat-logo.png)

**QAT** (Quantum Assembly Toolkit/Toolchain) is a low-level quantum compiler and runtime which facilitates executing quantum IRs
such as [QASM](https://openqasm.com/), [OpenPulse](https://openqasm.com/language/openpulse.html) and
[QIR](https://devblogs.microsoft.com/qsharp/introducing-quantum-intermediate-representation-qir/) against QPU drivers.
It facilitates the execution of largely-optimised code, converted into abstract pulse-level and hardware-level instructions,
which are then transformed and delivered to an appropriate driver.

For our full documentation check [here](https://oqc-community.github.io/qat/main/index.html).

### Installation

QAT can be installed from [PyPI](https://pypi.org/project/qat-compiler/>) via:
`pip install qat-compiler`

### Building from Source

We use [poetry](https://python-poetry.org/) for dependency management and run on
[Python 3.8+](https://www.python.org/downloads/).
Once both of these are installed run this in the root folder to install all the dependencies that you need: `poetry install`

### Roadmap

Here's a list of what we're currently working on, if you want to get involved contact the person or group linked.

In-development:

- Classical-quantum hybrid computation. [John Dumbell](jdumbell@oxfordquantumcircuits.com>)
- Runtime-embedded QEC. [Jamie Friel](jfriel@oxfordquantumcircuits.com>)

Designing / verifying suitability:

- Distributed QPU execution. [John Dumbell](jdumbell@oxfordquantumcircuits.com>)

To-do:

- Full QASM v3 support. Currently waiting for available parsers.

### Contributing

To take the first steps towards contributing to QAT, visit our
[contribution](https://github.com/oqc-community/qat/blob/main/contributing.md) documents, which provides details about our
process.

We also encourage new contributors to familiarise themselves with the
[code of conduct](https://github.com/oqc-community/qat/blob/main/code_of_conduct.md) and to adhere to these
expectations.

### Assistance

For support, please reach out in the [discussions](https://github.com/oqc-community/qat/discussions) tab of this repository or file an [issue](https://github.com/oqc-community/qat/issues).

### Licence

This code in this repository is licensed under the BSD 3-Clause Licence.
Please see [LICENSE](https://github.com/oqc-community/qat/blob/main/LICENSE) for more information.

### Feedback

Please let us know your feedback and any suggestions by reaching out in [Discussions](https://github.com/oqc-community/qat/discussions>).
Additionally, to report any concerns or
[code of conduct](https://github.com/oqc-community/qat/blob/main/code_of_conduct.md) violations please use this
[form](https://docs.google.com/forms/d/e/1FAIpQLSeyEX_txP3JDF3RQrI3R7ilPHV9JcZIyHPwLLlF6Pz7iGnocw/viewform?usp=sf_link).
