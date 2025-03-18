.. _compilation:

Compilation 
------------------------

As was shown previously, the compilation of source programs can be achieved in pipelines by
using :meth:`QAT.compile() <qat.core.qat.QAT.compile>`. This would retrieve a pipeline and
use its various modules to compile the source program. Loosely speaking, these modules can
be separated into three different stages: the **frontend**, **middleend** and **backend**.
The source code enters the front end, which compiles it to QAT's intermediate representation
(IR). This enters the middle end, which analyses, validates and optimises it. Finally,
the modified QAT IR enters the backend, which generates native code for the target device.


.. image:: images/compile.png

.. contents::

Frontend
***********************

The frontend describes the front-facing part of the compiler.
It's responsible for dealing with the semantics of the source language, and compiling it
into QAT's intermediate representation (IR) which is independent of source language. A
frontend is usually defined by two components: 

#. A pipeline which can be used to validate and modify the provided source program (before
   compiling to QAT IR), see :mod:`qat.frontend.passes` for a list of passes available.
#. A parser that generates an abstract syntax tree (AST) from the source program, and
   interprets the tree to produce QAT IR.

The general rule of thumb is that there is a single frontend per type of source program.
Currently, QAT supports source programs in the following formats through their appropriate 
frontends:

* `QASM2 <https://arxiv.org/abs/1707.03429>`_:
  :class:`Qasm2Frontend <qat.frontend.qasm.Qasm2Frontend>`
* `QASM3 <https://arxiv.org/abs/2104.14722>`_:
  :class:`Qasm3Frontend <qat.frontend.qasm.Qasm3Frontend>`
* `QIR <https://www.qir-alliance.org/qir-book/>`_:
  :class:`QIRFrontend <qat.frontend.qir.QIRFrontend>`

QASM2 example 
^^^^^^^^^^^^^^^^^^^^

Each frontend can be provided with a compilation pipeline, with a suitable default already 
chosen for each. As an example, let's try to compile a QASM2 program 

.. code-block:: python 
    :linenos:

    from qat.frontend.qasm import Qasm2Frontend
    from qat.model.loaders.legacy import EchoModelLoader
    from qat.frontend.analysis_passes import InputAnalysis
    from qat.frontend.transform_passes import InputOptimisation
    from qat.passes.pass_base import PassManager

    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    model = EchoModelLoader(8).load()
    pipeline = PassManager() | InputAnalysis() | InputOptimisation(model)
    frontend = Qasm2Frontend(model, pipeline=pipeline)
    ir = frontend.emit(qasm_str)

The :code:`ir` emitted is an instruction builder that contains QAT IR, and can now be used 
within the middleend of the compilation.

Automatically chosen frontends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often enough, we are not explicitly told what is the type of source program. We need 
to infer the type and then decide on the correct frontend to use. We would also like to 
avoid defining individual pipelines for each type of source program: it is sometimes
expected that the middleend and backend will be the same regardless of the type of source
program. For these reasons, it makes sense to have an automatic frontend that inspects the
source program to determine and deploy the matching frontend. 

This is achieved in QAT using the :class:`AutoFrontend <qat.frontend.auto.AutoFrontend>`.
This frontend is provided with a list of frontends and attempts to match the source
program with one of the given frontends. If a frontend is found to be compatible with the
source program, it is used. Defining an automatic frontend might look something like:

.. code-block:: python 

    frontend = AutoFrontend(
        model,
        Qasm2Frontend(model),
        Qasm3Frontend(model),
        QIRFrontend(model)
    )

The natural question that follows is "how does a frontend know if it is compatible with
the source program?" Each source-language specific frontend must be equipped with a 
:meth:`check_and_return_source <qat.frontend.base.BaseFrontend.check_and_return_source>`
method that inspects the contents of the source program, and returns it if it is found to be
compatible.

Let's consider a simple example to demonstrate how a custom frontend can be used within an 
:class:`AutoFrontend <qat.frontend.auto.AutoFrontend>`. 

.. code-block:: python
    :linenos:

    from qat.frontend import BaseFrontend, AutoFrontend, Qasm2Frontend
    from qat.purr.backends.echo import get_default_echo_hardware

    class MyCustomFrontend(BaseFrontend):

        def check_and_return_source(self, src):
            if not isinstance(src, str):
                return False
            if "this is a fancy new source language" in src:
                return src 
            return False
        
        def emit(self, src, *args):
            return src
        

    qasm_program = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    custom_program = "this is a fancy new source language"
    invalid_program = "this will not work"

    model = get_default_echo_hardware(8)
    frontend = AutoFrontend(model, Qasm2Frontend(model), MyCustomFrontend(model))
    qasm_frontend = frontend.assign_frontend(qasm_program)
    custom_frontend = frontend.assign_frontend(custom_program)
    no_frontend = frontend.assign_frontend(invalid_program)

The types of the returned objects will be
:class:`Qasm2Frontend <qat.frontend.qasm.Qasm2Frontend>`, :class:`MyCustomFrontend`and
:class:`NoneType` respectively.

Alternative frontends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QAT also has a few other frontends available:

* :class:`FallthroughFrontend <qat.frontend.fallthrough.FallthroughFrontend>`: Simply passes
  through the source program. Used in situations where no frontend is required.
* :class:`CustomFrontend <qat.frontend.custom.CustomFrontend>`: Allows a custom
  pipeline-oriented frontend to be defined.


Middleend 
**********************

The middleend module handles compilation responsibilities at the level of QAT IR. It is
passed a quantum program at the level of QAT IR, and applies a sequence of hardware-agnostic 
passes that

#. Perform analysis on the IR,
#. Validate the IR has appropriate specified properties,
#. Transforms the IR (e.g. optimization and santisation of the IR).

See :mod:`qat.middleend.passes` for a full list of available passes. 

Default Middleend
^^^^^^^^^^^^^^^^^^^^^^^

The standard middleend to use is the
:class:`DefaultMiddleend <qat.middleend.middleends.DefaultMiddleend>`, which has a
pre-defined pipeline.

.. code-block:: python 

    from qat.middleend import DefaultMiddleend
    middleend = DefaultMiddleend(hardware_model)
    ir = middleend.emit(ir)

Despite being hardware-agnostic, the middleend needs to be instantiated with the hardware 
model. 
Note that here hardware-agnostic means that it does not depend on the code-generation
related details of the target. However, the calibration file is still required to produce
the correction QAT IR instructions. Calling :code:`middleend.emit(ir)` will instruct the 
middleend to pass the IR through the compilation pipeline.

Custom Middleend
^^^^^^^^^^^^^^^^^^^^^^^

We can specify a middleend with a custom compilation pipeline using the
:class:`CustomMiddleend <qat.middleend.middleends.CustomMiddleend>` class. For example, the
following middleend would optimise over phase shifts and remove any unnecessary
post-processing instructions to reduce the overall instruction count.

.. code-block:: python 

    from qat.middleend import CustomMiddleend
    from qat.passes.pass_base import PassManager
    from qat.compiler.transform_passes import PhaseOptimisation, PostProcessingSanitisation

    pipeline = (
        PassManager()
        | PhaseOptimisation()
        | PostProcessingSanitisation()
    )
    middleend = CustomMiddleend(model, pipeline)


Backend 
***********************

After we have compiled the source program to QAT IR, and performed any validation and
transformation we wish to do at the level of QAT IR, the program will enter the backend. 
The objective of the backend is to compile the QAT IR into a language understandable by the
target, a process referred to as "code generation" (codegen). Like the frontend, the backend
has two components:

#. A compilation pipeline which performs analysis on the IR which is used during codegen,
   validation passes to verify the code is compatible with the native code, and
   transformation passes to make the intermediate code more appropriate for the codegen.
   See :mod:`qat.backend.passes` for a full list of passes (which might be target specific).
#. An emitter that walks through the QAT IR and generates native code.

WaveformV1Backend example
^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example, lets consider the
:class:`WaveformV1Backend <qat.backend.waveform_v1.codegen.WaveformV1Backend>`, a backend
that generates code for earlier (and now legacy) iterations of OQC hardware, but is still
maintained for some of our simulators and demonstration purposes. Like the frontend and
middleend, the emitter can be used to generate native code by calling the :code:`emit`
method,

.. code-block:: python 

    from qat.backend.waveform_v1 import WaveformV1Backend
    backend = WaveformV1Backend(model)
    pkg = backend.emit(ir)

The package returned from a backend is referred to as an
:py:class:`Executable <qat.runtime.executables.Executable>`. They are Pydantic data classes
that contain all the information needed to execute a program, including the instructions
needed by the control hardware (or simulator), and the classical post-processing
instructions required by the runtime needed to interpret and process the results (see the
execution section for more details).

Making ends meet
*********************

Now that we have covered each type of "end", we can bring it together to define a complete
compilation pipeline. Let's write one that compiles to "WaveformV1".

.. code-block:: python

    from qat.model.loaders.legacy import EchoModelLoader
    from qat.frontend import AutoFrontend
    from qat.middleend import DefaultMiddleend
    from qat.backend.waveform_v1 import WaveformV1Backend
    from compiler_config.config import CompilerConfig

    model = EchoModelLoader(8).load()
    frontend = AutoFrontend(model)
    middleend = DefaultMiddleend(model)
    backend = WaveformV1Backend(model)

    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    config = CompilerConfig(repeats=1000)


    ir = frontend.emit(qasm_str, compiler_config=config)
    ir = middleend.emit(ir, compiler_config=config)
    pkg = backend.emit(ir, compiler_config=config)

The result will be a freshly prepared package ready for execution! Notice that this just 
achieves what :code:`QAT().compile()` would, but not as neatly wrapped up.