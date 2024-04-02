Introduction
===============================

    *QAT (Quantum Assembly Toolkit/Toolchain) is a low-level quantum compiler and runtime which facilitates executing quantum IRs such as QASM, OpenPulse and QIR against QPU drivers. It facilitates the execution of largely-optimised code, converted into abstract pulse-level and hardware-level instructions, which are then transformed and delivered to an appropriate driver.*

Taken straight from the repository this is a good summary of what QAT is.

We call it a 'low-level' quantum compiler because we built it to sit just above our QPU and consume/run IRs that had already been mostly optimized already. The community already has quite a few optimizing compilers we felt there wasn't a need to add another one.

So QATs job is to turn quantum IR into low-level abstract instructions, which then get turned into hardware instructions with the drivers we have available and then run directly against hardware. Its varied components are flexible enough to suit whatever needs you want and doesn't *have* to be the final one in the chain though. For example you can easily use QAT as a parsing and scheduling library and you transform its instructions into whatever form you want.

More interestingly it's the compiler and runtime we use for our machines, so adding features here mean they will be deployed against any hardware we have available that is using the default QAT stack.

This means whatever is built in QAT sits directly next to a QPU for active use. This opens up some interesting options for community experimentation and development.
