Core Concepts
=====================

QAT is driven by a few core concepts:

1. A virtual hardware model. This holds all information about a piece of hardware you're wanting to compile and/or run against.

2. An instruction builder. An abstract builder which hides the implementation of various gates and hardware instructions. Builds up abstract hardware instructions.

3. A runtime. The outer shell of execution, deals with auxiliary and generic runtime requirements such as results/argument processing, request batching, things like that.

4. An engine. Consumes hardware instructions to actually execute against a driver, hardware, remote target or simulator. It's the piece that actually executes your request and returns the results.

Each of these can be extended to allow nearly full customization of how you want QAT to optimize, build and run the instructions it generates.

We'll cover each of them in turn, but the hardware model is the most critical so we'll start there.

Hardware Models
---------------

A hardware model is a virtual model of a piece of hardware. It can model down to the smallest nuance physical hardware that exists, doesn't exist, or you want to exist. This can also include simulators and more exotic hardware - we won't judge.

But more than that it also supplies the builders, runtimes and engines that are associated with a piece of hardware and which QAT will use to power almost the entirety of its functionality. It does this because there is no agreed common (low-level) way a quantum computer operates right now. So how each of them perform gates, what classical operations they have available and how they then run it all differ drastically.

Depending upon what you want to override you can change almost every step of compilation and execution purely from providing a custom model.

Instruction Builders
--------------------

The instruction builders are the objects called when QAT parses any IR's. It provides methods for gate-level, pulse-level and classical control. By default it will attempt to call the model for definitions around the various pulses require to represent rotations/unitaries. But if you need more fine-grained control than this or your engine requires an entirely custom set of instructions you can overwrite this entirely.

The builder acts as wrapper and instruction collection. The instructions don't (shouldn't) get pulled out of them, so you can represent your instructions and operations however you want within it.

Runtime
-------

The runtime is the most general section of the stack and will only need overriding if your backend uses increadibly custom instructions, execution and results formats. Or if you just don't like ours and want to replace it.

It deals with every aspect between getting given a builder to execute and it being refined enough to be sent to the engine, then also dealing with iterations, exceptions, results, pretty much everything that isn't running the instructions directly against a driver.

Engine
------

The engine deals with taking the instructions and metadata passed to it and then just executing it. In our systems it takes the role of transforming QATs proto-instructions into the actual hardware instructions and then running them against a driver.

Further Reading
---------------

If you use the existing models and engines in QAT as a template, you should be able to hook up most systems. There aren't as many overrides for runtimes and builders as they aren't required for most setups, but they have pretty clear API's and if you focus on the model and engine first, the other two will be much easier to build as you'll be more familiar with the code at that point.