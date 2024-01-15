FAQ
===============================

   *Why is this in Python?*

Mixture of reasons. Primary one is that v1.0 was an early prototype and since the majority of the quantum community
know Python it was the fastest way to build a system which the most people could contribute to building. The API's would
always stick around anyway, but as time goes on the majority of its internals has been, is being, or will be moved to Rust/C++.

   *Where do I get started?*

Our tests are a good place to start as they will show you the various ways to run QAT. Running and then stepping
through how it functions is the best way to learn.

We have what's known as an echo model and engine which is used to test QATs functionality when not attached to a QPU.
You'll see these used almost exclusively in the tests, but you can also use this model to see how QAT functions on
larger and more novel architectures.

High-level architectural documents are incoming and will help explain its various concepts at a glance, but
right now aren't complete.

   *What OS's does QAT run on?*

Windows and Linux are its primary development environments. Most of its code is OS-agnostic but we can't
guarantee it won't have bugs on untried ones. Dependencies are usually where you'll have problems, not the core
QAT code itself.

If you need to make changes to get your OS running feel free to PR them to get them included.

   *I don't see anything related to OQC's hardware here!*

Certain parts of how we run our QPU have to stay propriety and for our initial release we did not have time to
properly unpick this from things we can happily release. We want to release as much as possible and as you're
reading this are likely busy doing just that.

   *Do you have your own simulator?*

We have a real-time chip simulator that is used to help test potential changes and their ramifications to hardware.
It focuses on accuracy and testing small-scale changes so should not be considered a general simulator. 3/4 qubit
simulations is its maximum without runtime being prohibitive.
