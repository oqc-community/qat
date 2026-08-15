# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements functions for sampling waveform shapes.

Sampling is done under the pretense of a simple model where the whole waveform is defined
in dimensionless units, and can be sampled between the value :math:`x \\in [-1, 1]`, which
maps on to :math:`t \\in [0, T]` where :math:`T` is the total duration of the waveform.
Similarly, the amplitude is defined in dimensionless units, and the waveform samples can be
complex numbers in the range :math:`|z| \\leq 1`. The definitions of these shapes also do
not allow for global rotations of the waveform in complex space, as that can be factorised
outside of the definition.

This leaves just the shape on the envelope to be defined within the constraints described
above. There are many different shapes that are defined in this module; a best effort
attempt has been made to use consistent naming for the parameters of these shapes. Some
names you might see are:

* ``fractional_breadth``: Used in waveforms that have a slow fractional_rise and fall
  without a distinct square "top" region, such as Gaussian and Sech waveforms. The
  ``fractional_breadth`` describes the fractional_breadth of the envelope. It is
  dimensionless, so the waveform will take the same shape even as the total duration of the
  waveform is changed.
* ``fractional_top_width``: Used in waveforms that have a distinct square "top" region, such
  as Soft Square, Rounded Square, and Gaussian Square waveforms. The
  ``fractional_top_width`` describes the  proportion of the waveform that is square, and is
  a dimensionless parameter between 0 and 1. It is dimensionless, so the waveform will take
  the same shape even as the total duration of the waveform is changed.
* ``fractional_rise``: Used in waveforms that have a fractional_rise and fall sandwiched
  between a square "top" region, such as Soft Square, Rounded Square, and Gaussian Square
  waveforms. The ``fractional_rise`` describes the fractional_breadth of the rising /
  falling region between the approximately zero parts of the waveform and approximately
  unity parts of the waveform. It is dimensionless, so the waveform will take the same shape
  even as the total duration of the waveform is changed.
* ``regularize``: Makes the waveform zero at the edges and unity at the center. This is a
  boolean parameter.

Each waveform shape is implemented in its own module, with a function to sample the
waveform shape, and a function to sample the derivative of the waveform shape at any order.
The derivative is subject to the derivative being mathematically defined, and a concrete
implementation being provided. We try to provide at least the first two orders where
possible. The derivatives can be used to implement the DRAG pulse shaping technique, which
is used to reduce leakage in quantum gates.
"""
