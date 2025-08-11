// Scales the waveform (multiplies by a constant), templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = constant({{ width }}, {{ amp1 }});
    wf2 = scale(wf1, {{ amp2 }});
    play({{ frame }}, wf2);
}