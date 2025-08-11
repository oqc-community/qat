// Moves the internal phase of a waveform, templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = constant({{ width }}, {{ amp }});
    wf2 = phase_shift(wf1, {{ phase }});
    play({{ frame }}, wf2);
}