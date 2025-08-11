// Plays a constant waveform, templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = constant({{ width }}, {{ amp }});
    play({{ frame }}, wf1);
}