// Plays a sech waveform, templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = sech({{ amp }}, {{ width }}, {{ sigma }});
    play({{ frame }}, wf1);
}