// Plays a sine waveform, templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 =sine({{ amp }}, {{ width }}, {{ frequency }}, {{ phase }});
    play({{ frame }}, wf1);
}