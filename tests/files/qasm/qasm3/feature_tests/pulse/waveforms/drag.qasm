// Plays a DRAG waveform, templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = drag({{ amp }}, {{ width }}, {{ sigma }}, {{beta}});
    play({{ frame }}, wf1);
}