// Plays an arbitrary waveform, templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = [{{ samples }}];
    play({{ frame }}, wf1);
}