// Plays a gaussian square waveform, templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = gaussian_square({{ amp }}, {{ width }}, {{ square_width }}, {{ sigma }});
    play({{ frame }}, wf1);
}