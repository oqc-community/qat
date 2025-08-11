// Takes the sum of two waveforms, templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = constant({{ width }}, {{ amp1 }});
    waveform wf2 = constant({{ width }}, {{ amp2 }});
    wf3 = sum(wf1, wf2);
    play({{ frame }}, wf3);
}