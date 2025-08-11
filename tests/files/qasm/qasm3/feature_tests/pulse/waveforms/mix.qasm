// Mixs (element-wise multiplication) two waveforms, templated by jinja2.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = constant({{ width }}, {{ amp1 }});
    waveform wf2 = constant({{ width }}, {{ amp2 }});
    wf3 = mix(wf1, wf2);
    play({{ frame }}, wf3);
}