
OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    frame new_frame = newframe({{ frame }}, {{ frequency }}, 0.0);
    waveform wf = constant(80ns, 1e-4);
    play({{ frame }}, wf);
    play(new_frame, wf);
}