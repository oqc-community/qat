OPENQASM 3.0;
defcalgrammar "openpulse";
bit[2] c;
cal {
    extern frame r1_acquire;
    extern frame r2_acquire;
    extern frame r1_measure;
    extern frame r2_measure;
    waveform wf = constant(1e-06, 1.0);
    waveform wf2 = constant(1e-06, 0.5);
    play(r1_measure, wf);
    play(r2_measure, wf2);
    c[0] = capture_v1(r1_acquire, 1e-6);
    c[1] = capture_v1(r2_acquire, 1e-6);
}
