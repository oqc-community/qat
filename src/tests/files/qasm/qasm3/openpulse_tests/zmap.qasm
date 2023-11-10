OPENQASM 3;
defcalgrammar "openpulse";

cal{
    waveform wf = square(0.00000001, 0.0);
    extern frame r0_measure;
    extern frame r1_acquire;
}

defcal measure $0{
    barrier r0_measure, r1_acquire;
    play(r0_measure, wf);
    return capture_v1(r1_acquire, 0.0);
}

rx(90) $0;
measure $0;