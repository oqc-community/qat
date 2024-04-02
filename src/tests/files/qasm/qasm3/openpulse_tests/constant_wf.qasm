OPENQASM 3;
defcalgrammar "openpulse";
cal {
    extern frame q0_drive;
}
cal {
    waveform wf = constant(10e-9, 1.0);
}
defcal pulse $0 {
    play(q0_drive, wf);
}

bit[1] c;
pulse $0;
c[0] = measure $0;