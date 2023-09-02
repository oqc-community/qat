OPENQASM 3;
defcalgrammar "openpulse";
cal {
extern frame q0_drive;
}
cal {
    waveform wf1 = soft_square(1.0, 2e-08);
}
defcal rx(4, 2) q {
    play(q0_drive, wf1);
}
rx(4, 2) $0;
