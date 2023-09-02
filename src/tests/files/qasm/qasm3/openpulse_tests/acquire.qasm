OPENQASM 3;
defcalgrammar "openpulse";
cal {
    extern frame q1_q2_cross_resonance;
    extern frame q1_drive;
    extern frame r1_acquire;
    extern frame r2_acquire;
}
cal {
    waveform wf1 = [0.5, 1 + 0.0im, 0 + 1.0im, 0.3 - 0.5im];
    barrier q1_q2_cross_resonance, q1_drive;
    play(q1_q2_cross_resonance, wf1);
    delay[300ns] q1_drive;
    shift_phase(q1_drive, 4.366186381749424);
    shift_phase(q1_drive, 5.916747563126659);
    delay[300dt] q1_drive;
    barrier q1_q2_cross_resonance, q1_drive;
    shift_phase(q1_q2_cross_resonance, 2.183093190874712);
    bit[2] ro;
    ro[0] = measure $0;
    ro[1] = measure $1;
}
bit[1] c;
x $0;
c[0] = measure $0;