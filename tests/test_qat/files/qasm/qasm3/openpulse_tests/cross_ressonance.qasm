OPENQASM 3;
defcalgrammar "openpulse";
cal {
    extern frame q1_q2_cross_resonance;
    extern frame q1_drive;
}
cal {
    waveform wf1 = [0.5, 1 + 0.0im, 0 + 1.0im, 0.3 - 0.5im, 0.9im, 0.5, 0.5, 0.5];
}
defcal cz_custom $1, $0 {
    barrier q1_q2_cross_resonance, q1_drive;
    play(q1_q2_cross_resonance, wf1);
    delay[300ns] q1_drive;
    shift_phase(q1_drive, 4.366186381749424);
    delay[300dt] q1_drive;
    shift_phase(q1_drive, 5.916747563126659);
    barrier q1_q2_cross_resonance, q1_drive;
    shift_phase(q1_q2_cross_resonance, 2.183093190874712);
}
bit[2] ro;
cz_custom $1, $0;
ro[0] = measure $1;
ro[1] = measure $0;