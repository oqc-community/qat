OPENQASM 3;
defcalgrammar "openpulse";
cal {
    extern frame q0_q1_cross_resonance;
    waveform wf1 = [0.5, 0.5im, -0.5, -0.5im, 1 + 0.0im, 0 + 1.0im, 0.3 - 0.5im, 0.9im, -0.5 + 0.1im, 0.5, 0.5, 0.5];
}
defcal xy(angle theta) $0, $1 {
    set_phase(q0_q1_cross_resonance, 0.5);
    set_phase(q0_q1_cross_resonance, 0.0);
    shift_phase(q0_q1_cross_resonance, -0.9472540078352163);
    play(q0_q1_cross_resonance, wf1);
    shift_phase(q0_q1_cross_resonance, -0.6183955142424977 - 0.5 * theta);
    shift_phase(q0_q1_cross_resonance, -0.25 * theta);
    shift_frequency(q0_q1_cross_resonance, 1005.1);
    shift_frequency(q0_q1_cross_resonance, 100.0);
    set_frequency(q0_q1_cross_resonance, 1000000000.0);
    set_frequency(q0_q1_cross_resonance, 12000000.0 * theta);
    barrier q0_q1_cross_resonance;
}
bit[2] c;
xy(0.5) $0, $1;
c[0] = measure $0;