OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q1_drive;
    extern frame r1_measure;
    extern frame r1_acquire;

    extern frame q2_drive;
    extern frame r2_measure;
    extern frame r2_acquire;

    extern frame q1_q2_cross_resonance;
}
cal {
waveform wf1_0_0 = [0+0im,0+0im];
waveform wf2_0_0 = [a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a
a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a a+a];

play(q1_drive, wf1_0_0);
play(q2_drive, wf2_0_0);
play(q1_drive, wf1_0_0);
play(q2_drive, wf2_0_0);
play(q1_drive, wf1_0_0);

play(q2_drive, wf2_0_0);
}
bit[2] c;

measure $1 -> c[0];
measure $2 -> c[1];
