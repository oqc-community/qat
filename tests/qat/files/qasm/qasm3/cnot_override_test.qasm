OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q1_drive;
    extern frame q2_drive;
    extern port channel_1;
    extern port channel_2;
    frame q1_frame = newframe(q1_drive, 5e9, 0.0);
    frame q2_frame = newframe(q2_drive, 5.1e9, 0.0);
    waveform wf1 = extra_soft_square(1e-6, 1.0);
}

defcal cnot $1,$2 {
   play(q1_frame, wf1);
   play(q2_frame, wf1);
}


cnot $0,$1;
cnot $1,$2;
cnot $2,$3;