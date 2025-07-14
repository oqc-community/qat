OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q1_drive;
    extern frame q2_drive;
    extern port channel_1;
    extern port channel_2;
    frame q1_frame = newframe(q1_drive, 5e9, 0.0);
    frame q2_frame = newframe(q2_drive, 5.1e9, 0.0);
    waveform wf1 = constant(1e-7, 4e-4);
}

defcal ecr $1,$2 {
   play(q1_frame, wf1);
   play(q2_frame, wf1);
}


ecr $0,$1;
ecr $1,$2;
ecr $2,$3;