OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q1_drive;
    extern frame q2_drive;
    extern port channel_1;
    extern port channel_2;
    frame q1_frame = newframe(q1_drive, 5e9, 0.0);
    frame q2_frame = newframe(q2_drive, 5.1e9, 0.0);
    waveform wf1 = rounded_square(1e-6, 1.0, 50e-9, 8ns);
}

defcal cx $1,$2 {
   play(q1_frame, wf1);
   play(q2_frame, wf1);
}


cx $0,$1;
cx $1,$2;
cx $2,$3;