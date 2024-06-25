OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    extern port channel_1;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    waveform wf = square(1.0, 1.0);
}

defcal x $0 {
   play(q0_frame, wf);
}

x $0;

