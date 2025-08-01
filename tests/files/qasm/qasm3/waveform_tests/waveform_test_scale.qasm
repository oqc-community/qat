OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    extern port channel_1;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    waveform wf1 = constant(1e-6, 1e-3);
}

defcal x $0 {
   play(q0_frame, scale(wf1, 42));
}

x $0;
