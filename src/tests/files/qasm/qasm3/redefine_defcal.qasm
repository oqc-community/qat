OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    extern port channel_1;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    waveform wf1 = gaussian_square(0.1, 1e-6, 0.5e-6, 10e-9);
    waveform wf2 = square(1e-6, 0.0001);

}

defcal x $0 {
   play(q0_frame, wf1);
}

x $0;


defcal x $0 {
   play(q0_frame, wf2);
}

x $0;

