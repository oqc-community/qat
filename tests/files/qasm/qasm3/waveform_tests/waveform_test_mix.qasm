OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    extern port channel_1;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    waveform wf1 = constant(1e-6, 3e-2);
    waveform wf2 = constant(1e-6, 2e-2);
}

defcal x $0 {
   play(q0_frame, mix(wf1, wf2));
}

x $0;
