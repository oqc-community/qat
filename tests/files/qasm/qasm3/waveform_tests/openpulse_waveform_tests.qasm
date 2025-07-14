OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    waveform wf1 = gaussian(3e-3, 1e-6, 5e-5);
    waveform wf2 = rounded_square(1e-6, 5e-5, 1e-5, 3e-3);
    waveform wf3 = drag(1e-7, 160dt, 40dt, 0.1);
    waveform wf4 = constant(1e-6, 3e-3);

}

defcal x $0 {
   play(q0_frame, wf1);
   play(q0_frame, wf2);
   play(q0_frame, wf3);
   play(q0_frame, wf4);
}

x $0;
