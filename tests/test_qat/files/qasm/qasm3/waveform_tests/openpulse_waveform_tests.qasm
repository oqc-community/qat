OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    waveform wf1 = gaussian(1e-6, 3.0);
    waveform wf2 = rounded_square(1e-6, 2.0);
    waveform wf3 = DRAG_gaussian(1e-6, 2.0);
    waveform wf4 = constant(1e-6, 2.0);

}

defcal x $0 {
   play(q0_frame, wf1);
   play(q0_frame, wf2);
   play(q0_frame, wf3);
   play(q0_frame, wf4);
}

x $0;
