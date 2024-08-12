OPENQASM 3;
defcalgrammar "openpulse";
cal {
    extern frame q0_drive;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    // amp, width, rise, drag, phase
    waveform wf1 = gaussian_rise(1, 1e-6, 0.3, 0.0, 0.0);
    waveform wf2 = soft_square_rise(1, 1e-6, 0.3, 0.0, 0.0);
}
defcal custom_gate $0 {
   play(q0_frame, wf1);
   play(q0_frame, wf2);
}
bit[1] ro;
custom_gate $0;
ro[0] = measure $0;
