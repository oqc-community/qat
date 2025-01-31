OPENQASM 3;
defcalgrammar "openpulse";
cal {
    extern frame q0_drive;
    extern frame q1_drive;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    frame q1_frame = newframe(q1_drive, 5e9, 0.0);
    waveform wf1 = gaussian_square(1, 100e-9, 50e-9, 10e-9, 1);
    waveform wf2 = gaussian_square(2.5, 200e-9, 50e-9, 5e-9);
}
defcal custom_gate $0 {
    play(q0_frame, wf1);
}
defcal custom_gate $1 {
    play(q1_frame, wf2);
}

custom_gate $0;
custom_gate $1;