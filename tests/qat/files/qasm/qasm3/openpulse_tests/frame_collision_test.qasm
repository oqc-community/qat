OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    extern port channel_1;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    waveform wf1 = gaussian(1e-6, 1.0, 0);
    waveform wf2 = sech(1e-6, 1.0, 1.0);
}

defcal single_qubit_gate $0 {
  play(wf1, q0_frame);
}

defcal single_qubit_gate $1 {
  play(wf1, q0_frame);
}

// Compile-time error when requesting parallel usage of the same frame
single_qubit_gate $0,$1;
