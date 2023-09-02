OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    extern port channel_1;
    waveform wf1 = constant(10e-9, 0.3);
    waveform wf2 = constant(10e-9, 0.2);
}

defcal x $0 {
   play(q0_drive, sum(wf1, wf2));
}

x $0;
