OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    extern port channel_1;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    waveform arb_waveform = [1+0, 0+1im];
}

defcal x $0 {
   play(q0_frame, arb_waveform);
}

x $0;
