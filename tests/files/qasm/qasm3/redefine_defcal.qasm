OPENQASM 3;
defcalgrammar "openpulse";

cal {
   extern frame q0_drive;
   extern port channel_1;
   frame q0_frame = newframe(q0_drive, 5e9, 0.0);
   waveform wf1 = gaussian(1e-6, 0.0001, 0.0001);
   waveform wf2 = constant(0.0001, 1e-6);

}

defcal x $0 {
   play(q0_frame, wf1);
}

x $0;


defcal x $0 {
   play(q0_frame, wf2);
}

x $0;

