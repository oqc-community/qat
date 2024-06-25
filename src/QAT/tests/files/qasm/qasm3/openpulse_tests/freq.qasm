OPENQASM 3;
defcalgrammar "openpulse";

cal {
   extern port channel_1;
   extern port channel_2;
   extern frame q0_drive;
}

defcal cross_resonance $0, $1 {
    waveform wf1 = sine(1., 1024dt, 128dt, 32dt);
    waveform wf2 = sine(0.1, 1024dt, 128dt, 32dt);
    frame temp_frame = newframe(channel_2, q0_drive.frequency, 0);
    frame temp_frame2 = newframe(channel_2, get_frequency(q0_drive), 0);
    play(temp_frame2, wf1);
    play(temp_frame, wf2);
}
cross_resonance $0, $1;