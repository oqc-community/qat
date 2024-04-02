OPENQASM 3;
defcalgrammar "openpulse";
cal {
    extern port channel_1;
}
cal {
    frame my_frame = newframe(channel_1, 4500000000.0);
    waveform wf1 = constant(10e-9, 1.0);
    waveform wf2 = constant(10e-9, 0.1im);
    waveform wf3 = constant(10e-9, -0.1 + 0.1im);
    waveform wf4 = constant(10e-9, -0.1);
    waveform wf5 = constant(10e-9, -0.1 - 0.1im);
}
defcal x $0 {
    play(my_frame, wf1);
    play(my_frame, wf2);
    play(my_frame, wf3);
    play(my_frame, wf4);
    play(my_frame, wf5);
}

bit c;
x $0;
c = measure $0;