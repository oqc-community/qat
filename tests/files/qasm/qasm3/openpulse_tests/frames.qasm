
OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern port channel_1;
    extern frame q1_drive;
    frame new_frame_port = newframe(channel_1, 5e5, 0.0);
    frame new_frame_drive = newframe(q1_drive, 6e5, 0.0);
    frame new_frame_from_custom = newframe(new_frame_drive, 7e5, 0.0);
    waveform wf = constant(80ns, 1e-4);
    play(new_frame_port, wf);
    play(new_frame_drive, wf);
    play(new_frame_from_custom, wf);
}