OPENQASM 3;
defcalgrammar "openpulse";

cal {
   // Access globally (or externally) defined ports
   extern port channel_1;
   extern port channel_2;
   frame frame0 = newframe(channel_1, 5.0e9, 0);
}

defcal cross_resonance $0, $1 {
    waveform wf1 = gaussian_square(1., 1024dt, 128dt, 32dt);
    waveform wf2 = gaussian_square(0.1, 1024dt, 128dt, 32dt);

    /*** Do pre-rotation ***/

    // generate new frame for second drive that is locally scoped
    // initialized to time at the beginning of `cross_resonance`
    frame temp_frame = newframe(channel_2, get_frequency(frame0), get_phase(frame0));

    play(frame0, wf1);
    play(temp_frame, wf2);

    /*** Do post-rotation ***/

}