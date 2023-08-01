OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    extern port channel_1;
    frame q0_frame = newframe(q0_drive, 5e9, 0.0);
    waveform wf1 = gaussian(1e-6, 0.0001,  0.0001);
    waveform wf2 = sech(1e-6,  0.0001,  0.0001);
    //waveform wf3 = gaussian_square(1e-6, 0.0001, 0.0001, 0.0001);
    waveform wf4 = drag(1e-6, 0.0001, 0.0001, 0.0001);
    waveform wf5 = constant(1e-6, 0.0001);
    waveform wf6 = sine(1e-6, 0.0001, 2525.5, -pi);
    waveform wf7 = square(1e-6, 0.0001);
    waveform wf8 = soft_square(1e-6, 0.0001);
    waveform wf9 = blackman(1e-6, 0.0001);
    waveform wf10 = softer_square(1e-6, 0.0001);
    waveform wf11 = extra_soft_square(1e-6, 0.0001);

    waveform arb_waveform1 = [1+0, 0+1];
}

defcal x $0 {
   play(q0_frame, wf1);
}

defcal x $1 {
   play(q0_frame, wf1);
}

defcal x qb {
   play(q0_frame, wf1);
}

qreg qr[2];
qreg ql;
creg cl;
creg cr[2];

bit[1] c;
x $0;
x $1;
x $3;
c[0] = measure $0;

cal {
    complex[float] my_complex;      // 32-bit float
    complex[float[64]] my_complex_2;  // 64-bit float
    frame q0_q1_cz_frame = newframe(channel_1, 500ns);
    frame q0_rf_frame = newframe(channel_1, 300dt);
    extern frame r0_acquire;
    extern port channel_2;
    waveform arb_waveform2 = [0.5, 1+0im, 0+1im, 0.3 - 0.5im, 0.9im, pi, tau, euler, 2-4, 5/12, 2+5, 13+62];
}

defcal cz $2 $3 {
}

defcal cz $1, $0 {
    barrier q0_q1_cz_frame, q0_rf_frame;
//    play(q0_q1_cz_frame, mix(wf1, wf1));
//    play(q0_q1_cz_frame, sum(wf8, wf7));
//    play(q0_q1_cz_frame, phase_shift(wf1, pi));
//    play(q0_q1_cz_frame, sum(scale(wf7, 3), wf8));
    delay[300ns] q0_rf_frame;
    // Using shift_phase command
    shift_phase(q0_rf_frame, 4.366186381749424);
    set_phase(q0_rf_frame, 4.366186381749424);
    //get_phase(q0_rf_frame);
    shift_frequency(q0_rf_frame, 4.366186381749424);
    set_frequency(q0_rf_frame, 4.366186381749424);
    //get_frequency(q0_rf_frame);

    delay[300ns] q0_rf_frame;
    // Using += to shift the phase
    q0_rf_frame.phase += 5.916747563126659;
    q0_rf_frame.phase -= 5.916747563126659;
    q0_rf_frame.phase = 5.916747563126659;
    q0_drive.frequency += 4e8;
    q0_drive.frequency -= 4e8;
    q0_drive.frequency = 5e9;
    barrier q0_q1_cz_frame, q0_rf_frame;
    shift_phase(q0_q1_cz_frame, 2.183093190874712);
    reset q0_rf_frame;
    capture_v0(r0_acquire);
    capture_v1(r0_acquire, 1e-6);
    capture_v2(r0_acquire, 1e-6);
    capture_v3(r0_acquire, 1e-6);
    capture_v4(r0_acquire);
}

bit[2] ro;
cz $1,$0;
ro[0] = measure $1;
ro[1] = measure $0;
measure $0;

bit b;     // single bit
bit[2] ba;  // bit register

h $0;         // single unitary gate with physical qubit notation
cnot $0, $1;  // controlled gate with physical qubits (control is $0, target is $1)
swap $0, $1;  // multi-target gate with physical qubits
qubit q;      // virtual qubit declaration
qubit[2] qr;  // virtual qubit register

cal {
    extern frame r0_measure;
}

defcal measure $1 {
    barrier r0_measure, r0_acquire;
    play(r0_measure, wf1);
    return capture_v2(r0_acquire, 1e-6);
}
bit r;
r = measure $1;