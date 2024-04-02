OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    waveform wf1 = gaussian(1e-6, 0.0001, 0.0001);
    waveform wf2 = gaussian(1e-6, 0.0001, 0.0001, true);
    waveform wf3 = gaussian(1e-6, 0.0001, 0.0001, false);

}

defcal x $0 {
   play(q0_drive, wf1);
   play(q0_drive, wf2);
   play(q0_drive, wf3);
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

