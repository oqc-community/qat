OPENQASM 3;
defcalgrammar "openpulse";

cal {
   extern frame q0_drive;
   extern frame q1_drive;
}

defcal rz(theta) $0 {
   shift_phase(q0_drive, theta);
}

defcal rz(theta) $1 {
   shift_phase(q1_drive, theta);
}

defcal sx $0 {
   waveform sx_wf = drag(1e-7, 160dt, 40dt, 0.05);
   play(q0_drive, sx_wf);
}

defcal sx $1 {
   waveform sx_wf = drag(1e-7, 160dt, 40dt, 0.1);
   play(q1_drive, sx_wf);
}

defcal cx $1, $0 {
   waveform CR90p = square(1e-7, 560dt);
   waveform CR90m = gaussian(1e-7, 560dt, 240dt);

   rz(pi/2) $0; rz(-pi/2) $1;
   sx $0; sx $1;
   barrier $0, $1;
   play(q0_drive, CR90p);
   barrier $0, $1;
   sx $0;
   sx $0;
   barrier $0, $1;
   rz(-pi/2) $0; rz(pi/2) $1;
   sx $0; sx $1;
   play(q0_drive, CR90m);
}

defcal Y90p $0 {
   waveform y90p = drag(1e-7, 160dt, 40dt, 0.05);
   play(q0_drive, y90p);
}

// Teach the compiler what the unitary of a Y90p is
gate Y90p q {
   rz(-pi/2) q;
   sx q;
}

// Use this defcal explicitly
Y90p $0;
cx $1, $0;
