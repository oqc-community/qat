OPENQASM 3.0;
include "qelib1.inc";
defcalgrammar "openpulse";

cal {
    extern frame q1_drive;
    extern frame r1_measure;
    extern frame r1_acquire;
}

defcal sx $1 {
   waveform wf_drive = gaussian(1e-4, 160ns, 50ns);
   play(q1_drive, wf_drive);
}

defcal measure $1 {
    barrier r1_measure, r1_acquire;
    waveform wf_measure = constant(5us, 0.04);
    play(r1_measure, wf_measure);
    return capture_v1(r1_acquire, 5us);
}

bit[1] c;
sx $1;
measure $1 -> c[0];
