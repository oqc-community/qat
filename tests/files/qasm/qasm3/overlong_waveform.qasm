// openpulse waveform definition with a (ludicrously) long pulse width.
// Qat should reject this in the InstructionValidation step
// See COMPILER-936 for more info

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q1_drive;
    extern frame r1_measure;
    extern frame r1_acquire;

    extern frame q2_drive;
    extern frame r2_measure;
    extern frame r2_acquire;

    extern frame q1_q2_cross_resonance;
}

cal {
    waveform set = constant(2*3.141592653589793* get_frequency(q1_drive), 1e-8);
    play(q1_drive, set);
}

bit[1] c;
measure $1 -> c[0];
