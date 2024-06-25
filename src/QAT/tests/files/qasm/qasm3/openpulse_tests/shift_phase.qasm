OPENQASM 3;
defcalgrammar "openpulse";
cal {
    extern frame q0_drive;
}
defcal rz(angle theta) $0 {
    barrier $0;
    shift_phase(q0_drive, -1.0 * theta);
    shift_phase(q0_drive, -0.5 * theta);
    shift_phase(q0_drive, -0.5 * theta);
    barrier $0;
}

bit[1] c;
rz(0.5) $0;
c[0] = measure $0;