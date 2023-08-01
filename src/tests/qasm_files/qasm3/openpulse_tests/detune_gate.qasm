OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame q0_drive;
    extern frame r0_acquire;
}
defcal detune_gate $0 {
    shift_frequency(r0_acquire, 100.0);
    shift_frequency(q0_drive, -100.0);
}
defcal delay_gate $0 {
    delay[100ns] q0_drive;
}

bit[1] c;
detune_gate $0;
h $0;
delay_gate $0;
h $0;
c[0] = measure $0;