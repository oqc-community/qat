OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame r1_acquire;
    complex[float] my_number;
}

defcal measure $1 {
    my_number = capture_v1(r1_acquire, 1e-06);
    return my_number;
}

measure $1;
