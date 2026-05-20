OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame r1_acquire;
}

defcal measure $1 {
    my_number = capture_v1(r1_acquire, 1e-06);
    return my_number;
}

measure $1;
