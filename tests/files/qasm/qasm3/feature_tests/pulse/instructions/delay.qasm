OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    delay[{{ time }}] {{ frame }};
}
