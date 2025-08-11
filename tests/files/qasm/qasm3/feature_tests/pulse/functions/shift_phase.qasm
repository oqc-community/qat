OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    shift_phase({{ frame }}, 0.254);
}