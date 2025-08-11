OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    set_phase({{ frame }}, 0.5);
    set_phase({{ frame }}, 0.254);
}