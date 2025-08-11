OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    capture_v{{ capture_version }}({{ frame }}, 1e-6);
}