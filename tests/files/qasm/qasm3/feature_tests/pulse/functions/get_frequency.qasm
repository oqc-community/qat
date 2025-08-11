// Tests the ability to retrieve the frequency from a frame.
// It's coupled to shift_frequency, which is essentially required to properly test
// behaviour.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    shift_frequency({{ frame }}, 0.05*get_frequency({{ frame }}));
}