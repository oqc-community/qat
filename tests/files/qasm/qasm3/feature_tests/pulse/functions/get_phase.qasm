// Tests the ability to retrieve the phase from a frame.
// This might not cover every use case (e.g. more complex gate calls before getting the 
// phase). 
// It's coupled to shift_phase, which is essentially required to properly test behaviour.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    shift_phase({{ frame }}, pi);
    shift_phase({{ frame }}, get_phase({{ frame }}));
}