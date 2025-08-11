// Tests different constants. Coupled to phase shift isn't ideal, but we need a way to check
// the value enters the IR.
OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    shift_phase({{ frame }}, {{ value }});
}