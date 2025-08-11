// Tests the use of an external port declaration (not used as a literal)

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern port {{ port }};
}