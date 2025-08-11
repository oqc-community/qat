// There are other ways to use extern, but we don't provide any other instrinsics in this 
// way. We might adjust this test later...

OPENQASM 3;
defcalgrammar "openpulse";
cal {
    extern frame {{ frame }};
}