// Tests declaration of waveform; use cases are handled in other tests 

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    waveform wf = constant(1.0, 2.0);
}