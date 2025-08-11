OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
    waveform wf1 = constant(80e-9s, 1e-4);
}

defcal x ${{ physical_index }} {
    play({{ frame }}, wf1);
}

x ${{ physical_index }};