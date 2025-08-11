// Does an x gate to emphasize we don't have full support: cant change freq mid circuit

OPENQASM 3;
defcalgrammar "openpulse";

x ${{ physical_index }};

cal {
    extern frame {{ frame }};
    shift_frequency({{ frame }}, {{ frequency }});
}