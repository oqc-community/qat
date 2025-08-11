// The reset operation to set a gate back to its zero state.
// Contains a number of situations that should test the different use cases of a reset.
// The last one currently fails.

OPENQASM 3;
reset ${{ physical_index1 }};

x ${{ physical_index2 }};
reset ${{ physical_index2 }};

bit[1] c;
x ${{ physical_index3 }};
reset ${{ physical_index3 }};
x ${{ physical_index3 }};
measure ${{ physical_index3 }} -> c[0];