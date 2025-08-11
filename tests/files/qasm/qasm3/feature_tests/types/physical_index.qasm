// Does a basic x-measurement on a provided physical index; should be used in a test that
// verifies the targeted qubit is the qubit used.

OPENQASM 3.0;
bit[1] c;
x ${{ physical_index }};
measure ${{ physical_index }} -> c[0];
