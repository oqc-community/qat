// Tests the `ctrl @ ` modifer, that allows us to define arbitrary control-gates.

OPENQASM 3.0;
qubit[2] q;
ctrl @ x ${{ physical_index1 }}, ${{ physical_index2 }};