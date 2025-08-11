// Tests the `pow(x) @ ` modifer, that allows us to call the power of a gate.

OPENQASM 3.0;
qubit[1] q;
pow(2) @ x {{ physical_index }};