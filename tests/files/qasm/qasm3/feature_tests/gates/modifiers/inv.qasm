// Tests the `inv @ ` modifer, that allows us to call the inverse of a gate.

OPENQASM 3.0;
qubit[1] q;
inv @ x {{ physical_index }};