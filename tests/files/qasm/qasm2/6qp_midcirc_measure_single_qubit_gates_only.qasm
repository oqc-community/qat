OPENQASM 2.0;
include "qelib1.inc";

qreg q[6];
creg c0[1];  // first measurement result (single qubit)
creg c1[6];  // second measurement results (post-Hadamard)


// Apply Hadamard to all qubits
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];


// First measurement (q[0] only)
measure q[0] -> c0[0];

// Apply Hadamard to all qubits
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];

// Second measurement
measure q -> c1;
