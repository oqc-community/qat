OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg b[2];
h q[0];
cx q[0], q[1];
measure q -> b;