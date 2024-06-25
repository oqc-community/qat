OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg b[4];
h q[0];
cx q[0], q[1];
cx q[0], q[2];
cx q[0], q[3];
measure q -> b;