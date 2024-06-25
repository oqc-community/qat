OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q;
creg c[2];
measure q[0]->c[0];
measure q[1]->c[1];