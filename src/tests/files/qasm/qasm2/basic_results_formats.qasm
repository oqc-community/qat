OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg ab[2];
creg c[2];
h q;
measure q[0]->c[0];