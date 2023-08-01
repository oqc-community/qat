OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q;
creg c[2];
creg b[2];
creg a[2];
measure q->c;