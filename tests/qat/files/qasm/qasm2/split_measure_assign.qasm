OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rx(pi) q;
creg a[2];
creg b[2];
measure q[0]->a[1];
measure q[1]->b[0];