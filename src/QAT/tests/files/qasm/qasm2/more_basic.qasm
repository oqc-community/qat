OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
gate dave q1 { }
h q;
creg c[2];
measure q->c;