OPENQASM 2.0;
include "qelib1.inc";

gate fake_x a { u3(pi,0,pi) a; }

qreg q[3];
h q;
fake_x q[0];
creg c[3];
measure q->c;