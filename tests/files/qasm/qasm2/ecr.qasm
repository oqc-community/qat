OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg meas[2];
h q[0];
ecr q[0],q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];