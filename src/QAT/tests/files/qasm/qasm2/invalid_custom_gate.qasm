OPENQASM 2.0;
include "qelib1.inc";

gate fake_cswap a,b,c
{
  cx c,b;
  ccx a,b,c;
  cx c,b;
}

qreg q[3];
h q;
fake_cswap q[0], q[1], q[2];
creg c[3];
measure q->c;