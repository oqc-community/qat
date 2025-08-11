// Contains calls to all gates in stdgates.inc.
// Done on logical qubits; will require hardware model with at least 5 qubits and an 
// appropriate connectivity

OPENQASM 3;
include "stdgates.inc";
bit[2] c;
qubit[2] q;

// 1q gates 
u3({{ angle1 }}, {{ angle2 }}, {{angle3}}) q[0];
u2({{ angle1 }}, {{ angle2 }}) q[1];
u1({{ angle1 }}) q[1];
u({{ angle1 }}, {{ angle2 }}, {{ angle3 }}) q[1];
p({{ angle1 }}) q[0];
phase({{ angle1 }}) q[0];
x q[0];
y q[1];
z q[1];
h q[0];
s q[0];
sdg q[1];
t q[1];
tdg q[0];
rx({{ angle1 }}) q[0];
ry({{ angle2 }}) q[1];
rz({{ angle3 }}) q[1];
sx q[0];
id q[0];

// 2q gates 
CX q[0], q[1];
cx q[0], q[1];
cy q[0], q[1];
cz q[0], q[1];
cp( {{angle1}} ) q[0], q[1];
cphase( {{angle1}} ) q[0], q[1];
swap q[0], q[1];
ch q[0], q[1];
crx({{ angle1 }}) q[0], q[1];
cry({{ angle2 }}) q[0], q[1];
crz({{ angle3 }}) q[0], q[1];
cu({{ angle1 }}, {{ angle2 }}, {{ angle3 }}) q[0], q[1];

// 3q gates 
ccx q[0], q[1], q[2];
cswap q[0], q[1], q[2];