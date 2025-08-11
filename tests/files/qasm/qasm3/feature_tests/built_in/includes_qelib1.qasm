// Contains calls to all gates in qelib1.inc.
// Done on logical qubits; will require hardware model with at least 5 qubits and an 
// appropriate connectivity

OPENQASM 3;
include "qelib1.inc";
bit[5] c;
qubit[5] q;

// 1q gates 
u3({{ angle1 }}, {{ angle2 }}, {{angle3}}) q[0];
u2({{ angle1 }}, {{ angle2 }}) q[1];
u1({{ angle1 }}) q[2];
u0({{ angle1 }}) q[3];
id q[0];
u({{ angle1 }}, {{ angle2 }}, {{ angle3 }}) q[2];
p({{ angle1 }}) q[3];
x q[0];
y q[1];
z q[2];
h q[3];
s q[0];
sdg q[1];
t q[2];
tdg q[3];
rx({{ angle1 }}) q[0];
ry({{ angle2 }}) q[1];
rz({{ angle3 }}) q[2];
sx q[3];
sxdg q[0];

// 2q gates 
cx q[0], q[1];
cz q[0], q[1];
swap q[0], q[1];
ch q[0], q[1];
crx({{ angle1 }}) q[0], q[1];
cry({{ angle2 }}) q[0], q[1];
crz({{ angle3 }}) q[0], q[1];
cu1({{ angle1 }}) q[0], q[1];
cp({{ angle1 }}) q[0], q[1];
cu3({{ angle1 }}, {{ angle2 }}, {{ angle3 }}) q[0], q[1];
cu({{ angle1 }}, {{ angle2 }}, {{ angle3 }}) q[0], q[1];
csx q[0], q[1];
rxx({{ angle1 }}) q[0], q[1];
rzz({{ angle2 }});

// 3q gates 
ccx q[0], q[1], q[2];
cswap q[0], q[1], q[2];
rccx q[0], q[1], q[2];

// 4q gates 
rc3x q[0], q[1], q[2], q[3];
c3x q[0], q[1], q[2], q[3];
c3sqrtx q[0], q[1], q[2], q[3];

// 5q gates 
c4x q[0], q[1], q[2], q[3], q[4];