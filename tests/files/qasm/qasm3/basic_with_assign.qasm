// This test is designed to test the assign operator = within a measurement, which is in
// contrast to the more used syntax measure q -> c;

OPENQASM 3;
bit[2] c;
qubit[2] q;
h q;
c[0] = measure q[0];
c[1] = measure q[1];
