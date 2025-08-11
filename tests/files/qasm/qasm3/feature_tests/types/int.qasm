// Tests the use of an int declaration

OPENQASM 3;

int q_index = 3;
qubit[5] q; 
qubit my_qubit = q[q_index];