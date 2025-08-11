OPENQASM 3.0;

// redefine the x gate with the identity
gate x q { id q; }

qubit[1] q;
x q[0]; 
