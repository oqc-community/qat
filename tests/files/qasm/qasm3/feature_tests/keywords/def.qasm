// Tests function declaration 
OPENQASM 3;

def x_gate(qubit q) {
    x q;
}

qubits[1] q;
x_gate(q[0]);
