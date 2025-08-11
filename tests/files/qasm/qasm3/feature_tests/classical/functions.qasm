// Tests function definitions and calls 

OPENQASM 3.0;

def xmeasure(qubit q) -> bit {
    h q;
    return measure q;
}

qubit[2] q;
bit[2] c;

c[0] = xmeasure(q[0]);
c[1] = xmeasure(q[1]);