// Tests a break naively; should only do one iteration

OPENQASM 3;

bit[1] c;
qubit[1] q;

int i = 0;
while (i < 5) {
    x q[0];
    break;
}