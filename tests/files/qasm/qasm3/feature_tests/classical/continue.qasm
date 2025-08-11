// Tests a continue naively; should only apply to first qubit

OPENQASM 3;

bit[2] c;
qubit[2] q;

for int i in [0:1:3] {
    x q[0];

    continue;
    x q[1];
}
measure q -> c;