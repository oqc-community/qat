// Tests a for loop

OPENQASM 3;

bit[1] c;
qubit[1] q;

for int i in [0:1:10] {
    x q[0];
}
measure q -> c;