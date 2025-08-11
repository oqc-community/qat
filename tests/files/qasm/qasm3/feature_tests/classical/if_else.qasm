// Tests an if-else statement 

OPENQASM 3;

bit[1] c;
qubit[1] q;
bit[1] output;

x q[0];
output[0] = measure q[0];
if (output[0] == false) {
    x q[0];
} else {
    h q[0];
}
