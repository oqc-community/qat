defcalgrammar "openpulse";

bit[2] ro;
cz $1,$0;
ro[0] = measure $1;
ro[1] = measure $0;
measure $0;
