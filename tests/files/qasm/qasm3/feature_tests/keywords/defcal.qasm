// Tests the defcal and cal functionality by delibraitely making an x gate do nothing 
// Would be great if this didn't use delay, but it fails without some operation in defcal 
// and noop / id do not work.

OPENQASM 3;
defcalgrammar "openpulse";

cal {
    extern frame {{ frame }};
}

defcal x ${{ physical_index1 }} {
    delay[0s] {{ frame }};
}

x ${{ physical_index1 }};
x ${{ physical_index2 }};