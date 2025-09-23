OPENQASM 3.0;

// load in some external file, templated with jinja2
include "{{lib}}";

bit[1] c;

y2 ${{ physical_index }}; 
measure ${{ physical_index }} -> c;

