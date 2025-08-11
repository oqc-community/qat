OPENQASM 3.0;

// load in some external file, templated with jinja2
include "{{lib}}";

x q; 
measure q -> c;

