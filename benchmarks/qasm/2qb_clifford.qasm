OPENQASM 2.0;
include "qelib1.inc";
gate gate_Clifford_2Q_5451_ q0,q1 { h q1; h q0; s q0; h q1; s q1; y q0; z q1; }
gate gate_Clifford_2Q_5896_ q0,q1 { h q0; cx q0,q1; h q0; s q0; sdg q1; h q1; y q0; }
gate gate_Clifford_2Q_8699_ q0,q1 { h q0; h q1; cx q0,q1; h q1; s q1; y q0; z q1; }
gate gate_Clifford_2Q_10949_ q0,q1 { h q0; h q1; h q0; s q0; sdg q1; h q1; cx q0,q1; sdg q0; h q0; x q0; x q1; }
gate gate_Clifford_2Q_401_ q0,q1 { sdg q1; h q1; cx q0,q1; sdg q0; h q0; sdg q1; h q1; x q1; }
gate gate_Clifford_2Q_1660_ q0,q1 { sdg q0; h q0; h q1; s q1; cx q0,q1; h q1; s q1; z q0; }
gate gate_Clifford_2Q_9480_ q0,q1 { h q0; h q1; h q1; s q1; cx q0,q1; cx q1,q0; h q1; s q1; y q0; }
gate gate_Clifford_2Q_10928_ q0,q1 { h q0; h q1; h q0; s q0; sdg q1; h q1; cx q0,q1; h q1; s q1; }
gate gate_Clifford_2Q_2871_ q0,q1 { h q0; s q0; h q1; s q1; cx q0,q1; cx q1,q0; cx q0,q1; x q0; z q1; }
gate gate_Clifford_2Q_3592_ q0,q1 { h q1; h q1; s q1; cx q0,q1; sdg q0; h q0; y q0; }
gate gate_Clifford_2Q_10011_ q0,q1 { h q0; h q1; sdg q0; h q0; sdg q1; h q1; cx q0,q1; sdg q0; h q0; sdg q1; h q1; y q0; z q1; }
gate gate_Clifford_2Q_4876_ q0,q1 { h q1; h q0; s q0; cx q0,q1; sdg q0; h q0; z q0; }
gate gate_Clifford_2Q_3146_ q0,q1 { h q1; cx q0,q1; cx q1,q0; h q0; s q0; y q0; y q1; }
gate gate_Clifford_2Q_9535_ q0,q1 { h q0; h q1; h q1; s q1; cx q0,q1; cx q1,q0; sdg q0; h q0; h q1; s q1; z q0; z q1; }
gate gate_Clifford_2Q_2960_ q0,q1 { h q1; cx q0,q1; sdg q0; h q0; sdg q1; h q1; }
gate gate_Clifford_2Q_4713_ q0,q1 { h q1; sdg q0; h q0; h q1; s q1; cx q0,q1; cx q1,q0; sdg q0; h q0; sdg q1; h q1; y q0; x q1; }
gate gate_Clifford_2Q_7416_ q0,q1 { h q0; sdg q0; h q0; h q1; s q1; cx q0,q1; h q1; s q1; y q0; }
gate gate_Clifford_2Q_6331_ q0,q1 { h q0; sdg q1; h q1; cx q0,q1; cx q1,q0; sdg q0; h q0; h q1; s q1; y q0; z q1; }
gate gate_Clifford_2Q_987_ q0,q1 { sdg q0; h q0; cx q0,q1; y q0; z q1; }
gate gate_Clifford_2Q_317_ q0,q1 { cx q0,q1; cx q1,q0; cx q0,q1; z q0; x q1; }
gate gate_Clifford_2Q_976_ q0,q1 { sdg q0; h q0; cx q0,q1; }
qreg q[2];
creg meas[2];
gate_Clifford_2Q_5451_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_5896_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_8699_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_10949_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_401_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_1660_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_9480_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_10928_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_2871_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_3592_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_10011_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_4876_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_3146_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_9535_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_2960_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_4713_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_7416_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_6331_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_987_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_317_ q[0],q[1];
barrier q[0],q[1];
gate_Clifford_2Q_976_ q[0],q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];