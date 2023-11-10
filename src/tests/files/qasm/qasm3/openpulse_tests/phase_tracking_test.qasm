defcalgrammar "openpulse";

cal {
  extern port tx0;
  waveform p = /* ... some 100ns waveform ... */;

  // Frame initialized with accrued phase of 0
  frame driveframe0 = newframe(tx0, 5.0e9, 0);
}

defcal single_qubit_gate $0 {
  play(driveframe0, wf);
}

defcal single_qubit_delay $0 {
  delay[13ns] driveframe0;
}

// get_phase(driveframe0) == 0
single_qubit_gate $0;
// Implicit advancement: -> shift_phase(driveframe0, 2π * get_frequency(driveframe0) * durationof(wf))
//                        = shift_phase(driveframe0, 2π * 5e9 * 100e-9)

// Change the frequency
cal {
  set_frequency(driveframe0, 6e9);
}

single_qubit_delay $0;
// Implicit advancement: -> set_phase(driveframe0, 2π * get_frequency(driveframe0) * 13e-9)
//                        = set_phase(driveframe0, 2π * 6e9 * 13e-9)