OPENQASM 3;
defcalgrammar "openpulse";

cal {
  extern port d0;
  // initialized with absolute time 0 because `cal` is global scope
  frame driveframe1 = newframe(d0, 5.0e9, 0.0);
  waveform wf = gaussian(0.5, 16ns, 4ns);
}

defcal my_gate1 $0 {
  play(driveframe1, wf);
}

defcal my_gate2 $0 {
  // initialized to time at beginning of `my_gate2`
  frame driveframe2 = newframe(d0, 5.0e9, 0.0);
  play(driveframe2, wf);
}

defcal my_gate3 $0 {
  // initialized to time at beginning of `my_gate3`
  frame driveframe3 = newframe(d0, 5.0e9, 0.0);
  play(driveframe3, wf);
}

// driveframe1.time = 0ns when `play(driveframe1, wf)` is issued, advances to 16ns after `play`
my_gate1 $0;
// driveframe2.time = 16ns when initialized via `newframe`
my_gate2 $0;
// driveframe3.time = 32ns when initialized via `newframe`
my_gate3 $0;