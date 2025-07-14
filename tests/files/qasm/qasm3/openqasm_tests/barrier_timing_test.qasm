OPENQASM 3;
defcalgrammar "openpulse";

cal {
  extern port d0;
  extern port d1;

  driveframe1 = newframe(d0, 5.1e9, 0.0);
  driveframe2 = newframe(d1, 5.2e9, 0.0);

  delay[13ns] driveframe1;

  // driveframe1.time == 13ns, driveframe2.time == 0ns

  // Align frames
  barrier driveframe1, driveframe2;

  // driveframe1.time == driveframe2.time == 13ns
}