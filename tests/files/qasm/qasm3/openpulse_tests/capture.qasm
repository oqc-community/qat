OPENQASM 3;
defcalgrammar "openpulse";

cal {
      extern frame r0_measure;
      extern frame r0_acquire;
      waveform wf1 = gaussian(1.0, 18ns, 0.20);

}
defcal measure $0 {
      barrier r0_measure, r0_acquire;
      play(r0_measure, wf1);
      capture_v2(r0_acquire, 0.000001);
}

measure $0;