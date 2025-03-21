defcalgrammar "openpulse";

const duration pulse_length_start = 20dt;
const duration pulse_length_step = 1dt;
const int pulse_length_num_steps = 100;

for int i in [1:pulse_length_num_steps] {
    duration pulse_length = pulse_length_start + (i-1)*pulse_length_step);
    duration sigma = pulse_length / 4;
    // since we are manipulating pulse lengths it is easier to define and play the waveform in a `cal` block
    cal {
        waveform wf = gaussian(0.5, pulse_length, sigma);
        // assume frame can be linked from a vendor supplied `cal` block
        play(driveframe, wf);
    }
    measure $0;
}