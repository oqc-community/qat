defcalgrammar "openpulse";

// sweep parameters would be programmed in by some higher level bindings
const float frequency_start = 4.5e9;
const float frequency_step = 1e6
const int frequency_num_steps = 301;

// define a long saturation pulse of a set duration and amplitude
defcal saturation_pulse $0 {
    // assume frame can be linked from a vendor supplied `cal` block
    play(driveframe, constant(0.1, 100e-6));
}

// step into a `cal` block to set the start of the frequency sweep
cal {
    set_frequency(driveframe, frequency_start);
}

for i in [1:frequency_num_steps] {
    // step into a `cal` block to adjust the pulse frequency via the frame frequency
    cal {
        shift_frequency(driveframe, frequency_step);
    }

    saturation_pulse $0;
    measure $0;
}