class Constants:
    MAX_GAIN_SIZE = pow(2, 16) - 1
    """Max size of gain in Q1ASM programs."""
    MAX_OFFSET_SIZE = pow(2, 16) - 1
    """Max size of offset in Q1ASM programs."""
    MAX_WAIT_TIME = pow(2, 16) - 4
    """Max size of wait instruction immediates in Q1ASM programs. Max value allowed by
    assembler is 2**16-1, but this is the largest that is a multiple of 4 ns."""
    REGISTER_SIZE = pow(2, 32) - 1
    """Size of registers in Q1ASM programs."""
    NCO_MIN_FREQ = -500e6
    """Minimum NCO frequency"""
    NCO_MAX_FREQ = 500e6
    """Maximum NCO frequency"""
    NCO_PHASE_STEPS_PER_DEG = 1e9 / 360
    """The number of steps per degree for NCO phase instructions arguments."""
    NCO_FREQ_STEPS_PER_HZ = 4.0
    """The number of steps per Hz for the NCO set_freq instruction."""
    NCO_FREQ_LIMIT_STEPS = 2e9
    """The maximum and minimum frequency expressed in steps for the NCO set_freq instruction.
    For the minimum we multiply by -1."""
    NUMBER_OF_SEQUENCERS_QCM = 6
    """Number of sequencers supported by a QCM/QCM-RF in the latest firmware."""
    NUMBER_OF_SEQUENCERS_QRM = 6
    """Number of sequencers supported by a QRM/QRM-RF in the latest firmware."""
    NUMBER_OF_REGISTERS: int = 64
    """Number of registers available in the Qblox sequencers."""
    MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS: int = 16384
    """Maximal amount of scope trace acquisition datapoints returned."""
    MAX_SAMPLE_SIZE_WAVEFORMS: int = 16384
    """Maximal amount of samples in the waveforms to be uploaded to a sequencer."""
    GRID_TIME = 4  # ns
    """
    Clock period of the sequencers. All time intervals used must be multiples of this value.
    """
    LOOP_UNROLL_THRESHOLD = 4
    """
    Size above which loops have tolerable overhead
    """
    MIN_QCM_OFFSET_V = -2.5
    """
    Minimum offset for QCM
    """
    MAX_QCM_OFFSET_V = 2.5
    """
    Maximum offset for QCM
    """
    MIN_QCM_RF_OFFSET_MV = -84
    """
    Minimum offset for QCM-RF
    """
    MAX_QCM_RF_OFFSET_MV = 73
    """
    Maximum offset for QCM-RF
    """
    MIN_QRM_OFFSET_V = -0.09
    """
    Minimum offset for QRM
    """
    MAX_QRM_OFFSET_V = 0.09
    """
    Maximum offset for QRM
    """
    MIN_QRM_RF_OFFSET_V = -0.09
    """
    Minimum offset for QRM-RF
    """
    MAX_QRM_RF_OFFSET_V = 0.09
    """
    Maximum offset for QRM-RF
    """
