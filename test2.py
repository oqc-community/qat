from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware

hw = get_default_RTCS_hardware(2)
hw.save_calibration_to_file("test2.json")

builder = hw.create_builder()

q0 = hw.get_qubit(0)
q1 = hw.get_qubit(1)


# builder.X(q0).X(q1).Z(q0, -0.3).Z(q1, 0.1)
# engine = hw.create_engine()
# instructions = engine.create_duration_timeline(builder.instructions)
# for key, item in instructions.items():
#    print("-")
#    print(key)
#    print(item)
