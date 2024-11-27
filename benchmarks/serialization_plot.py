# Need to run first:
# poetry run pytest --benchmark-only --benchmark-save "benchmarks" "benchmarks/serialization.py"

from json import loads

import matplotlib.pyplot as plt
import numpy as np

with open(".benchmarks/Linux-CPython-3.10-64bit/0001_benchmarks.json", "r") as f:
    data = loads(f.read())["benchmarks"]

keys = ["2qb_random_cnot", "10qb_random_cnot", "bell_state", "10qb_ghz", "2qb_clifford"]
ser = {}
deser = {}
ser_conv = {}
deser_conv = {}
for key in keys:
    stats = {}
    for bm in data:
        if bm["name"] == f"test_benchmarks_legacy_serialize[{key}]":
            stats["legacy_serialize"] = bm["stats"]["mean"]
        elif bm["name"] == f"test_benchmarks_legacy_deserialize[{key}]":
            stats["legacy_deserialize"] = bm["stats"]["mean"]
        elif bm["name"] == f"test_benchmarks_pydantic_serialize[{key}]":
            stats["pydantic_serialize"] = bm["stats"]["mean"]
        elif bm["name"] == f"test_benchmarks_pydantic_deserialize[{key}]":
            stats["pydantic_deserialize"] = bm["stats"]["mean"]
        elif bm["name"] == f"test_benchmarks_legacy_pydantic_serialize[{key}]":
            stats["legacy_pydantic_serialize"] = bm["stats"]["mean"]
        elif bm["name"] == f"test_benchmarks_legacy_pydantic_deserialize[{key}]":
            stats["legacy_pydantic_deserialize"] = bm["stats"]["mean"]

    ser[key] = stats["legacy_serialize"] / stats["pydantic_serialize"]
    deser[key] = stats["legacy_deserialize"] / stats["pydantic_deserialize"]
    ser_conv[key] = stats["legacy_serialize"] / stats["legacy_pydantic_serialize"]
    deser_conv[key] = stats["legacy_deserialize"] / stats["legacy_pydantic_deserialize"]


# make a plot which compares everything to legacy
xs = 3 * np.arange(len(keys))
plt.bar(xs - 0.6, [ser[key] for key in keys], 0.4, label="Serialize")
plt.bar(xs - 0.2, [deser[key] for key in keys], 0.4, label="Deserialize")
plt.bar(xs + 0.2, [ser_conv[key] for key in keys], 0.4, label="Convert + Serialize")
plt.bar(xs + 0.6, [deser_conv[key] for key in keys], 0.4, label="Convert + Deserialize")
plt.plot(
    [xs[0] - 1, xs[-1] + 1],
    [1, 1],
    color="black",
)
plt.xlabel("Circuits")
plt.ylabel("Speed-up")
plt.legend()
plt.xticks(xs, keys, rotation=90)
plt.title("Serialization speed-up with pydantic")
plt.savefig("Serialization.pdf", bbox_inches="tight")
plt.show()

# a zoomed plot so we can check things are always faster
plt.bar(xs - 0.6, [ser[key] for key in keys], 0.4, label="Serialize")
plt.bar(xs - 0.2, [deser[key] for key in keys], 0.4, label="Deserialize")
plt.bar(xs + 0.2, [ser_conv[key] for key in keys], 0.4, label="Convert + Serialize")
plt.bar(xs + 0.6, [deser_conv[key] for key in keys], 0.4, label="Convert + Deserialize")
plt.plot(
    [xs[0] - 1, xs[-1] + 1],
    [1, 1],
    color="black",
)
plt.xlabel("Circuits")
plt.ylabel("Speed-up")
plt.legend()
plt.xticks(xs, keys, rotation=90)
plt.title("Serialization speed-up with pydantic")
plt.ylim([0, 10])
plt.savefig("Serialization_zoomed.pdf", bbox_inches="tight")
plt.show()
