from pathlib import Path

benchmarks_path = Path(__file__).parent.parent


# QASM2 Benchmarks
def load_qasm(qasm_string):
    path = Path(qasm_string)
    if not path.is_file() and not path.is_absolute():
        path = benchmarks_path.joinpath("qasm", path)
    with path.with_suffix(".qasm").open("r") as f:
        return f.read()
