# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.frontend.parsers.qasm.qasm2 import CloudQasmParser
from qat.frontend.parsers.qasm.qasm3 import Qasm3Parser


def get_qasm_parser(qasm_str: str):
    """Gets the appropriate QASM parser for the passed-in QASM string."""
    parsers = [CloudQasmParser(), Qasm3Parser()]
    attempts = []
    for parser in parsers:
        parse_attempt = parser.can_parse(qasm_str)
        if parse_attempt:
            return parser
        attempts.append((parser.parser_language().name, parse_attempt.errors))

    raise ValueError(
        "No valid parser could be found. Attempted: "
        f"{', '.join([f'{a} with error {b}' for a, b in attempts])}"
    )


def qasm_from_file(file_path):
    """Get QASM from a file."""
    with open(file_path) as ifile:
        return ifile.read()
