# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pathlib import Path

import pytest

from qat.engines.qblox.live import QbloxCompositeInstrument, QbloxLeafInstrument
from qat.instrument.base import (
    CompositeInstrument,
    CsvInstrumentBuilder,
    LeafInstrument,
)


@pytest.mark.parametrize(
    "cinstr_type, linstr_type",
    [
        (None, None),
        (CompositeInstrument, LeafInstrument),
        (QbloxCompositeInstrument, QbloxLeafInstrument),
    ],
)
def test_instrument_csv_builder(testpath, cinstr_type, linstr_type):
    filepath = Path(
        testpath,
        "files",
        "config",
        "instrument_info.csv",
    )

    composite = CsvInstrumentBuilder(filepath, cinstr_type, linstr_type).build()
    if cinstr_type in [None, CompositeInstrument]:
        assert isinstance(composite, CompositeInstrument)
        assert all(
            isinstance(comp, LeafInstrument) for comp in composite.components.values()
        )
    else:
        assert isinstance(composite, QbloxCompositeInstrument)
        assert all(
            isinstance(comp, QbloxLeafInstrument) for comp in composite.components.values()
        )
        assert all(comp.ref_source == "internal" for comp in composite.components.values())
    assert len(composite.components) == 8
