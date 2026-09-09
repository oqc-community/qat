# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd
from pathlib import Path

import pytest

from qat.engines.qblox.live import QbloxCompositeInstrument, QbloxLeafInstrument
from qat.instrument.base import CompositeInstrument, CsvInstrumentBuilder, LeafInstrument


class StubLeafInstrument(LeafInstrument):
    def __init__(self, instrument_id: str, playback_result: dict[str, int]):
        super().__init__(instrument_id, instrument_id, "127.0.0.1")
        self.playback_result = playback_result
        self.setup_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def setup(self, *args: object, **kwargs: object) -> None:
        self.setup_calls.append((args, kwargs))

    def playback(self, *_args: object, **_kwargs: object) -> dict[str, int]:
        return self.playback_result


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


def test_composite_instrument_delegates_lifecycle_and_combines_results():
    first = StubLeafInstrument("first", {"a": 1})
    second = StubLeafInstrument("second", {"b": 2})
    composite = CompositeInstrument[StubLeafInstrument]()
    composite.add(first)
    composite.add(second)

    assert composite.components == {"first": first, "second": second}
    assert not composite.is_connected
    assert repr(composite) == f"{first!r}\n{second!r}"
    assert str(composite) == f"{first}\n{second}"

    composite.connect()
    composite.setup("program", shots=10)

    assert composite.is_connected
    assert first.setup_calls == [(("program",), {"shots": 10})]
    assert second.setup_calls == [(("program",), {"shots": 10})]
    assert composite.playback() == {"a": 1, "b": 2}

    composite.disconnect()

    assert not composite.is_connected


def test_composite_instrument_rejects_duplicate_component():
    composite = CompositeInstrument[StubLeafInstrument]()
    component = StubLeafInstrument("instrument", {})
    composite.add(component)

    with pytest.raises(ValueError, match="already exists"):
        composite.add(component)


def test_composite_instrument_rejects_conflicting_playback_results():
    composite = CompositeInstrument[StubLeafInstrument]()
    composite.add(StubLeafInstrument("first", {"result": 1}))
    composite.add(StubLeafInstrument("second", {"result": 2}))

    with pytest.raises(ValueError, match="conflicting keys"):
        composite.playback()


def test_csv_instrument_builder_rejects_missing_file(tmp_path):
    missing = tmp_path / "missing.csv"

    with pytest.raises(ValueError, match="not found"):
        CsvInstrumentBuilder(missing).build()
