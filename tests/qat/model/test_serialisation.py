from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
import pytest

from qat.model.autopopulate import AutoPopulate
from qat.model.component import Component, ComponentId, make_refdict
from qat.model.serialisation import Ref, RefDict, RefList


class At(Component):
    x: int


class Bt(Component):
    x: int
    s: str
    As: RefDict[At]
    Ds: RefList[Dt]


class Ct(Component):
    x: int
    As: RefDict[At]
    Bs: RefList[Bt]
    someB: Ref[Bt]


class Dt(Component):
    x: int
    Cs: RefDict[Ct]


class Outer(AutoPopulate):
    A: list[At]
    B: list[Bt]
    C: list[Ct]
    D: list[Dt]


def make_Outer(count=10, connections=3, seed=42):
    rng = np.random.default_rng(seed)
    pick = lambda L, size=3: make_refdict(*rng.choice(L, size=size))
    A = [At(x=rng.integers(100)) for _ in range(count)]
    B = [
        Bt(x=rng.integers(100), As=pick(A, 3), s=f"blah{rng.integers(100)}", Ds=[])
        for i in range(count)
    ]
    C = [
        Ct(
            x=rng.integers(100),
            As=pick(A, 3),
            Bs=list(pick(B, 3).values()),
            someB=list(pick(B, 1).values())[0],
        )
        for i in range(count)
    ]
    D = [Dt(x=rng.integers(100), Cs=pick(C, 3)) for i in range(count)]

    for bt in B:
        bt.Ds = list(pick(D, 3).values())

    return Outer(A=A, B=B, C=C, D=D, id="outer")


class Test_Refs:
    def test_ref_fields(self):
        class I(Component):
            s: str

        class O(Component):
            x: RefDict[I]
            y: None | int
            z: RefList[I]
            p: Ref[I]
            j: Ref[I]
            q: Optional[int]

        a = I(s="test1")
        b = I(s="test2")
        c = O(
            x=make_refdict(a, b),
            y=None,
            z=[ComponentId(), ComponentId()],
            p=ComponentId(),
            j=a,
            q=None,
        )

        expected = {
            "x": {"type": "RefDict", "populated": True},
            "z": {"type": "RefList", "populated": False},
            "p": {"type": "Ref", "populated": False},
            "j": {"type": "Ref", "populated": True},
        }

        assert c._ref_fields == expected

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_dump_load_eq(self, seed):
        O1 = make_Outer(seed=seed)
        blob = O1.model_dump()
        O2 = Outer(**blob)

        assert O1._deepequals(O2)

        O3 = make_Outer(seed=6353234234)
        assert not O1._deepequals(O3)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_dump_eq(self, seed):
        O1 = make_Outer(seed=seed)
        blob = O1.model_dump()

        O2 = Outer(**blob)
        blob2 = O2.model_dump()

        O3 = make_Outer(seed=6353234234)
        blob3 = O3.model_dump()

        assert blob == blob2
        assert blob != blob3

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_deep_equals(self, seed):
        O1 = make_Outer(seed=seed)
        O2 = deepcopy(O1)

        assert O2._deepequals(O1)
        O2.C[3].x = -1
        assert not O2._deepequals(O1)
