# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.hardware.qblox.models import (
    PortReference,
    QbloxModuleKind,
    QbloxModuleLocation,
)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("QCM", QbloxModuleKind.qcm),
        ("qcm-rf", QbloxModuleKind.qcm_rf),
        ("QRM_RF", QbloxModuleKind.qrm_rf),
        ("QRC", QbloxModuleKind.qrc),
    ],
)
def test_module_kind_official_parsing(name, kind):
    assert QbloxModuleKind.from_qblox_name(name) is kind
    assert QbloxModuleKind.from_qblox_identifier(f"A-CH-{name}-2") is kind


def test_module_kind_values():
    assert [kind.value for kind in QbloxModuleKind] == [
        "qcm",
        "qcm_rf",
        "qrm",
        "qrm_rf",
        "qrc",
    ]


def test_module_kind_identifier_matching_uses_complete_tokens():
    assert QbloxModuleKind.from_qblox_identifier("A-CH-NOTQCM-2") is None
    assert QbloxModuleKind.from_qblox_identifier(None) is None


def test_module_kind_name_requires_string():
    with pytest.raises(ValueError, match="must be a string"):
        QbloxModuleKind.from_qblox_name(1)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: QbloxModuleLocation("", 1), "instrument id"),
        (lambda: QbloxModuleLocation(1, 1), "instrument id"),
        (lambda: QbloxModuleLocation("cluster", True), "module slot"),
        (lambda: QbloxModuleLocation("cluster", "1"), "module slot"),
        (lambda: QbloxModuleLocation("cluster", 0), "module slot"),
        (
            lambda: PortReference(
                kind=QbloxModuleKind.qcm,
                module_location=QbloxModuleLocation("cluster", 1),
                oscillator_id="",
            ),
            "oscillator id",
        ),
    ],
)
def test_qblox_references_reject_invalid_identity(factory, expected):
    with pytest.raises(ValueError, match=expected):
        factory()
