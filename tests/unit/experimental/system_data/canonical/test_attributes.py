# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from qat.experimental.system_data.canonical.attributes import get_attribute_value
from qat.experimental.system_data.canonical.schema import AttributeEntry


def test_get_attribute_value_returns_none_for_absent_key():
    """_get_attribute_value returns None when the key is not present in attributes."""
    attrs = (AttributeEntry(key="duration", value=500),)
    assert get_attribute_value(attrs, "nonexistent") is None


def test_get_attribute_value_returns_entry_for_present_key():
    """_get_attribute_value returns the correct AttributeEntry when the key is present."""
    attrs = (AttributeEntry(key="duration", value=500),)
    assert get_attribute_value(attrs, "duration") == attrs[0]


def test_get_attribute_value_returns_explicit_none():
    """_get_attribute_value returns the correct AttributeEntry when the key is present with
    None value."""
    attrs = (AttributeEntry(key="optional", value=None),)
    assert get_attribute_value(attrs, "optional") == attrs[0]
