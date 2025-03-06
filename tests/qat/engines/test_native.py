# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.engines.native import ConnectionMixin


class TestConnectionMixin:

    class MockConnection(ConnectionMixin):
        is_connected = False

        def __init__(self, mock_obj):
            self.mock_obj = mock_obj

        def connect(self):
            self.mock_obj.val = True
            self.is_connected = True

        def disconnect(self):
            self.mock_obj.val = False
            self.is_connected = False

    class MockObject:
        def __init__(self):
            self.val = False

    def test_disconnect_on_delete(self):
        obj = self.MockObject()
        connection = self.MockConnection(obj)
        assert obj.val == False
        connection.connect()
        assert obj.val == True
        del connection
        assert obj.val == False
