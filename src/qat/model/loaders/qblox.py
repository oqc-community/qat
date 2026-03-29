# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

from qblox_instruments import ClusterType

from qat.engines.qblox.dummy import DummyQbloxInstrument
from qat.engines.qblox.live import QbloxLeafInstrument

# TODO: 32Q support: COMPILER-728
_QBLOX_DUMMY_CONFIG = {
    1: ClusterType.CLUSTER_QCM,
    2: ClusterType.CLUSTER_QCM_RF,
    4: ClusterType.CLUSTER_QCM_RF,
    6: ClusterType.CLUSTER_QRC,  # slots 6-7 are reserved for QRC module
    12: ClusterType.CLUSTER_QCM_RF,
    13: ClusterType.CLUSTER_QRM,
    14: ClusterType.CLUSTER_QRM_RF,
    16: ClusterType.CLUSTER_QRM_RF,
    18: ClusterType.CLUSTER_QRM_RF,
    19: ClusterType.CLUSTER_QRC,  # slots 19-20 are reserved for QRC module
}


def create_instrument(
    id: str,
    name: str,
    address: str = None,
    dummy_config: dict = None,
) -> QbloxLeafInstrument:
    if address is None:
        return DummyQbloxInstrument(
            id=id, name=name, dummy_config=dummy_config or _QBLOX_DUMMY_CONFIG
        )
    else:
        return QbloxLeafInstrument(id=id, name=name, address=address)
