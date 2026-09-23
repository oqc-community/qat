# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.backend import graph as backend_graph
from qat.purr.backends.qblox import graph as purr_graph


@pytest.fixture(params=[backend_graph, purr_graph])
def graph_module(request):
    return request.param


def test_basic_block_exposes_elements(graph_module):
    empty = graph_module.BasicBlock()
    block = graph_module.BasicBlock([1, 2])

    assert empty.head() is None
    assert empty.tail() is None
    assert empty.is_empty()
    assert list(empty.iterator()) == []
    assert block.head() == 1
    assert block.tail() == 2
    assert not block.is_empty()
    assert list(block.iterator()) == [1, 2]
    assert graph_module.BasicBlock.__repr__(block) == [1, 2]


def test_flow_exposes_endpoint_summaries(graph_module):
    flow = graph_module.Flow(
        graph_module.BasicBlock([1, 2]),
        graph_module.BasicBlock([3, 4]),
    )

    assert graph_module.Flow.__repr__(flow) == ("1 .. 2", "3 .. 4")


def test_control_flow_graph_reuses_nodes_and_edges(graph_module):
    graph = graph_module.ControlFlowGraph()
    source = graph.get_or_create_node(1)
    destination = graph.get_or_create_node(2)

    assert graph.entry is source
    assert graph.get_or_create_node(1) is source

    edge = graph.get_or_create_edge(source, destination)

    assert graph.get_or_create_edge(source, destination) is edge
    assert graph.out_nbrs(source) == [destination]
    assert graph.in_nbrs(destination) == [source]
    assert graph.out_edges(source) == [edge]
    assert graph.in_edges(destination) == [edge]
