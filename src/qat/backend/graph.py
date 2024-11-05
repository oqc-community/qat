from typing import List


class BasicBlock:
    def __init__(self, elements=None):
        self.elements: List = elements or []

    def head(self):
        if self.elements:
            return self.elements[0]
        return None

    def tail(self):
        if self.elements:
            return self.elements[-1]
        return None

    def is_empty(self):
        return len(self.elements) == 0

    def iterator(self):
        return iter(self.elements)

    def __repr__(self):
        return self.elements


class Flow:
    def __init__(self, src=None, dest=None):
        self.src: BasicBlock = src
        self.dest: BasicBlock = dest

    def __repr__(self):
        return (
            f"{self.src.head()} .. {self.src.tail()}",
            f"{self.dest.head()} .. {self.dest.tail()}",
        )


class ControlFlowGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes: List[BasicBlock] = nodes or []
        self.edges: List[Flow] = edges or []
        self.entry = None

    def get_or_create_node(self, header: int) -> BasicBlock:
        node = next((n for n in self.nodes if n.head() == header), None)
        if not node:
            node = BasicBlock([header])
            self.nodes.append(node)

        self.entry = self.entry or node
        return node

    def get_or_create_edge(self, src: BasicBlock, dest: BasicBlock) -> Flow:
        edge = next((f for f in self.edges if f.src == src and f.dest == dest), None)
        if not edge:
            edge = Flow(src, dest)
            self.edges.append(edge)
        return edge

    def out_nbrs(self, node) -> List[BasicBlock]:
        return [e.dest for e in self.edges if e.src == node]

    def in_nbrs(self, node) -> List[BasicBlock]:
        return [e.src for e in self.edges if e.dest == node]

    def out_edges(self, node) -> List[Flow]:
        return [e for e in self.edges if e.src == node]

    def in_edges(self, node) -> List[Flow]:
        return [e for e in self.edges if e.dest == node]
