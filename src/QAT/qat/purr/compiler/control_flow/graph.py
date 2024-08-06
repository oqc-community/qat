from typing import List

from qat.purr.compiler.control_flow.instructions import EndRepeat, EndSweep
from qat.purr.compiler.instructions import DeviceUpdate, Instruction, Repeat, Sweep


class BasicBlock:
    def __init__(self, indices=None):
        self.indices: List[int] = indices or []
        self.is_entry = False
        self.is_exit = self.is_entry  # TODO - implement properly

    def head(self):
        if self.indices:
            return self.indices[0]
        return None

    def tail(self):
        if self.indices:
            return self.indices[-1]
        return None

    def is_empty(self):
        return len(self.indices) == 0

    def iterator(self):
        return iter(self.indices)


class Flow:
    def __init__(self, src=None, dest=None):
        self.src: BasicBlock = src
        self.dest: BasicBlock = dest


class ControlFlowGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes: List[BasicBlock] = nodes or []
        self.edges: List[Flow] = edges or []
        self.has_entry = False

    @property
    def entry(self):
        return next((b for b in self.nodes if b.is_entry))

    @property
    def exit(self):
        return next((b for b in self.nodes if b.is_exit))

    def get_or_create_node(self, header: int) -> BasicBlock:
        node = next((n for n in self.nodes if n.head() == header), None)
        if not node:
            node = BasicBlock([header])
            self.nodes.append(node)

        if not self.has_entry:
            self.has_entry = True
            node.is_entry = True

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


class DfsTraversal:
    def __init__(self):
        self._entered: List[BasicBlock] = []

    def clear(self):
        self._entered.clear()

    def run(self, graph: ControlFlowGraph):
        self.clear()
        self._visit(graph.entry, graph)

    def _visit(self, node: BasicBlock, graph: ControlFlowGraph):
        self.enter(node)
        self._entered.append(node)
        for neighbour in graph.out_nbrs(node):
            if neighbour not in self._entered:
                self._visit(neighbour, graph)
        self.exit(node)

    def enter(self, node: BasicBlock):
        pass

    def exit(self, node: BasicBlock):
        pass


class EmitterMixin:
    def __init__(self, instructions: List[Instruction]):
        self.instructions = instructions

    def build_cfg(self, cfg: ControlFlowGraph):
        """
        Recursively looks (re)discovers headers and new flow information.
        """

        flow = [(e.src.head(), e.dest.head()) for e in cfg.edges]
        headers = sorted([n.head() for n in cfg.nodes]) or [
            i
            for i, inst in enumerate(self.instructions)
            if isinstance(inst, (Sweep, Repeat, EndSweep, EndRepeat))
        ]
        if headers[0] != 0:
            headers.insert(0, 0)

        next_flow = set(flow)
        next_headers = set(headers)
        for i, h in enumerate(headers):
            inst_at_h = self.instructions[h]
            src = cfg.get_or_create_node(h)
            if isinstance(inst_at_h, Repeat):
                next_headers.add(h + 1)
                next_flow.add((h, h + 1))
                dest = cfg.get_or_create_node(h + 1)
                cfg.get_or_create_edge(src, dest)
            elif isinstance(inst_at_h, Sweep):
                s = next(
                    (
                        s
                        for s, inst in enumerate(self.instructions[h + 1 :])
                        if not isinstance(inst, DeviceUpdate)
                    )
                )
                next_headers.add(s + h + 1)
                next_flow.add((h, s + h + 1))
                src.indices.extend(range(src.tail() + 1, s + h + 1))
                dest = cfg.get_or_create_node(s + h + 1)
                cfg.get_or_create_edge(src, dest)
            elif isinstance(inst_at_h, (EndSweep, EndRepeat)):
                if h < len(self.instructions) - 1:
                    next_headers.add(h + 1)
                    next_flow.add((h, h + 1))
                    dest = cfg.get_or_create_node(h + 1)
                    cfg.get_or_create_edge(src, dest)
                type = Sweep if isinstance(inst_at_h, EndSweep) else Repeat
                p = next(
                    (p for p in headers[i::-1] if isinstance(self.instructions[p], type))
                )
                next_headers.add(p)
                next_flow.add((h, p))
                dest = cfg.get_or_create_node(p)
                cfg.get_or_create_edge(src, dest)
            else:
                k = next(
                    (
                        s
                        for s, inst in enumerate(self.instructions[h + 1 :])
                        if isinstance(inst, (Sweep, Repeat, EndSweep, EndRepeat))
                    ),
                    None,
                )
                if k:
                    next_headers.add(k + h + 1)
                    next_flow.add((h, k + h + 1))
                    src.indices.extend(range(src.tail() + 1, k + h + 1))
                    dest = cfg.get_or_create_node(k + h + 1)
                    cfg.get_or_create_edge(src, dest)

        if next_headers == set(headers) and next_flow == set(flow):
            return

        self.build_cfg(cfg)

    def emit_cfg(self) -> ControlFlowGraph:
        cfg = ControlFlowGraph()
        self.build_cfg(cfg)
        return cfg
