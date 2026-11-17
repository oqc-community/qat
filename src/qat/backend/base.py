# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

import abc
from collections import defaultdict
from typing import Generic

import networkx as nx

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.executables import Executable, Program
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class BaseBackend(Generic[Program], abc.ABC):
    """Base class for a backend that takes an intermediate representation (IR)
    :class:`QatIR` and lowers it to machine code that can be executed on a given target."""

    def __init__(self, model: None | QuantumHardwareModel):
        """
        :param model: The hardware model that holds calibrated information on the qubits on
            the QPU.
        """
        self.model = model

    @abc.abstractmethod
    def emit(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        **kwargs,
    ) -> Executable[Program]:
        """Converts an IR :class:`QatIR` to machine instructions of a given target
        architecture.

        How targets convert the IR is at their discretion but they mostly follow macro-
        expansion techniques where target instructions are selected for each instruction in
        the IR.
        """
        ...


class CustomBackend(BaseBackend[Program], Generic[Program], abc.ABC):
    """Backends may need to run pre-codegen passes on the IR :class:`QatIR` as emitted from
    the middle end.

    These passes are specified via a custom pipeline.
    """

    def __init__(self, model: QuantumHardwareModel, pipeline: PassManager = None):
        super().__init__(model=model)
        self.pipeline = pipeline or PassManager()


class AllocatingBackend(CustomBackend[Program], Generic[Program], abc.ABC):
    """A backend that's responsible for allocating FPGA card and sequencers AOT."""

    def __init__(self, model: QuantumHardwareModel, pipeline: PassManager = None):
        super().__init__(model=model, pipeline=pipeline)
        self.allocations: dict[int, dict[str, int]] = defaultdict(dict)

    def allocate(self, target: PulseChannel) -> tuple[int, int]:
        """For a given target, allocate an FPGA card and a sequencer.

        This is a greedy, first-come choice: it tracks one set of taken sequencers per
        slot but each physical channel has its own candidate set, so the outcome depends
        on the order targets arrive in and it can fail when a valid assignment exists.
        Prefer :meth:`allocate_bipartite`, which allocates all targets together.

        :param target: The pulse channel to allocate a sequencer for.
        :returns: The sequencer index and the slot index.
        :raises ValueError: If the physical channel has no sequencer left.
        """

        slot_idx = target.physical_channel.slot_idx
        target2seq = self.allocations[slot_idx]
        if (seq_idx := target2seq.get(target.full_id(), None)) is None:
            total = set(target.physical_channel.config.sequencers.keys())
            available = total - set(target2seq.values())
            if not available:
                raise ValueError(f"""
                No more available sequencers for {target}
                Physical channel: {target.physical_channel}
                Total available: {total}
                Current allocations: {target2seq}
                """)
            seq_idx = next(iter(available))
            target2seq[target.full_id()] = seq_idx
        return seq_idx, slot_idx

    def allocate_bipartite(self, targets: list[PulseChannel]) -> dict[int, dict[str, int]]:
        """Allocate sequencers to targets using a bipartite graph matching algorithm.

        All targets are matched together, respecting each physical channel's own candidate
        sequencers, so an assignment is found whenever one exists. Targets and candidates
        are sorted first, so the result does not depend on the order of ``targets``.

        :param targets: Pulse channels that need a sequencer.
        :returns: Mapping of slot index to ``{target_id: sequencer_index}``.
        :raises ValueError: If any target cannot be given a sequencer; the message names
            every target left unplaced.
        """
        targets = sorted(targets, key=lambda target: target.full_id())
        # NetworkX turns ``top_nodes`` into a set, and string hashes vary with
        # PYTHONHASHSEED. Integer labels iterate in a fixed order, so the matching is
        # reproducible across processes.
        G = nx.Graph()
        for index, target in enumerate(targets):
            G.add_node(index, bipartite=0)
            slot_idx = target.physical_channel.slot_idx
            for sequencer_id in sorted(target.physical_channel.config.sequencers):
                G.add_edge(index, (slot_idx, sequencer_id))
        matching = nx.bipartite.maximum_matching(G, top_nodes=range(len(targets)))

        unplaced = [target for i, target in enumerate(targets) if i not in matching]
        if unplaced:
            details = "\n".join(
                f"  {target.full_id()} (slot {target.physical_channel.slot_idx}, "
                f"candidate sequencers {sorted(target.physical_channel.config.sequencers)})"
                for target in unplaced
            )
            raise ValueError(
                f"Could not allocate a sequencer to {len(unplaced)} of {len(targets)} "
                f"target(s); no assignment gives every target a distinct sequencer from "
                f"its candidates:\n{details}"
            )

        allocations: dict[int, dict[str, int]] = defaultdict(dict)
        for index, target in enumerate(targets):
            slot_idx, sequencer_id = matching[index]
            allocations[slot_idx][target.full_id()] = sequencer_id
        return allocations
