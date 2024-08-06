
class PassManager:
    def __init__(self):
        self.passes: List[Pass] = []
        self.pass_results: Dict = {}

    def run(self, builder: InstructionBuilder, *args):
        for p in self.passes:
            result = p.run(builder, args)


class PassRegistry(Flag):
    EMPTY = auto()
    TARGET_COLLECTION = auto()
    SWEEP_DECOMPOSITION = auto()
    HW_LOWERABILITY = auto()
    HW_BATCHING = auto()
    SCOPE_BALANCING = auto()
