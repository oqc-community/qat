from qat.compiler.transform_passes import PhaseOptimisation, PostProcessingOptimisation
from qat.compiler.validation_passes import InstructionValidation, ReadoutValidation
from qat.ir.pass_base import InvokerMixin, PassManager
from qat.ir.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.runtime import QuantumRuntime


class QuantumRuntime(QuantumRuntime, InvokerMixin):
    """
    Uses the new pass infrastructure.

    Notice how polymorphic calls to XEngine.optimize() and XEngine.validate() are avoided. Instead, we have
    a flat structure of passes. This allows developers to focus on efficiently implementing a pass and easily test,
    demonstrate, and register passes without worrying too much about where it fits into the global compilation
    workflow.

    The new QuantumRuntime deliberately recognises the builder as the only acceptable form of input "IR" and refuses
    to take in a bare list of instructions. This reduces the constant confusion of "builder" vs "instructions".

    The new QuantumRuntime is also deliberately stripped out of any handling of compilation metrics. In fact, ideas
    similar to the new pass infrastructure can be applied to compilation metrics, that's why we're excluding them
    during this iteration partly because other pieces need to come together and partly because the current iteration
    needs to be kept light-weight and technically tractable.
    """

    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | PhaseOptimisation()
            | PostProcessingOptimisation()
            | InstructionValidation()
            | ReadoutValidation()
        )

    def _common_execute(
        self,
        fexecute: callable,
        builder: InstructionBuilder,
        results_format=None,
        repeats=None,
        error_mitigation=None,
    ):
        if self.engine is None:
            raise ValueError("No execution engine available.")

        if not isinstance(builder, InstructionBuilder):
            raise ValueError(
                f"Expected InstructionBuilder, but got {type(builder)} instead"
            )

        res_mgr = ResultManager()
        self.run_pass_pipeline(builder, res_mgr, self.model, self.engine)
        results = fexecute(builder)
        results = self._transform_results(results, results_format, repeats)
        return self._apply_error_mitigation(results, builder, error_mitigation)
