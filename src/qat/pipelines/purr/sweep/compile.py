# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.executables import Executable
from qat.model.target_data import AbstractTargetData
from qat.pipelines.base import AbstractPipeline
from qat.pipelines.pipeline import CompilePipeline
from qat.pipelines.updateable import UpdateablePipeline
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel

from .flattener import SweepFlattener
from .passes import FrequencyAssignSanitisation


class CompileSweepPipeline(AbstractPipeline):
    """A pipeline that shims support for compiling instruction builders with sweeps and
    devices assigns by making repeated calls to an underlying pipeline.

    The underlying pipeline must be updateable. This is done because applying device assigns
    can change properties of the hardware model, and the pipeline must be rebuilt to ensure
    it is not invalidated. Furthermore, even if device assigns aren't used, sweeps are
    often done side-by-side with changes to the hardware model, so it is useful to ensure
    its rebuilt as a precaution.

    Eventually the risks associated with device assigns can be mitigated by mapping device
    assigns onto instructions in the IR, where possible. For example, a device assign that
    changes the frequency of a channel can be mapped onto a frequency set instruction.
    Similar behaviour could implemented for pulse channel scales too, with an e.g.
    "set scale" instruction, or appropriate mapping onto pulse amplitudes.
    """

    def __init__(
        self,
        base_pipeline: UpdateablePipeline,
        preprocessing_pipeline: PassManager | None = None,
    ):
        """
        :param base_pipeline: The underlying pipeline that is used to compile each
            instance of the sweep.
        :param preprocessing_pipeline: An optional preprocessing pipeline that runs on the
            IR before sweeps are flattened out.
        """
        if not base_pipeline.is_subtype_of(UpdateablePipeline):
            raise TypeError(
                "The base pipeline must be an UpdateablePipeline to support dynamic "
                "hardware models."
            )
        if not base_pipeline.is_subtype_of(CompilePipeline):
            raise TypeError("CompileSweepPipeline can only wrap CompilePipelines.")

        self._base_pipeline = base_pipeline
        self._preprocessing_pipeline = (
            preprocessing_pipeline
            if preprocessing_pipeline is not None
            else self._build_preprocessing_pipeline()
        )

    def compile(
        self,
        program: str | InstructionBuilder,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ):
        """Compile a builder with sweeps and device assigns against the base pipeline.

        :param program: The program to compile, which might include an instruction builder
            with sweeps.
        :param compiler_config: Optional compiler configuration to use for this compile
            call. If not provided, the compiler configuration from the pipeline will be
            used.
        :return: A batched executable containing all instances of the sweeps, along with
            the metrics from the final compile call.
        """
        if not isinstance(program, InstructionBuilder):
            return self._base_pipeline.compile(program, compiler_config)

        # done as precaution, since external forces might change the model between compile
        # calls
        self._rebuild_pipeline()
        self._run_preprocessing_pipeline(program)

        flattener = SweepFlattener(program)
        sweep_shape = flattener.sweep_sizes
        sweep_instances = flattener.create_flattened_builders()

        executables = []
        for sweep_instance in sweep_instances:
            if sweep_instance.has_device_assigns:
                executable, metrics = self._compile_with_device_assigns(
                    sweep_instance.builder,
                    sweep_instance.device_assigns,
                    compiler_config,
                    **kwargs,
                )
            else:
                executable, metrics = self._base_pipeline.compile(
                    sweep_instance.builder, compiler_config, **kwargs
                )
            executables.append(executable)

        executable = self._combine_executables(executables, sweep_shape)
        return executable, metrics

    def is_subtype_of(self, cls):
        return isinstance(self, cls) or self._base_pipeline.is_subtype_of(cls)

    @property
    def name(self) -> str:
        return self._base_pipeline.name

    @property
    def model(self) -> QuantumHardwareModel:
        return self._base_pipeline.model

    @property
    def target_data(self) -> AbstractTargetData:
        return self._base_pipeline.target_data

    @staticmethod
    def _build_preprocessing_pipeline() -> PassManager:
        """Builds a preprocessing pipeline that runs on the IR before sweeps are flattened
        out, such as resolving device assigns onto other instructions. Currently only
        sanitises frequencies, but more passes can be added as needed."""

        return PassManager() | FrequencyAssignSanitisation()

    def _run_preprocessing_pipeline(self, ir: InstructionBuilder) -> InstructionBuilder:
        """Runs the preprocessing pipeline on the IR, if one is set."""
        return self._preprocessing_pipeline.run(ir, ResultManager(), MetricsManager())

    def _rebuild_pipeline(self):
        """Rebuilds the underlying pipelines with the same model to ensure any changes to
        model do not invalidate the pipeline."""
        self._base_pipeline.update(model=self.model)

    def _compile_with_device_assigns(
        self,
        builder: InstructionBuilder,
        device_assigns,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ) -> tuple[Executable, MetricsManager]:
        """Makes required changes to the hardware model using the device assigns,
        compiles the builder, then restores the hardware model to its original state."""
        with device_assigns.apply():
            self._rebuild_pipeline()
            return self._base_pipeline.compile(builder, compiler_config, **kwargs)

    def _combine_executables(self, executables: list[Executable], sweep_shape: tuple[int]):
        """Combines the programs within the executables into a single executable."""

        programs = []
        for executable in executables:
            programs.extend(executable.programs)

        acquire_data = executables[0].acquires
        for acquire in acquire_data.values():
            # This is done intentionally: legacy qat returns with an extra dimension (list)
            # if sweeps aren't used... so its like [[result1, result2, ...]]
            if len(sweep_shape) == 0:
                sweep_shape = (1,)
            acquire.shape = tuple(list(sweep_shape) + list(acquire.shape))

        return Executable(
            programs=programs,
            acquires=acquire_data,
            assigns=executables[0].assigns,
            returns=executables[0].returns,
            calibration_id=executables[0].calibration_id,
        )
