# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.frontend.base import BaseFrontend
from qat.frontend.fallthrough import FallthroughFrontend
from qat.frontend.passes.purr.transform import FlattenIR
from qat.frontend.passes.transform import PydFlattenIR
from qat.frontend.purr.purr import PurrFrontend
from qat.frontend.qasm import Qasm2Frontend, Qasm3Frontend
from qat.frontend.qat_ir import QatFrontend
from qat.frontend.qir import QIRFrontend
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.utils.hardware_model import check_type_legacy_or_pydantic


class AutoFrontend(BaseFrontend):
    """Automatically deploys a frontend to match the type of source language.

    Provided with a number of frontends, the :class:`AutoFrontend` will attempt to match
    a source program to the frontend. If successful, it will use the matched frontend to
    emit the source program as Qat IR.

    .. warning::
        The :class:`AutoFrontend` will check the source programs against the frontends in
        the order provided. If multiple frontends are capable of processing a source file,
        then the order should be strategically chosen to have the desired outcome. For
        example, if using the :class:`FallthroughFrontend`, this should be placed at the
        end of line to allow other frontends the opportunity to be matched.
    """

    def __init__(
        self,
        model: QuantumHardwareModel | PydHardwareModel | None,
        *frontends: BaseFrontend,
    ):
        """
        :param model: The hardware model is needed to instantiate the default frontends,
            defaults to None.
        :param frontends: The different frontends to check the source file against.
        """

        self.model = check_type_legacy_or_pydantic(model)
        if len(frontends) == 0:
            frontends = self.default_frontends(model)
        self.frontends = frontends

    @staticmethod
    def default_frontends(model) -> list[BaseFrontend]:
        """Returns a default list of frontends to try.

        :param model: The hardware model is needed to instantiate the default frontends.
        """
        return [Qasm2Frontend(model), Qasm3Frontend(model), QIRFrontend(model)]

    @classmethod
    def default_for_purr(cls, model: QuantumHardwareModel) -> "AutoFrontend":
        """Temporary factory for creating an AutoFrontend that allows a PurrFrontend.

        :param model: The purr hardware model is needed to instantiate the default
            frontends.
        """

        # TODO: remove when purr is fully deprecated (COMPILER-724)
        frontends = cls.default_frontends(model) + [PurrFrontend(model)]
        return cls(model, *frontends)

    @classmethod
    def default_for_pydantic(cls, model: PydHardwareModel) -> "AutoFrontend":
        """Temporary factory for creating an AutoFrontend that allows a QatFrontend.

        :param model: The hardware model is needed to instantiate the default
            frontends.
        """

        # TODO: The default_frontends methods should eventually be replaced with this.
        # (COMPILER-724)
        frontends = cls.default_frontends(model) + [QatFrontend(model)]
        return cls(model, *frontends)

    @classmethod
    def default_for_legacy(cls, model: QuantumHardwareModel) -> "AutoFrontend":
        """Temporary factory for creating an AutoFrontend that has a fallthrough frontend
        to allow any legacy source."""

        # TODO: remove when legacy is fully deprecated (COMPILER-724)
        frontends = cls.default_frontends(model) + [FallthroughFrontend(model)]
        return cls(model, *frontends)

    def check_and_return_source(self, src):
        """Checks the source file sequentially against the different frontends. If a
        suitable frontend is found, the source program is returned.

        :param src: The source program, or path to the program.
        :returns: If the program is determined to not be valid, False is returned.
            Otherwise, the program is returned (and loaded if required).
        """
        assigned_frontend = self.assign_frontend(src)
        if assigned_frontend is None:
            return False
        return assigned_frontend.check_and_return_source(src)

    def assign_frontend(self, src) -> BaseFrontend | None:
        """Determines the source language and assigns the correct frontend.

        :param src: The source program to discriminate.
        :returns: The first frontend that is able to recognise the source program. Returns
            None if no suitable frontend is found.
        """

        for frontend in self.frontends:
            if frontend.check_and_return_source(src):
                return frontend
        return None

    def emit(
        self,
        src,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ):
        """Chooses a suitable frontend and compiles a given source program.

        :param src: The source program, or path to the source program.
        :param res_mgr: The result manager to store results from passes within the pipeline,
            defaults to None.
        :param met_mgr: The metrics manager to store metrics, such as the optimized QASM
            circuit, defaults to None.
        :param compiler_config: The compiler config, defaults to None.
        :raises ValueError: An error is thrown if no suitable frontend could be found.
        """

        for frontend in self.frontends:
            if frontend.check_and_return_source(src):
                return frontend.emit(src, res_mgr, met_mgr, compiler_config, **kwargs)
        raise ValueError("No suitable frontend could be found for the source program.")


class AutoFrontendWithFlattenedIR(AutoFrontend):
    """
    An :class:`AutoFrontend` that flattens the IR after parsing.
    """

    def emit(
        self,
        src,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ):
        ir = super().emit(src, res_mgr, met_mgr, compiler_config, **kwargs)

        if self.model and isinstance(self.model, PydHardwareModel):
            return PydFlattenIR().run(ir)
        return FlattenIR().run(ir)
