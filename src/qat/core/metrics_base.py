# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from compiler_config.config import MetricsType
from pydantic import BaseModel, Field, ValidationError, model_validator

snake_cased_flags = [
    val.snake_case_name()
    for val in list(dict(MetricsType.__members__).values())
    if not val.is_composite()
]


class MetricsManager(BaseModel):
    """Stores useful intermediary metrics that are generated during compilation, such as an
    optimised circuit or the instruction count.

    Pydantic version based on :mod:`qat.purr.compiler.metrics` elements.
    """

    enabled_metrics: MetricsType | None = Field(
        default=MetricsType.Experimental, repr=False, exclude=True
    )

    optimized_circuit: str | None = Field(default=None)
    optimized_instruction_count: int | None = Field(default=None)
    physical_qubit_indices: list[int] | None = Field(default=None)

    @model_validator(mode="before")
    def validate_all_fields_exist(cls, value):
        """Validate that all expected MetricsType flags have defined fields."""
        missing_fields = [val for val in snake_cased_flags if val not in cls.model_fields]
        if any(missing_fields):
            raise ValidationError(
                f"MetricsManager is missing type hints for {', '.join(missing_fields)}."
            )
        return value

    def __init__(self, enabled_metrics: MetricsType = MetricsType.Experimental, **kwargs):
        """
        :param enabled_metrics: Which metrics to enable? Default,
            :attr:`MetricsType.Experimental.`"""

        super().__init__(enabled_metrics=enabled_metrics, **kwargs)

    def enable(self, enabled_metrics: MetricsType, overwrite: bool = False):
        """Enable these sets of metrics for collection.

        :param enabled_metrics: Which metrics to enable? Default,
            :attr:`MetricsType.Experimental`.
        :param overwrite: If true, overwrite these metrics with a new collection.
            If false, just enable them in the current collection. Default, true.
        """
        if enabled_metrics is None:
            return

        if overwrite:
            self.enabled_metrics = enabled_metrics
        else:
            self.enabled_metrics = self.enabled_metrics | enabled_metrics

    def enable_metrics(
        self,
        enabled_metrics: MetricsType = MetricsType.Experimental,
        overwrite: bool = True,
    ):
        """Enables the set of metrics in the current collection.

        :param enabled_metrics: Which metrics to enable? Default,
            :attr:`MetricsType.Experimental`.
        :param overwrite: If true, overwrite these metrics with a new collection.
            If false, just enable them in the current collection. Default, true.
        """
        if enabled_metrics is None:
            # log.warning("Attempted to enable metrics with no value. Defaulting.")
            enabled_metrics = MetricsType.Experimental

        # log.info(f"Enabling metrics with flags: {str(enabled_metrics)}")
        self.enable(enabled_metrics, overwrite)

    def are_enabled(self, metric: MetricsType):
        return self.enabled_metrics is not None and metric in self.enabled_metrics

    def record_metric(self, metric: MetricsType, value):
        if not self.are_enabled(metric):
            return

        setattr(self, metric.snake_case_name(), value)

    def get_metric(self, metric: MetricsType):
        return getattr(self, metric.snake_case_name())

    def as_dict(self):
        """Generates a dictionary of the valid metrics."""
        return self.model_dump()

    def merge(self, other: "MetricsManager"):
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Argument other must be of type MetricsManager, not of type: {type(other)}"
            )
        for metric in MetricsType:
            if not metric.is_composite() and self.get_metric(metric) is None:
                self.record_metric(metric, other.get_metric(metric))

        return self
