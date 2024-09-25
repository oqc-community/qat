# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from typing import List, Optional

from compiler_config.config import MetricsType

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class _FlagFieldValidation(type):
    """
    Validate that the CompilationsMetrics class has type hints for all the metrics,
    which'll be dynamically assigned later.
    """

    def __new__(mcs, name, inheritance, attributes):
        hint_names = set(attributes.get("__annotations__", {}).keys())
        snake_cased_flags = [
            val.snake_case_name()
            for val in list(dict(MetricsType.__members__).values())
            if not val.is_composite()
        ]
        missing_fields = [val for val in snake_cased_flags if val not in hint_names]
        if any(missing_fields):
            raise ValueError(
                "CompilationMetrics is missing type hints for "
                f"{', '.join(missing_fields)}."
            )

        return super().__new__(mcs, name, inheritance, attributes)


class CompilationMetrics(metaclass=_FlagFieldValidation):
    """
    Container object for all metrics generated during the compilation pipeline. All
    fields are generated from the MetricsType flag after the names are snake-cased and
    hold the value associated with that particular flag.
    """

    optimized_circuit: Optional[str]
    optimized_instruction_count: Optional[int]

    def __init__(self, enabled_metrics=None):
        self.enabled_metrics: Optional[MetricsType] = enabled_metrics or MetricsType.Default
        for key in [val.snake_case_name() for val in self._target_metrics()]:
            setattr(self, key, None)

    def _target_metrics(self) -> List[MetricsType]:
        """Get a list of the enum types that we should function on."""
        return [
            val
            for val in list(dict(MetricsType.__members__).values())
            if not val.is_composite()
        ]

    def enable(self, enabled_metrics: MetricsType, overwrite=False):
        """
        Enable these sets of metrics for collection. If overwrite is True then the
        passed-in values will overwrite existing ones.
        """
        if enabled_metrics is None:
            return

        if overwrite:
            self.enabled_metrics = enabled_metrics
        else:
            self.enabled_metrics = self.enabled_metrics | enabled_metrics

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
        return {
            met.snake_case_name(): self.get_metric(met) for met in self._target_metrics()
        }

    def merge(self, other: "CompilationMetrics"):
        if not isinstance(other, type(self)):
            raise TypeError(
                "Argument other must be of type Compilation metrics, not of type: "
                f"{type(other)}"
            )
        for metric in other._target_metrics():
            if self.get_metric(metric) is None:
                self.record_metric(metric, other.get_metric(metric))


class MetricsMixin:
    def __init__(self):
        super().__init__()
        self.compilation_metrics: Optional[CompilationMetrics] = None

    def are_metrics_enabled(self, metric_type: MetricsType = None):
        """
        Do we have a metrics collection, and if so does it have any active metrics.
        """
        return self.compilation_metrics is not None and (
            metric_type is None or self.compilation_metrics.are_enabled(metric_type)
        )

    def record_metric(self, metric: MetricsType, value):
        """
        Records a metric value if the collection has this sort of metric enabled.
        """
        if self.are_metrics_enabled(metric):
            self.compilation_metrics.record_metric(metric, value)

    def enable_metrics(self, enabled_metrics=None, overwrite=True):
        """
        Enables the set of metrics in the current collection. If overwrite is set to
        true, or there are no compilation metrics it'll create a new collection, if
        overwrite is false it'll enable these metrics in the currently-active
        collection.
        """
        if enabled_metrics is None:
            log.warning("Attempted to enable metrics with no value. Defaulting.")
            enabled_metrics = MetricsType.Default

        if self.compilation_metrics is None:
            return

        log.info(f"Enabling metrics with flags: {str(enabled_metrics)}")
        self.compilation_metrics.enable(enabled_metrics, overwrite)
