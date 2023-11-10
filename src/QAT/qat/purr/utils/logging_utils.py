# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import sys
import time

from qat.purr.compiler.metrics import MetricsMixin
from qat.purr.utils.logger import LoggerLevel, get_default_logger

log = get_default_logger()


class _LogContextManager(MetricsMixin):
    def __init__(
        self,
        message: str,
        metric_type=None,
        metrics_collection=None,
        level=LoggerLevel.INFO,
    ):
        super().__init__()
        if "{" not in message:
            raise ValueError("Need one matching {} to inject duration time into.")

        self.message = message
        self.level: int = level.value
        self.metric_type = metric_type
        self.start_time = None
        self.compilation_metrics = metrics_collection

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args, **kwargs):
        # Arbitrarily limiting to 6 decimal places as greater precision
        # shouldn't be necessary.
        total_time = round(time.time() - self.start_time, 6)

        if total_time > sys.float_info.epsilon:
            log.log(self.level, self.message.format(total_time), stacklevel=1)

        if self.metric_type is not None:
            self.record_metric(self.metric_type, total_time)


def log_duration(
    message, metric_type=None, metric_collection=None, log_level: LoggerLevel = None
):
    if metric_type is not None and metric_collection is None:
        raise ValueError("Can't have a metric to record but no collection.")

    return _LogContextManager(
        message, metric_type, metric_collection, level=log_level or LoggerLevel.INFO
    )
