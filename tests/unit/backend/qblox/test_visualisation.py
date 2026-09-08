# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from types import SimpleNamespace

import numpy as np
import pytest

from qat.backend.qblox.visualisation import plot_playback, plot_program


@pytest.mark.parametrize(
    "packages",
    [
        pytest.param({}, id="no-packages"),
        pytest.param(
            {"empty": SimpleNamespace(timeline=np.array([]))},
            id="empty-timeline",
        ),
    ],
)
def test_plot_program_ignores_empty_programs(mocker, packages):
    show = mocker.patch("qat.backend.qblox.visualisation.plt.show")

    plot_program(SimpleNamespace(packages=packages))

    show.assert_not_called()


def test_plot_program_pads_and_plots_each_timeline(mocker):
    short_package = SimpleNamespace(
        timeline=np.array([1 + 2j]),
        pulse_channel_id="q0.drive",
    )
    long_package = SimpleNamespace(
        timeline=np.array([3 + 4j, 5 + 6j]),
        pulse_channel_id="q1.drive",
    )
    figure = mocker.Mock()
    axes = np.array([[mocker.Mock()], [mocker.Mock()]])
    subplots = mocker.patch(
        "qat.backend.qblox.visualisation.plt.subplots",
        return_value=(figure, axes),
    )
    tight_layout = mocker.patch("qat.backend.qblox.visualisation.plt.tight_layout")
    show = mocker.patch("qat.backend.qblox.visualisation.plt.show")

    plot_program(
        SimpleNamespace(
            packages={
                "q0.drive": short_package,
                "q1.drive": long_package,
            }
        )
    )

    np.testing.assert_array_equal(short_package.timeline, np.array([1 + 2j, 0]))
    subplots.assert_called_once_with(
        nrows=2,
        ncols=1,
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=(10, 5),
    )
    figure.suptitle.assert_called_once_with("Timeline plots")
    assert axes[0, 0].plot.call_count == 2
    assert axes[1, 0].plot.call_count == 2
    axes[0, 0].set_title.assert_called_once_with("q0.drive")
    axes[1, 0].set_title.assert_called_once_with("q1.drive")
    tight_layout.assert_called_once_with()
    show.assert_called_once_with()


def test_plot_playback_ignores_empty_results(mocker):
    show = mocker.patch("qat.backend.qblox.visualisation.plt.show")

    plot_playback({})

    show.assert_not_called()


def test_plot_playback_plots_scope_integration_and_threshold_data(mocker):
    acquisition = SimpleNamespace(
        name="readout",
        acquisition=SimpleNamespace(
            scope=SimpleNamespace(
                path0=SimpleNamespace(data=np.array([1.0, 2.0])),
                path1=SimpleNamespace(data=np.array([3.0, 4.0])),
            ),
            bins=SimpleNamespace(
                integration=SimpleNamespace(
                    path0=np.array([5.0]),
                    path1=np.array([6.0]),
                ),
                threshold=np.array([1]),
            ),
        ),
    )
    figure = mocker.Mock()
    axes = np.array([[mocker.Mock()], [mocker.Mock()], [mocker.Mock()]])
    subplots = mocker.patch(
        "qat.backend.qblox.visualisation.plt.subplots",
        return_value=(figure, axes),
    )
    tight_layout = mocker.patch("qat.backend.qblox.visualisation.plt.tight_layout")
    show = mocker.patch("qat.backend.qblox.visualisation.plt.show")

    plot_playback({"q0.readout": [acquisition]})

    subplots.assert_called_once_with(
        nrows=3,
        ncols=1,
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=(10, 5),
    )
    figure.suptitle.assert_called_once_with("Playback plots for readout on q0.readout")
    assert axes[0, 0].plot.call_count == 2
    assert axes[1, 0].plot.call_count == 2
    axes[2, 0].plot.assert_called_once_with(np.array([1]), label="I")
    axes[0, 0].title.set_text.assert_called_once_with("Scope acquisition")
    axes[1, 0].title.set_text.assert_called_once_with("Integrated acquisition")
    axes[2, 0].title.set_text.assert_called_once_with("Thresholded acquisition")
    tight_layout.assert_called_once_with()
    show.assert_called_once_with()
