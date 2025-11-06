# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.executables import AcquireData
from qat.ir.measure import AcquireMode
from qat.runtime.aggregator import ResultsAggregator, ResultsCollection


class TestResultsAggregator:
    @pytest.mark.parametrize(
        "mode, shape",
        [
            (AcquireMode.RAW, (1000, 100)),  # (1000 repeats, 100 samples)
            (AcquireMode.RAW, (10, 1000, 100)),  # (10 sweeps, 1000 repeats, 100 samples)
            (AcquireMode.SCOPE, (100,)),  # (100 samples)
            (AcquireMode.SCOPE, (10, 100)),  # (10 sweeps, 100 samples)
            (AcquireMode.INTEGRATOR, (1000,)),  # (1000 repeats)
            (AcquireMode.INTEGRATOR, (10, 1000)),  # (10 sweeps, 1000 repeats)
            (AcquireMode.INTEGRATOR, (5, 10, 1000)),  # (5 sweeps, 10 sweeps, 1000 repeats)
        ],
    )
    @pytest.mark.parametrize("return_as_list", [True, False])
    def test_results_shape(self, mode, shape, return_as_list):
        aggregator = ResultsAggregator(mode, shape, return_as_list=return_as_list)
        if return_as_list:
            assert isinstance(aggregator.results, list)
            assert np.array(aggregator.results).shape == shape
        else:
            assert isinstance(aggregator.results, np.ndarray)
            assert aggregator.results.shape == shape

    @pytest.mark.parametrize(
        "batch_sizes",
        [
            [10000],
            [2000, 2000, 2000, 2000, 2000],
            [3000, 3000, 3000, 1000],
            [4000, 3000, 2005, 995],
        ],
    )
    def test_with_integrator_and_one_iterator(self, batch_sizes):
        """This tests the situation you'd expect to see where shots need to be broken into
        batches."""

        aggregator = ResultsAggregator(AcquireMode.INTEGRATOR, (10000,))
        for i, batch_size in enumerate(batch_sizes):
            batch = (i + 1) * np.ones((batch_size,))
            aggregator.append(batch)

        results = aggregator.results
        assert results.shape == (10000,)

        cumulative_size = 0
        for i, batch_size in enumerate(batch_sizes):
            assert np.all(
                results[cumulative_size : cumulative_size + batch_size] == (i + 1)
            )
            cumulative_size += batch_size

    @pytest.mark.parametrize(
        "batch_sizes", [[10000], [2500, 2500, 2500, 2500], [8000, 2000]]
    )
    def test_with_integrator_over_multiple_iterators_with_shot_batching(self, batch_sizes):
        """Simulates a situation with multiple iterators, e.g., two loops with shots, and
        batching of shots."""

        aggregator = ResultsAggregator(AcquireMode.INTEGRATOR, (5, 10, 10000))
        for i in range(5):
            for j in range(10):
                for k, batch_size in enumerate(batch_sizes):
                    batch = (i + j / 10 + k / 100) * np.ones((batch_size,))
                    aggregator.append(batch)

        results = aggregator.results
        assert results.shape == (5, 10, 10000)

        for i in range(5):
            for j in range(10):
                cumulative_size = 0
                for k, batch_size in enumerate(batch_sizes):
                    assert np.all(
                        results[i, j, cumulative_size : cumulative_size + batch_size]
                        == (i + j / 10 + k / 100)
                    )
                    cumulative_size += batch_size

    @pytest.mark.parametrize("batch_sizes", [[10], [5, 3, 2], [6, 4], [5, 5]])
    def test_with_integrator_over_multiple_iterators_with_iterator_batching(
        self, batch_sizes
    ):
        """Simulates a situation with multiple iterators, e.g., two loops with shots, and
        batching of iterators."""

        aggregator = ResultsAggregator(AcquireMode.INTEGRATOR, (5, 10, 10000))
        for i in range(5):
            for j, batch_size in enumerate(batch_sizes):
                batch = (i + j / 10) * np.ones((batch_size, 10000))
                aggregator.append(batch)

        results = aggregator.results
        assert results.shape == (5, 10, 10000)

        for i in range(5):
            cumulative_size = 0
            for j, batch_size in enumerate(batch_sizes):
                assert np.all(
                    results[i, cumulative_size : cumulative_size + batch_size, :]
                    == (i + j / 10)
                )
                cumulative_size += batch_size

    @pytest.mark.parametrize("batch_sizes", [[5], [2, 2, 1], [3, 2]])
    def test_with_integrator_on_outermost_iterator(self, batch_sizes):
        aggregator = ResultsAggregator(AcquireMode.INTEGRATOR, (5, 10, 1000))
        for i, batch_size in enumerate(batch_sizes):
            batch = (i + 1) * np.ones((batch_size, 10, 1000))
            aggregator.append(batch)

        results = aggregator.results
        assert results.shape == (5, 10, 1000)

        cumulative_size = 0
        for i, batch_size in enumerate(batch_sizes):
            assert np.all(
                results[cumulative_size : cumulative_size + batch_size, :, :] == (i + 1)
            )
            cumulative_size += batch_size

    @pytest.mark.parametrize("mode", [AcquireMode.RAW, AcquireMode.SCOPE])
    def test_wrong_data_size(self, mode):
        aggregator = ResultsAggregator(mode, (1000, 100))

        with pytest.raises(ValueError, match="Expected readout length"):
            aggregator.append(np.ones((50)))
        with pytest.raises(ValueError, match="Expected readout length"):
            aggregator.append(np.ones((500, 50)))

    @pytest.mark.parametrize("mode", [AcquireMode.RAW, AcquireMode.SCOPE])
    @pytest.mark.parametrize(
        "batch_sizes",
        [[1000], [200, 200, 200, 200, 200], [300, 300, 300, 100], [400, 300, 205, 95]],
    )
    def test_with_scope_and_raw_with_batched_repeats(self, mode, batch_sizes):
        aggregator = ResultsAggregator(mode, (1000, 254))
        for i, batch_size in enumerate(batch_sizes):
            batch = (i + 1) * np.ones((batch_size, 254))
            aggregator.append(batch)

        results = aggregator.results
        assert results.shape == (1000, 254)

        cumulative_size = 0
        for i, batch_size in enumerate(batch_sizes):
            assert np.all(
                results[cumulative_size : cumulative_size + batch_size, :] == (i + 1)
            )
            cumulative_size += batch_size

    @pytest.mark.parametrize("mode", [AcquireMode.RAW, AcquireMode.SCOPE])
    @pytest.mark.parametrize("batch_sizes", [[10], [5, 3, 2], [6, 4], [5, 5]])
    def test_with_scope_and_raw_with_multiple_iterators(self, mode, batch_sizes):
        aggregator = ResultsAggregator(mode, (5, 10, 1000, 254))
        for i in range(5):
            for j, batch_size in enumerate(batch_sizes):
                batch = (i + j / 10) * np.ones((batch_size, 1000, 254))
                aggregator.append(batch)

        results = aggregator.results
        assert results.shape == (5, 10, 1000, 254)

        for i in range(5):
            cumulative_size = 0
            for j, batch_size in enumerate(batch_sizes):
                assert np.all(
                    results[i, cumulative_size : cumulative_size + batch_size, :, :]
                    == (i + j / 10)
                )
                cumulative_size += batch_size

    def test_overflow(self):
        aggregator = ResultsAggregator(AcquireMode.INTEGRATOR, (1000,))
        aggregator.append(np.ones((800,)))
        with pytest.raises(
            ValueError, match="Attempting to append more results than allocated"
        ):
            aggregator.append(np.ones((300,)))


class TestResultsCollection:
    def test_with_multiple_acquires(self):
        acquires = {
            "acquire_1": AcquireData(
                mode=AcquireMode.INTEGRATOR, shape=(5, 1000), physical_channel="ch1"
            ),
            "acquire_2": AcquireData(
                mode=AcquireMode.RAW, shape=(5, 1000, 254), physical_channel="ch2"
            ),
            "acquire_3": AcquireData(
                mode=AcquireMode.SCOPE, shape=(5, 128), physical_channel="ch3"
            ),
        }

        collection = ResultsCollection(acquires)

        for i in range(5):
            collection.append(
                {
                    "acquire_1": (i + 1) * np.ones((1000,)),
                    "acquire_2": (i + 1) * np.ones((1000, 254)),
                    "acquire_3": (i + 1) * np.ones((128,)),
                }
            )

        results = collection.results
        for i in range(5):
            assert np.all(results["acquire_1"][i, :] == (i + 1))
            assert np.all(results["acquire_2"][i, :, :] == (i + 1))
            assert np.all(results["acquire_3"][i, :] == (i + 1))
