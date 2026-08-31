import unittest

import numpy as np
from OneEuroFilter import OneEuroFilter

from src.utils.filter import OneEuroFilterNumpy


class OneEuroFilterNumpyTests(unittest.TestCase):
    def setUp(self):
        self.freq = 60.0
        self.mincutoff = 25.0
        self.beta = 0.64
        self.dcutoff = 1.0
        self.reference = [
            OneEuroFilter(
                self.freq,
                mincutoff=self.mincutoff,
                beta=self.beta,
                dcutoff=self.dcutoff,
            )
            for _ in range(45)
        ]
        self.actual = OneEuroFilterNumpy(
            self.freq,
            mincutoff=self.mincutoff,
            beta=self.beta,
            dcutoff=self.dcutoff,
        )

    def assert_matches_reference(self, values, timestamp):
        expected = np.array(
            [
                filter_(float(value), timestamp)
                for filter_, value in zip(self.reference, values)
            ]
        )
        actual = self.actual(values, timestamp)
        np.testing.assert_array_equal(actual, expected)

    def test_random_sequence_matches_scalar_filters_exactly(self):
        rng = np.random.default_rng(20260831)
        timestamp = 1000.0
        for _ in range(1_000):
            timestamp += float(rng.uniform(1 / 90, 1 / 30))
            values = rng.uniform(-1.5, 1.5, size=45).astype(np.float32)
            self.assert_matches_reference(values, timestamp)

    def test_reset_and_parameter_updates_match_scalar_filters(self):
        first = np.linspace(-1.0, 1.0, 45, dtype=np.float32)
        second = first[::-1].copy()
        self.assert_matches_reference(first, 1000.0)

        for filter_ in self.reference:
            filter_.reset()
            filter_.setParameters(50.0, 10.0, 0.25, 2.0)
        self.actual.reset()
        self.actual.setParameters(50.0, 10.0, 0.25, 2.0)

        self.assert_matches_reference(second, 1000.02)
        self.assert_matches_reference(first, 1000.04)

    def test_shape_change_is_rejected(self):
        self.actual(np.zeros(45, dtype=np.float32), 1000.0)

        with self.assertRaisesRegex(ValueError, 'doesn.t match'):
            self.actual(np.zeros(4, dtype=np.float32), 1000.1)


if __name__ == '__main__':
    unittest.main()
