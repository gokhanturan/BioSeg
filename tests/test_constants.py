import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.constants import (
    TARGET_SAMPLE_RATE, N_FFT, HOP_LENGTH, SPEC_MAX_TIME_COLS,
    DEFAULT_MIN_DURATION, DEFAULT_MAX_DURATION, TARGET_LOUDNESS_LUFS
)


class TestConstants(unittest.TestCase):

    def test_sample_rate(self):
        self.assertEqual(TARGET_SAMPLE_RATE, 22050)

    def test_fft_params(self):
        self.assertEqual(N_FFT, 2048)
        self.assertEqual(HOP_LENGTH, 512)

    def test_duration_limits(self):
        self.assertGreaterEqual(DEFAULT_MAX_DURATION, DEFAULT_MIN_DURATION)
        self.assertGreaterEqual(DEFAULT_MIN_DURATION, 0)

    def test_lufs_value(self):
        self.assertEqual(TARGET_LOUDNESS_LUFS, -16.0)

    def test_spectrogram_max_cols(self):
        self.assertEqual(SPEC_MAX_TIME_COLS, 4000)


if __name__ == '__main__':
    unittest.main()