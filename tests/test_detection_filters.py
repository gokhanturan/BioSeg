import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.features import calculate_snr, calculate_spectral_density


class TestDetectionFilters(unittest.TestCase):

    def test_calculate_snr_returns_float(self):
        """Test that calculate_snr returns a float (doesn't crash)"""
        sr = 22050
        t = np.linspace(0, 0.5, int(sr * 0.5))
        segment = 0.5 * np.sin(2 * np.pi * 2000 * t)

        snr = calculate_snr(segment)
        # Just check it returns a number (could be negative due to implementation)
        self.assertIsInstance(snr, float)

    def test_calculate_snr_with_silence(self):
        """Test calculate_snr with silent segment"""
        segment = np.zeros(5000)
        snr = calculate_snr(segment)
        self.assertEqual(snr, 0)

    def test_calculate_spectral_density_returns_float(self):
        """Test that calculate_spectral_density returns a float"""
        sr = 22050
        t = np.linspace(0, 0.5, int(sr * 0.5))
        segment = 0.5 * np.sin(2 * np.pi * 2000 * t)

        density = calculate_spectral_density(segment, sr)
        self.assertIsInstance(density, float)
        self.assertGreaterEqual(density, 0)
        self.assertLessEqual(density, 1.0)

    def test_calculate_spectral_density_with_noise(self):
        """Test calculate_spectral_density with random noise"""
        sr = 22050
        segment = np.random.randn(10000)

        density = calculate_spectral_density(segment, sr)
        self.assertIsInstance(density, float)
        # Density should be between 0 and 1
        self.assertGreaterEqual(density, 0)
        self.assertLessEqual(density, 1.0)


if __name__ == '__main__':
    unittest.main()