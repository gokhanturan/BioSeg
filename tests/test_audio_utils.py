import unittest
import os
import tempfile
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.utils import get_audio_file_info, write_wav_safe


class TestAudioUtils(unittest.TestCase):

    def test_write_wav_safe_float(self):
        """Test writing float array to WAV"""
        y = np.array([0.5, -0.2, 0.0, 0.3, -0.1], dtype=np.float32)
        sr = 22050
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)

        write_wav_safe(path, y, sr)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)
        os.remove(path)

    def test_write_wav_safe_int16(self):
        """Test writing int16 array to WAV"""
        y = np.array([10000, -5000, 0, 20000, -10000], dtype=np.int16)
        sr = 22050
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)

        write_wav_safe(path, y, sr)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)
        os.remove(path)

    def test_get_audio_file_info_nonexistent(self):
        """Test getting info from non-existent file"""
        result = get_audio_file_info("nonexistent_file.wav")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()