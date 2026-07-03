import warnings
import numpy as np
import scipy.signal as sps
from scipy.io.wavfile import read as wavread, WavFileWarning
from PyQt5.QtCore import QThread, pyqtSignal
from .utils import get_audio_file_info
from .constants import TARGET_SAMPLE_RATE, MAX_FILE_SIZE_MB


class FileLoaderThread(QThread):
    """QThread class to load audio files and resample to target rate without blocking the UI."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, float, dict)
    error = pyqtSignal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        """Reads file, converts to mono, and resamples to target rate."""
        try:
            self.progress.emit(10)
            file_info = get_audio_file_info(self.file_path)
            if not file_info:
                raise RuntimeError("Could not retrieve file information")

            # Validate file size against limit
            if file_info['file_size_mb'] > MAX_FILE_SIZE_MB:
                self.error.emit(
                    f"File too large ({file_info['file_size_mb']:.1f} MB). Use files smaller than {MAX_FILE_SIZE_MB}MB.")
                return

            # Read WAV file, ignoring WavFileWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", WavFileWarning)
                original_sr, data = wavread(self.file_path)

            # Convert stereo to mono by averaging channels
            if getattr(data, "ndim", 1) > 1:
                data = data.mean(axis=1)

            # Normalize integer data to float32 in range [-1.0, 1.0]
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max if data.dtype != np.int16 else 32767
                y = (data.astype(np.float32) / max_val)
            else:
                y = data.astype(np.float32)

            target_sr = TARGET_SAMPLE_RATE
            if int(original_sr) != target_sr:
                # Resample using polyphase filtering
                gcd = np.gcd(int(original_sr), int(target_sr))
                up, down = target_sr // gcd, int(original_sr) // gcd

                original_length_samples = len(y)
                new_length = int(original_length_samples * target_sr / original_sr)

                y_resampled = sps.resample_poly(y, up, down)

                # Trim or pad to exact target length
                if len(y_resampled) != new_length:
                    if len(y_resampled) > new_length:
                        y_resampled = y_resampled[:new_length]
                    else:
                        y_resampled = np.pad(y_resampled, (0, new_length - len(y_resampled)), mode='constant')

                y = y_resampled.astype(np.float32)
                sr = target_sr
                file_info['processed_sample_rate'] = sr
                file_info['original_sample_rate'] = original_sr
            else:
                sr = int(original_sr)
                file_info['processed_sample_rate'] = sr
                file_info['original_sample_rate'] = original_sr

            self.progress.emit(70)
            duration = len(y) / sr
            file_info['processed_duration'] = duration

            self.progress.emit(100)
            self.finished.emit(y, sr, duration, file_info)
        except Exception as e:
            self.error.emit(f"File loading error: {str(e)}")
