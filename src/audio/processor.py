
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from .constants import TARGET_LOUDNESS_LUFS, NOISE_REDUCTION_N_FFT, NOISE_REDUCTION_WIN_LENGTH

# Optional libraries with fallback
try:
    import noisereduce as nr
except ImportError:
    nr = None

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None


class AudioProcessThread(QThread):
    """QThread class to perform audio processing (noise reduction, LUFS normalization) in the background."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, y, sr, t0, t1, do_denoise, do_lufs, target_lufs=TARGET_LOUDNESS_LUFS, parent=None):
        super().__init__(parent)
        self.y = y
        self.sr = sr
        self.t0 = max(0.0, float(t0))
        self.t1 = max(self.t0, float(t1))
        self.do_denoise = do_denoise
        self.do_lufs = do_lufs
        self.target_lufs = float(target_lufs)

    def run(self):
        """Applies noise reduction and/or LUFS normalization to the audio segment."""
        try:
            # Validate input
            if self.y is None or self.sr is None:
                raise RuntimeError("Load an audio file first.")

            # Calculate start and end sample indices
            start = int(self.t0 * self.sr)
            end = int(self.t1 * self.sr) if self.t1 > self.t0 else len(self.y)

            start = max(0, min(start, len(self.y)))
            end = max(start, min(end, len(self.y)))

            # Handle empty or too short intervals
            if start >= end or (end - start) < self.sr * 0.01:
                self.finished.emit({
                    'y_out': self.y[start:end].copy().astype(np.float32),
                    't0': self.t0,
                    't1': self.t1,
                    'msg': 'Interval too short (No-op)'
                })
                return

            original_length = end - start

            # Extract audio segment
            try:
                y_work = self.y[start:end].copy().astype(np.float32)
            except IndexError:
                raise RuntimeError(f"Audio array slicing error: Start={start}, End={end}, Total={len(self.y)}")

            # Ensure consistent length after slicing
            if len(y_work) != original_length:
                if len(y_work) > original_length:
                    y_work = y_work[:original_length]
                else:
                    y_work = np.pad(y_work, (0, original_length - len(y_work)), mode='constant')

            self.progress.emit(5)

            # Skip if no operations requested
            if not self.do_denoise and not self.do_lufs:
                self.finished.emit({'y_out': y_work, 't0': self.t0, 't1': self.t1, 'msg': 'No-op'})
                return

            # Apply noise reduction
            if self.do_denoise:
                if nr is None:
                    raise RuntimeError("noisereduce not installed (pip install noisereduce).")

                n_fft_safe = NOISE_REDUCTION_N_FFT
                win_length_safe = NOISE_REDUCTION_WIN_LENGTH

                if len(y_work) < n_fft_safe:
                    print("Warning: Audio segment too short for noise reduction")
                else:
                    # Estimate noise from the first part of the segment (1 second or 10%)
                    noise_len_s = min(int(1 * self.sr), len(y_work) // 10)
                    y_noise = y_work[:noise_len_s] if noise_len_s > 0 else y_work

                    original_len = len(y_work)
                    y_work = nr.reduce_noise(
                        y=y_work,
                        y_noise=y_noise,
                        sr=self.sr,
                        prop_decrease=0.9,
                        stationary=False,
                        n_fft=n_fft_safe,
                        win_length=win_length_safe
                    ).astype(np.float32)

                    # Restore original length if changed
                    if len(y_work) != original_len:
                        if len(y_work) > original_len:
                            y_work = y_work[:original_len]
                        else:
                            y_work = np.pad(y_work, (0, original_len - len(y_work)), mode='constant')

                self.progress.emit(45)

            # Apply LUFS loudness normalization
            if self.do_lufs:
                if pyln is None:
                    raise RuntimeError("pyloudnorm not installed (pip install pyloudnorm).")

                if len(y_work) > 0:
                    meter = pyln.Meter(self.sr)
                    try:
                        current_loudness = meter.integrated_loudness(y_work.astype(float))
                        gain = self.target_lufs - current_loudness
                        linear_gain = 10 ** (gain / 20.0)

                        original_len = len(y_work)
                        # Apply gain and clip to [-1.0, 1.0] range
                        y_work = np.clip(y_work * linear_gain, -1.0, 1.0).astype(np.float32)

                        # Restore original length if changed
                        if len(y_work) != original_len:
                            if len(y_work) > original_len:
                                y_work = y_work[:original_len]
                            else:
                                y_work = np.pad(y_work, (0, original_len - len(y_work)), mode='constant')
                    except Exception as e:
                        print(f"LUFS normalization error: {e}")

                self.progress.emit(80)

            self.finished.emit({'y_out': y_work, 't0': self.t0, 't1': self.t1, 'msg': "Process complete."})

        except Exception as e:
            import traceback
            print(f"CRITICAL PROCESSING ERROR: {e}")
            traceback.print_exc()
            self.error.emit(str(e))
