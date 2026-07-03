"""Audio I/O utility functions."""

import os
import wave
import numpy as np
from datetime import datetime
from scipy.io.wavfile import write as wavwrite


def get_audio_file_info(file_path):
    """Extract metadata (size, duration, sample rate, channels, etc.) from a WAV file."""
    try:
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        file_size_mb = file_size / (1024 * 1024)
        creation_time = datetime.fromtimestamp(file_stats.st_ctime)

        with wave.open(file_path, 'rb') as wav_file:
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            duration = num_frames / float(frame_rate)

            bit_depth_map = {1: "8 bit", 2: "16 bit", 3: "24 bit", 4: "32 bit"}
            bit_depth = bit_depth_map.get(sample_width, f"{sample_width * 8} bit")
            channels = "Mono" if num_channels == 1 else "Stereo" if num_channels == 2 else f"{num_channels} Channels"
            file_format = os.path.splitext(file_path)[1].upper().replace('.', '')

        return {
            'file_size': file_size,
            'file_size_mb': file_size_mb,
            'channels': channels,
            'num_channels': num_channels,
            'sample_rate': frame_rate,
            'bit_depth': bit_depth,
            'sample_width': sample_width,
            'duration': duration,
            'num_frames': num_frames,
            'creation_time': creation_time,
            'file_path': file_path,
            'file_format': file_format,
            'recording_date': creation_time.strftime('%Y-%m-%d'),
            'recording_time': creation_time.strftime('%H:%M:%S')
        }
    except Exception as e:
        print(f"Could not retrieve file info: {e}")
        return None


def write_wav_safe(path: str, y: np.ndarray, sr: int):
    """Write a WAV file safely as 16-bit PCM format."""
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Convert floating-point audio to int16
    if np.issubdtype(y.dtype, np.floating):
        y = np.clip(y, -1.0, 1.0)
        y_int16 = (y * 32767.0).astype(np.int16)

    # Convert integer audio to int16
    elif np.issubdtype(y.dtype, np.integer):
        if y.dtype != np.int16:
            max_val = np.iinfo(y.dtype).max
            y_float = y.astype(np.float32) / max_val
            y_int16 = (y_float * 32767.0).astype(np.int16)
        else:
            y_int16 = y.astype(np.int16)

    # Fallback for other types
    else:
        y = y.astype(np.float32)
        y = np.clip(y, -1.0, 1.0)
        y_int16 = (y * 32767.0).astype(np.int16)

    wavwrite(path, sr, y_int16)