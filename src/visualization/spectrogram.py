"""Spectrogram computation functions and background thread."""

import numpy as np
import scipy.signal as sps
from PyQt5.QtCore import QThread, pyqtSignal

from ..audio.constants import N_FFT, HOP_LENGTH, SPEC_MAX_TIME_COLS


def _stft_mag(y, sr, n_fft=N_FFT, hop=HOP_LENGTH):
    n_fft_safe = n_fft
    hop_safe = hop
    if len(y) < n_fft:
        n_fft_safe = min(256, len(y))
        hop_safe = min(128, n_fft_safe // 2)
        if n_fft_safe < 2:
            return np.array([]), np.array([]), np.array([])
    f, t, Zxx = sps.stft(y.astype(np.float32), fs=sr, window='hann',
                         nperseg=n_fft_safe, noverlap=n_fft_safe - hop_safe, nfft=n_fft_safe,
                         boundary=None, padded=False, return_onesided=True)
    return f.astype(np.float32), t.astype(np.float32), np.abs(Zxx, dtype=np.float32)


def _to_db(S, ref=None, amin=1e-10):
    if S.size == 0:
        return S
    P = (S ** 2).astype(np.float32)
    if ref is None:
        ref = float(np.max(P)) if P.size else 1.0
    ref = max(ref, amin)
    return 10.0 * np.log10(np.maximum(P, amin) / ref).astype(np.float32)


def _downsample_time(X, max_cols=SPEC_MAX_TIME_COLS):
    ncol = X.shape[1]
    if ncol <= max_cols:
        return X
    step = int(np.ceil(ncol / max_cols))
    return X[:, ::step].copy()


def compute_linear_spectrogram(y, sr, n_fft=N_FFT, hop=HOP_LENGTH):
    if len(y) < n_fft:
        n_fft = min(256, len(y))
        hop = min(128, n_fft // 2)
        if n_fft < 2:
            return np.array([]), [0.0, 0.0, 0.0, 0.0]
    f, t, S = _stft_mag(y, sr, n_fft=n_fft, hop=hop)
    if S.size == 0:
        return np.array([]), [0.0, 0.0, 0.0, 0.0]
    S_db = _to_db(S)
    S_db = _downsample_time(S_db)
    return S_db, [0.0, len(y) / sr, float(f[0]), float(f[-1])]


def hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def compute_mel_spectrogram(y, sr, n_fft=N_FFT, hop=HOP_LENGTH, n_mels=128):
    if len(y) < n_fft:
        n_fft = min(256, len(y))
        hop = min(128, n_fft // 2)
        if n_fft < 2:
            return np.array([]), [0.0, 0.0, 0.0, 0.0]
    f, t, S = _stft_mag(y, sr, n_fft=n_fft, hop=hop)
    if S.size == 0:
        return np.array([]), [0.0, 0.0, 0.0, 0.0]

    def mel_filterbank(n_mels, n_fft, sr, fmin=0.0, fmax=None):
        if fmax is None:
            fmax = sr / 2
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mels = np.linspace(mel_min, mel_max, n_mels + 2)
        hz = mel_to_hz(mels)
        freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        fb = np.zeros((n_mels, len(freqs)))
        for i in range(n_mels):
            l, c, r = hz[i], hz[i + 1], hz[i + 2]
            for j, fr in enumerate(freqs):
                if l <= fr < c:
                    fb[i, j] = (fr - l) / (c - l)
                elif c <= fr < r:
                    fb[i, j] = (r - fr) / (r - c)
        return fb

    fb = mel_filterbank(n_mels, n_fft, sr)
    if S.shape[0] != fb.shape[1]:
        S = S[:fb.shape[1], :]
    mel_spec = _to_db(np.dot(fb, S))
    mel_spec = _downsample_time(mel_spec)
    return mel_spec, [0.0, len(y) / sr, 0.0, float(n_mels)]


class SpectrogramComputeThread(QThread):
    finished = pyqtSignal(str, object, object)
    error = pyqtSignal(str)

    def __init__(self, y, sr, parent=None):
        super().__init__(parent)
        self.y = y
        self.sr = sr

    def run(self):
        if self.y is None or self.sr is None:
            return
        try:
            S, ext = compute_linear_spectrogram(self.y, self.sr)
            self.finished.emit('spec', S, ext)
        except Exception as e:
            self.error.emit(f"Linear Spectrogram error: {e}")
            self.finished.emit('spec', np.array([]), [0.0, 0.0, 0.0, 0.0])
        try:
            S, ext = compute_mel_spectrogram(self.y, self.sr)
            self.finished.emit('melspec', S, ext)
        except Exception as e:
            self.error.emit(f"Mel Spectrogram error: {e}")
            self.finished.emit('melspec', np.array([]), [0.0, 0.0, 0.0, 0.0])