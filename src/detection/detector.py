"""
Advanced bird call detection thread.
Performs multi-resolution detection, feature extraction, clustering, and grouping.
"""

import numpy as np
import scipy.signal as sps
import traceback
import sys
from PyQt5.QtCore import QThread, pyqtSignal

# UMAP and HDBSCAN (Advanced Clustering Libraries)
try:
    import umap
    import hdbscan
except ImportError:
    umap = None
    hdbscan = None
    print("WARNING: HDBSCAN and/or UMAP could not be loaded. Clustering will work with basic DBSCAN (if available).")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFileDialog, QComboBox, QTextEdit, QGroupBox,
    QProgressDialog, QMessageBox, QTabWidget, QSplitter, QSizePolicy, QFormLayout, QCheckBox, QMenu,
    QLineEdit, QSpinBox, QAction, QTableWidget, QTableWidgetItem, QInputDialog, QTableView, QAbstractItemView,
    QDialog, QHeaderView, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QUrl, QAbstractTableModel, QModelIndex
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QGuiApplication, QCursor, QDesktopServices, QColor

# Optional libraries with fallbacks
try:
    import noisereduce as nr
except ImportError:
    nr = None

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

# Signal processing and machine learning libraries
try:
    import librosa
    import librosa.display
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import RobustScaler
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score

    # Disable DBSCAN if HDBSCAN is available (prefer HDBSCAN)
    if hdbscan is not None:
        DBSCAN = None
except ImportError:
    librosa = None
    DBSCAN = None
    RobustScaler = None
    silhouette_score = None
    KMeans = None

from ..audio.constants import (
    DEFAULT_SENSITIVITY, DEFAULT_MIN_DURATION, DEFAULT_MAX_DURATION,
    DEFAULT_MIN_SILENCE, DEFAULT_FRAME_LEN, DEFAULT_HOP_LEN
)


class AdvancedBirdCallDetectionThread(QThread):
    """
    QThread class for advanced multi-resolution bird call detection and clustering.
    Performs preprocessing, feature extraction, region detection, quality filtering,
    and optional UMAP+HDBSCAN clustering.
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, y, sr, sensitivity=DEFAULT_SENSITIVITY,
                 min_duration=DEFAULT_MIN_DURATION, max_duration=DEFAULT_MAX_DURATION, parent=None):
        super().__init__(parent)
        self.y = y
        self.sr = sr
        self.sensitivity = sensitivity
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_silence = DEFAULT_MIN_SILENCE
        self.frame_len = DEFAULT_FRAME_LEN
        self.hop_len = DEFAULT_HOP_LEN
        self.min_len_samples = int(self.min_duration * sr)

    def run(self):
        """Execute the full detection pipeline."""
        try:
            # Validate required libraries
            if self.y is None or librosa is None or RobustScaler is None:
                raise RuntimeError(
                    "Audio data not loaded or required base libraries (librosa/sklearn) not found.")

            if hdbscan is None or umap is None:
                if DBSCAN is None:
                    raise RuntimeError(
                        "HDBSCAN/UMAP not installed and DBSCAN is not available. Grouping cannot be performed.")

            # Skip if audio is too short
            if len(self.y) < self.sr * 0.5:
                self.finished.emit([])
                return

            self.progress.emit(10)

            # Apply enhanced preprocessing (bandpass filter + spectral subtraction)
            y_processed = self.enhanced_preprocessing(self.y, self.sr)

            self.progress.emit(20)

            # Multi-resolution region detection
            regions_high_res = self.detect_with_resolution(y_processed, self.sr, hop_length=256)
            regions_low_res = self.detect_with_resolution(y_processed, self.sr, hop_length=512)

            self.progress.emit(40)

            # Merge overlapping regions and filter by quality
            all_regions = regions_high_res + regions_low_res
            merged_regions = self.merge_overlapping_regions(all_regions)
            filtered_regions = self.quality_filter(merged_regions, y_processed, self.sr)

            self.progress.emit(70)

            # Perform clustering-based grouping
            bird_calls = self.advanced_grouping(filtered_regions, y_processed, self.sr)

            self.progress.emit(100)
            self.finished.emit(bird_calls)

        except Exception as e:
            self.error.emit(f"Bird call detection general error: {str(e)}")

    def enhanced_preprocessing(self, y, sr):
        """
        Apply bandpass filter (500-10000 Hz) and spectral subtraction
        to highlight bird vocalizations.
        """
        try:
            # Bandpass filter for typical bird call frequencies
            nyquist = sr / 2
            lowcut = 500.0 / nyquist
            highcut = 10000.0 / nyquist
            b, a = sps.butter(4, [lowcut, highcut], btype='bandpass', output='ba')
            y_filtered = sps.filtfilt(b, a, y)

            if librosa is not None:
                # Spectral subtraction to reduce background noise
                n_fft_safe = min(self.frame_len, len(y_filtered) // 2)
                hop_length_safe = n_fft_safe // 4

                if n_fft_safe >= 256 and hop_length_safe > 0:
                    S = librosa.stft(y_filtered, n_fft=n_fft_safe, hop_length=hop_length_safe)
                    magnitude = np.abs(S)
                    phase = np.angle(S)

                    # Estimate background noise magnitude
                    num_frames = magnitude.shape[1]
                    if num_frames > 10:
                        magnitude_sorted = np.sort(magnitude, axis=1)
                        background_mag = np.mean(magnitude_sorted[:, :min(10, num_frames)], axis=1, keepdims=True)
                    else:
                        background_mag = np.mean(magnitude, axis=1, keepdims=True)

                    # Subtract background with sensitivity-based scaling
                    subtraction_factor = 1.0 - (self.sensitivity * 0.5)
                    magnitude_enhanced = magnitude - subtraction_factor * background_mag
                    magnitude_enhanced = np.maximum(magnitude_enhanced, magnitude * 0.05)

                    # Reconstruct signal
                    y_enhanced = librosa.istft(
                        magnitude_enhanced * np.exp(1j * phase),
                        hop_length=hop_length_safe,
                        length=len(y_filtered)
                    )
                    return y_enhanced
            return y_filtered

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return y

    def detect_with_resolution(self, y, sr, hop_length=512):
        """
        Detect acoustic regions using feature extraction and adaptive thresholding
        at a specified temporal resolution.
        """
        frame_length = hop_length * 4

        if len(y) < frame_length or librosa is None:
            return []

        features = self.extract_advanced_features(y, sr, hop_length, frame_length)
        if not features:
            return []

        # Ensure all feature arrays have the same length
        min_len = min(len(f) for f in features.values())
        for k in features:
            features[k] = features[k][:min_len]

        activity = self.adaptive_thresholding(features)
        regions = self.merge_activity_regions(activity, hop_length)

        # Filter by duration limits
        return [(start, end) for start, end in regions
                if self.min_duration <= (end - start) <= self.max_duration]

    def extract_advanced_features(self, y, sr, hop_length, frame_length):
        """Extract temporal and spectral features (RMS, centroid, flatness, bandwidth, deltas)."""
        features = {}

        if librosa is None or len(y) < frame_length:
            return features

        try:
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

            features['rms'] = librosa.feature.rms(
                y=y, frame_length=frame_length, hop_length=hop_length)[0]

            features['spectral_flatness'] = librosa.feature.spectral_flatness(
                y=y, n_fft=frame_length, hop_length=hop_length)[0]

            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

            features['delta_rms'] = librosa.feature.delta(features['rms'])
            features['delta_centroid'] = librosa.feature.delta(features['spectral_centroid'])

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {}

        # Pad or truncate to equalize lengths
        min_len = min(len(f) for f in features.values())
        for k in features:
            if len(features[k]) > min_len:
                features[k] = features[k][:min_len]
            elif len(features[k]) < min_len:
                features[k] = np.pad(features[k], (0, min_len - len(features[k])), mode='constant')

        return features

    def adaptive_thresholding(self, features):
        """Compute activity score and apply adaptive thresholding."""
        if not features or min(len(f) for f in features.values()) == 0:
            return np.array([], dtype=bool)

        # Normalize features to [0, 1]
        normalized_features = {}
        for key, feature in features.items():
            min_val, max_val = np.min(feature), np.max(feature)
            if max_val - min_val < 1e-9:
                normalized_features[key] = np.zeros_like(feature)
            else:
                normalized_features[key] = (feature - min_val) / (max_val - min_val)

        # Invert flatness and bandwidth (lower = more tonal/bird-like)
        if 'spectral_flatness' in normalized_features:
            normalized_features['spectral_flatness'] = 1.0 - normalized_features['spectral_flatness']

        if 'spectral_bandwidth' in normalized_features:
            normalized_features['spectral_bandwidth'] = 1.0 - normalized_features['spectral_bandwidth']

        # Weighted combination of features
        w_rms = 0.25
        w_sc = 0.25
        w_sf = 0.25
        w_delta = 0.25

        activity_score = (
            w_rms * normalized_features.get('rms', 0) +
            w_sc * normalized_features.get('spectral_centroid', 0) +
            w_sf * normalized_features.get('spectral_flatness', 0) +
            w_delta * normalized_features.get('delta_rms', 0)
        )

        # Adaptive threshold based on median and sensitivity
        global_median_score = np.median(activity_score)
        threshold_factor = 0.5 + (1.0 - self.sensitivity) * 0.4
        threshold = max(global_median_score * threshold_factor, 0.05)

        return activity_score > threshold

    def merge_activity_regions(self, activity, hop_length):
        """Merge consecutive active frames respecting a minimum silence gap."""
        if len(activity) == 0:
            return []

        regions = []
        in_region = False
        start_frame = 0
        silence_frames = 0
        max_silence_frames = int(self.min_silence * self.sr / hop_length)

        for i, active in enumerate(activity):
            if active:
                if not in_region:
                    in_region = True
                    start_frame = i
                    silence_frames = 0
                else:
                    silence_frames = 0
            elif not active and in_region:
                silence_frames += 1
                if silence_frames > max_silence_frames:
                    end_frame = i - silence_frames + 1
                    in_region = False

                    duration = (end_frame - start_frame) * hop_length / self.sr
                    if duration >= self.min_duration:
                        start_time = start_frame * hop_length / self.sr
                        end_time = end_frame * hop_length / self.sr
                        regions.append((start_time, end_time))
                    silence_frames = 0

        # Handle region that continues to the end
        if in_region:
            end_frame = len(activity)
            duration = (end_frame - start_frame) * hop_length / self.sr
            if duration >= self.min_duration:
                start_time = start_frame * hop_length / self.sr
                end_time = end_frame * hop_length / self.sr
                regions.append((start_time, end_time))

        return regions

    def merge_overlapping_regions(self, regions, overlap_threshold=0.1):
        """Merge regions that overlap or are very close."""
        if not regions:
            return []

        regions = sorted(regions, key=lambda x: x[0])
        merged = []
        current_start, current_end = regions[0]

        for start, end in regions[1:]:
            if start <= current_end + overlap_threshold:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))
        return merged

    def quality_filter(self, regions, y, sr):
        """Keep only regions with sufficient SNR and spectral density."""
        filtered = []

        for start, end in regions:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            start_sample = max(0, min(start_sample, len(y)))
            end_sample = max(start_sample, min(end_sample, len(y)))
            segment = y[start_sample:end_sample]

            if len(segment) < sr * 0.01:
                continue

            snr = self.calculate_snr(segment)
            spectral_density = self.calculate_spectral_density(segment, sr)

            # Adaptive thresholds based on sensitivity
            min_snr = 2.0 + (1.0 - self.sensitivity) * 10.0
            min_density = 0.05 + self.sensitivity * 0.1

            if snr > min_snr and spectral_density > min_density:
                filtered.append((start, end))

        return filtered

    def calculate_snr(self, segment):
        """Compute signal-to-noise ratio using start/end of segment as noise reference."""
        if len(segment) < 2000:
            return 0.0

        signal_power = np.mean(segment ** 2)
        noise_samples = min(len(segment) // 10, 1000)

        if noise_samples == 0:
            return 0.0

        noise_segment = np.concatenate([segment[:noise_samples], segment[-noise_samples:]])
        noise_power = np.mean(noise_segment ** 2) if len(noise_segment) > 0 else 1e-10

        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0

    def calculate_spectral_density(self, segment, sr):
        """Calculate ratio of energy in bird frequency band (500-10000 Hz) to total energy."""
        nperseg = min(1024, len(segment))
        if nperseg < 256:
            return 0.0

        f, Pxx = sps.welch(segment, sr, nperseg=nperseg)
        bird_band = (f >= 500) & (f <= 10000)
        total_energy = np.sum(Pxx)
        bird_energy = np.sum(Pxx[bird_band])

        return bird_energy / total_energy if total_energy > 0 else 0.0

    def advanced_grouping(self, regions, y, sr):
        """Group segments using UMAP + HDBSCAN or DBSCAN fallback."""
        if not regions:
            return []

        features = []
        hop_length = 512

        for start, end in regions:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            start_sample = max(0, min(start_sample, len(y)))
            end_sample = max(start_sample, min(end_sample, len(y)))
            segment = y[start_sample:end_sample]

            if len(segment) < hop_length or librosa is None:
                continue

            try:
                duration = end - start

                # Extract MFCCs (13 coefficients)
                mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, hop_length=hop_length)
                mean_mfccs = np.mean(mfccs, axis=1)

                # Extract spectral contrast
                contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, hop_length=hop_length)
                mean_contrast = np.mean(contrast, axis=1)

                # Zero-crossing rate and RMS energy
                zcr = librosa.feature.zero_crossing_rate(segment, hop_length=hop_length)
                mean_zcr = np.mean(zcr)
                rms = librosa.feature.rms(y=segment, hop_length=hop_length)[0]
                mean_rms = np.mean(rms)

                # Combine all features
                feature_vector = [duration, mean_zcr, mean_rms] + mean_mfccs.tolist() + mean_contrast.tolist()
                features.append(feature_vector)

            except Exception as e:
                print(f"Grouping feature extraction error: {e}")
                continue

        if len(features) < 2:
            return self.create_basic_calls(regions)

        try:
            features = np.array(features)
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features)
            labels = None

            # UMAP + HDBSCAN
            if hdbscan is not None and umap is not None:
                umap_reducer = umap.UMAP(
                    n_components=10,
                    n_neighbors=min(15, len(features_scaled) - 1),
                    min_dist=0.0,
                    random_state=42
                )
                features_reduced = umap_reducer.fit_transform(features_scaled)

                min_cluster_size = max(2, int(len(features) * 0.01))
                clustering = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    gen_min_span_tree=True,
                    allow_single_cluster=True
                )
                labels = clustering.fit_predict(features_reduced)
                print("UMAP + HDBSCAN used.")

            # Fallback to DBSCAN
            elif DBSCAN is not None:
                min_samples = max(1, len(features) // 100)
                if min_samples < 2 and len(features) >= 2:
                    min_samples = 1

                base_eps = 0.8
                eps_range = 0.5
                eps_value = base_eps + (self.sensitivity / 10.0) * eps_range
                eps_value = min(eps_value, 1.3)

                clustering = DBSCAN(eps=eps_value, min_samples=min_samples)
                labels = clustering.fit_predict(features_scaled)
                print("Fallback DBSCAN used.")
            else:
                return self.create_basic_calls(regions)

            if labels is not None and len(labels) == len(regions):
                return self.create_advanced_calls(regions, labels)
            else:
                return self.create_basic_calls(regions)

        except Exception as e:
            print(f"Grouping error: {e}")
            traceback.print_exc(file=sys.stdout)
            return self.create_basic_calls(regions)

    def create_basic_calls(self, regions):
        """Create ungrouped bird call objects (all assigned to group 0)."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        bird_calls = []
        for i, (start, end) in enumerate(regions):
            bird_calls.append({
                'segment_id': i,
                'start': start,
                'end': end,
                'duration': end - start,
                'center_time': (start + end) / 2,
                'color': colors[i % len(colors)],
                'group': 0,
                'confidence': self.calculate_confidence(end - start)
            })
        return bird_calls

    def create_advanced_calls(self, regions, labels):
        """Create grouped bird call objects with cluster-based colors."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                  'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']

        unique_labels = sorted(np.unique(labels))
        color_map = {}
        color_index = 0

        for label in unique_labels:
            if label == -1:
                color_map[label] = 'gray'      # Noise/outliers
            else:
                color_map[label] = colors[color_index % len(colors)]
                color_index += 1

        bird_calls = []
        for i, ((start, end), label) in enumerate(zip(regions, labels)):
            bird_calls.append({
                'segment_id': i,
                'start': start,
                'end': end,
                'duration': end - start,
                'center_time': (start + end) / 2,
                'color': color_map.get(label, 'gray'),
                'group': label,
                'confidence': self.calculate_confidence(end - start)
            })
        return bird_calls

    def calculate_confidence(self, duration):
        """Compute confidence score based on segment duration."""
        ideal_min, ideal_max = 0.3, 1.5
        if ideal_min <= duration <= ideal_max:
            return 0.9
        elif 0.1 <= duration < ideal_min or ideal_max < duration <= 3.0:
            return 0.7
        else:
            return 0.5
