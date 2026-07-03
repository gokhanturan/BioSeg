"""Feature extraction and detection helper functions for bird call detection."""

import numpy as np
import scipy.signal as sps

try:
    import librosa
except ImportError:
    librosa = None


def extract_advanced_features(y, sr, hop_length, frame_length):
    """Extract temporal and spectral features (RMS, centroid, flatness, bandwidth, deltas)."""
    features = {}
    if librosa is None or len(y) < frame_length:
        return features

    try:
        # Spectral centroid (brightness of sound)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

        # RMS energy
        features['rms'] = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Spectral flatness (tonal vs noise-like)
        features['spectral_flatness'] = librosa.feature.spectral_flatness(
            y=y, n_fft=frame_length, hop_length=hop_length)[0]

        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

        # Temporal deltas for RMS and centroid
        features['delta_rms'] = librosa.feature.delta(features['rms'])
        features['delta_centroid'] = librosa.feature.delta(features['spectral_centroid'])

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return {}

    # Ensure all feature arrays have the same length
    min_len = min(len(f) for f in features.values())
    for key in features:
        if len(features[key]) > min_len:
            features[key] = features[key][:min_len]
        elif len(features[key]) < min_len:
            features[key] = np.pad(features[key], (0, min_len - len(features[key])), mode='constant')

    return features


def adaptive_thresholding(features, sensitivity):
    """Detect activity using adaptive thresholding on weighted feature scores."""
    if not features or min(len(f) for f in features.values()) == 0:
        return np.array([], dtype=bool)

    # Normalize features to [0, 1] range
    normalized = {}
    for key, feature in features.items():
        min_val, max_val = np.min(feature), np.max(feature)
        if max_val - min_val < 1e-9:
            normalized[key] = np.zeros_like(feature)
        else:
            normalized[key] = (feature - min_val) / (max_val - min_val)

    # Invert flatness and bandwidth (lower = more tonal/bird-like)
    if 'spectral_flatness' in normalized:
        normalized['spectral_flatness'] = 1.0 - normalized['spectral_flatness']
    if 'spectral_bandwidth' in normalized:
        normalized['spectral_bandwidth'] = 1.0 - normalized['spectral_bandwidth']

    # Weighted combination of features (sum = 1.0)
    w_rms, w_sc, w_sf, w_delta = 0.25, 0.25, 0.25, 0.25

    activity_score = (
        w_rms * normalized.get('rms', 0) +
        w_sc * normalized.get('spectral_centroid', 0) +
        w_sf * normalized.get('spectral_flatness', 0) +
        w_delta * normalized.get('delta_rms', 0)
    )

    # Adaptive threshold based on median score and sensitivity
    threshold = np.median(activity_score) * (0.5 + (1.0 - sensitivity) * 0.4)
    threshold = max(threshold, 0.05)

    return activity_score > threshold


def merge_activity_regions(activity, min_silence, sr, hop_length):
    """Merge consecutive active frames into regions respecting minimum silence gap."""
    if len(activity) == 0:
        return []

    regions = []
    in_region = False
    start_frame = 0
    silence_frames = 0
    max_silence_frames = int(min_silence * sr / hop_length)

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

                duration = (end_frame - start_frame) * hop_length / sr
                if duration >= 0.05:  # Minimum 50ms region duration
                    start_time = start_frame * hop_length / sr
                    end_time = end_frame * hop_length / sr
                    regions.append((start_time, end_time))
                silence_frames = 0

    # Handle region that continues to the end
    if in_region:
        end_frame = len(activity)
        duration = (end_frame - start_frame) * hop_length / sr
        if duration >= 0.05:
            start_time = start_frame * hop_length / sr
            end_time = end_frame * hop_length / sr
            regions.append((start_time, end_time))

    return regions


def merge_overlapping_regions(regions, overlap_threshold=0.1):
    """Merge overlapping or closely spaced regions."""
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


def calculate_snr(segment):
    """Calculate signal-to-noise ratio using segment start/end as noise reference."""
    if len(segment) < 2000:
        return 0.0

    signal_power = np.mean(segment ** 2)
    noise_samples = min(len(segment) // 10, 1000)

    if noise_samples == 0:
        return 0.0

    # Estimate noise from first and last 10% of segment
    noise_segment = np.concatenate([segment[:noise_samples], segment[-noise_samples:]])
    noise_power = np.mean(noise_segment ** 2) if len(noise_segment) > 0 else 1e-10

    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0


def calculate_spectral_density(segment, sr):
    """Calculate ratio of energy in bird frequency band (500-10000 Hz) to total energy."""
    nperseg = min(1024, len(segment))
    if nperseg < 256:
        return 0.0

    f, Pxx = sps.welch(segment, sr, nperseg=nperseg)
    bird_band = (f >= 500) & (f <= 10000)
    total_energy = np.sum(Pxx)
    bird_energy = np.sum(Pxx[bird_band])

    return bird_energy / total_energy if total_energy > 0 else 0.0


def quality_filter(regions, y, sr, sensitivity):
    """
    Filter regions based on SNR and spectral density.
    Only keep regions that meet adaptive quality thresholds.
    """
    filtered = []

    for start, end in regions:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        start_sample = max(0, min(start_sample, len(y)))
        end_sample = max(start_sample, min(end_sample, len(y)))
        segment = y[start_sample:end_sample]

        if len(segment) < sr * 0.01:  # Minimum 10ms segment
            continue

        snr = calculate_snr(segment)
        spectral_density = calculate_spectral_density(segment, sr)

        # Adaptive thresholds based on sensitivity (higher sensitivity = lower thresholds)
        min_snr = 2.0 + (1.0 - sensitivity) * 10.0
        min_density = 0.05 + sensitivity * 0.1

        if snr > min_snr and spectral_density > min_density:
            filtered.append((start, end))

    return filtered
