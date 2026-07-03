"""Feature extraction and detection helper functions for bird call detection."""

import numpy as np
import scipy.signal as sps

try:
    import librosa
except ImportError:
    librosa = None


def extract_advanced_features(y, sr, hop_length, frame_length):
    """Extracts advanced temporal and spectral features (RMS, centroid, flatness, bandwidth, deltas)."""
    features = {}

    if librosa is None or len(y) < frame_length:
        return features

    try:
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
        )[0]

        features['rms'] = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]

        features['spectral_flatness'] = librosa.feature.spectral_flatness(
            y=y, n_fft=frame_length, hop_length=hop_length
        )[0]

        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
        )[0]

        features['delta_rms'] = librosa.feature.delta(features['rms'])
        features['delta_centroid'] = librosa.feature.delta(features['spectral_centroid'])

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return {}

    # Equalize feature lengths
    min_len = min(len(f) for f in features.values())
    for k in features:
        if len(features[k]) > min_len:
            features[k] = features[k][:min_len]
        elif len(features[k]) < min_len:
            features[k] = np.pad(features[k], (0, min_len - len(features[k])), mode='constant')

    return features


def adaptive_thresholding(features, sensitivity):
    """Activity detection with adaptive thresholding based on a weighted feature score."""
    if not features or min(len(f) for f in features.values()) == 0:
        return np.array([], dtype=bool)

    # Normalize features to 0-1 range
    normalized_features = {}
    for key, feature in features.items():
        min_val, max_val = np.min(feature), np.max(feature)
        if max_val - min_val < 1e-9:
            normalized_features[key] = np.zeros_like(feature)
        else:
            normalized_features[key] = (feature - min_val) / (max_val - min_val)

    # Invert spectral flatness and bandwidth (lower values = more tonal/bird-like)
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

    # Calculate adaptive threshold
    global_median_score = np.median(activity_score)
    threshold_factor = 0.5 + (1.0 - sensitivity) * 0.4
    threshold = max(global_median_score * threshold_factor, 0.05)

    return activity_score > threshold


def merge_activity_regions(activity, min_silence, sr, hop_length):
    """Merges consecutive active frames respecting a minimum silence gap."""
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
                if duration >= 0.05:  # Minimum 50ms
                    start_time = start_frame * hop_length / sr
                    end_time = end_frame * hop_length / sr
                    regions.append((start_time, end_time))
                silence_frames = 0

    # Close the last region if activity continues until the end
    if in_region:
        end_frame = len(activity)
        duration = (end_frame - start_frame) * hop_length / sr
        if duration >= 0.05:
            start_time = start_frame * hop_length / sr
            end_time = end_frame * hop_length / sr
            regions.append((start_time, end_time))

    return regions


def merge_overlapping_regions(regions, overlap_threshold=0.1):
    """Merges overlapping regions with a specific threshold."""
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
    """Calculates signal-to-noise ratio (noise estimation from segment start/end)."""
    if len(segment) < 2000:
        return 0.0

    signal_power = np.mean(segment ** 2)

    noise_samples = min(len(segment) // 10, 1000)
    if noise_samples == 0:
        return 0.0

    noise_segment = np.concatenate([segment[:noise_samples], segment[-noise_samples:]])
    noise_power = np.mean(noise_segment ** 2) if len(noise_segment) > 0 else 1e-10

    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0


def calculate_spectral_density(segment, sr):
    """Calculates spectral density in typical bird call band (500-10000 Hz)."""
    nperseg_val = min(1024, len(segment))
    if nperseg_val < 256:
        return 0.0

    f, Pxx = sps.welch(segment, sr, nperseg=nperseg_val)
    bird_band = (f >= 500) & (f <= 10000)
    total_energy = np.sum(Pxx)
    bird_energy = np.sum(Pxx[bird_band])

    return bird_energy / total_energy if total_energy > 0 else 0.0


def quality_filter(regions, y, sr, sensitivity):
    """Filters regions based on audio quality metrics (SNR and spectral density)."""
    filtered = []

    for start, end in regions:
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        start_sample = max(0, min(start_sample, len(y)))
        end_sample = max(start_sample, min(end_sample, len(y)))

        segment = y[start_sample:end_sample]

        if len(segment) < sr * 0.01:
            continue

        snr = calculate_snr(segment)
        spectral_density = calculate_spectral_density(segment, sr)

        # Adaptive thresholds based on sensitivity
        min_snr = 2.0 + (1.0 - sensitivity) * 10.0
        min_density = 0.05 + sensitivity * 0.1

        if snr > min_snr and spectral_density > min_density:
            filtered.append((start, end))

    return filtered