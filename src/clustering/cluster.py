"""Advanced clustering functions for grouping similar bird calls."""

import numpy as np
import traceback

# Optional imports with fallbacks
try:
    import librosa
    from sklearn.preprocessing import RobustScaler
except ImportError:
    librosa = None
    RobustScaler = None

try:
    import umap
    import hdbscan
except ImportError:
    umap = None
    hdbscan = None
    print("WARNING: HDBSCAN and/or UMAP could not be loaded in cluster.py")

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None


def advanced_grouping(regions, y, sr, sensitivity):
    """Group acoustic segments using UMAP + HDBSCAN (or DBSCAN as fallback)."""
    if not regions:
        return []

    features = []
    hop_length = 512

    # Extract feature vectors for each segment
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

            # Extract zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(segment, hop_length=hop_length)
            mean_zcr = np.mean(zcr)

            # Extract RMS energy
            rms = librosa.feature.rms(y=segment, hop_length=hop_length)[0]
            mean_rms = np.mean(rms)

            # Combine all features into a single vector
            feature_vector = [duration, mean_zcr, mean_rms] + mean_mfccs.tolist() + mean_contrast.tolist()
            features.append(feature_vector)

        except Exception as e:
            print(f"Grouping feature extraction error: {e}")
            continue

    # Not enough segments for meaningful clustering
    if len(features) < 2:
        return _create_basic_calls(regions)

    try:
        features = np.array(features)
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
        labels = None

        # Use UMAP + HDBSCAN if available
        if hdbscan is not None and umap is not None:
            # Reduce dimensionality with UMAP
            umap_reducer = umap.UMAP(
                n_components=10,
                n_neighbors=min(15, len(features_scaled) - 1),
                min_dist=0.0,
                random_state=42
            )
            features_reduced = umap_reducer.fit_transform(features_scaled)

            # Cluster with HDBSCAN
            min_cluster_size = max(2, int(len(features) * 0.01))
            clustering = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                gen_min_span_tree=True,
                allow_single_cluster=True
            )
            labels = clustering.fit_predict(features_reduced)
            print("UMAP + HDBSCAN used.")

        # Fallback to DBSCAN if UMAP/HDBSCAN not available
        elif DBSCAN is not None:
            min_samples = max(1, len(features) // 100)
            if min_samples < 2 and len(features) >= 2:
                min_samples = 1

            # Adjust epsilon based on sensitivity (higher sensitivity = smaller epsilon)
            base_eps = 0.8
            eps_range = 0.5
            eps_value = base_eps + (sensitivity / 10.0) * eps_range
            eps_value = min(eps_value, 1.3)

            clustering = DBSCAN(eps=eps_value, min_samples=min_samples)
            labels = clustering.fit_predict(features_scaled)
            print("Fallback DBSCAN used.")

        else:
            return _create_basic_calls(regions)

        # Return advanced calls if clustering succeeded
        if labels is not None and len(labels) == len(regions):
            return _create_advanced_calls(regions, labels)
        else:
            return _create_basic_calls(regions)

    except Exception as e:
        print(f"Grouping error: {e}")
        traceback.print_exc()
        return _create_basic_calls(regions)


def _create_basic_calls(regions):
    """Create bird call objects without grouping (all assigned to group 0)."""
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
            'confidence': _calculate_confidence(end - start)
        })
    return bird_calls


def _create_advanced_calls(regions, labels):
    """Create bird call objects with cluster-based grouping and colors."""
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
              'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']

    unique_labels = sorted(np.unique(labels))
    color_map = {}
    color_index = 0

    for label in unique_labels:
        if label == -1:
            color_map[label] = 'gray'      # Noise/outlier cluster
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
            'confidence': _calculate_confidence(end - start)
        })
    return bird_calls


def _calculate_confidence(duration):
    """Calculate confidence score based on segment duration."""
    # Ideal duration range for bird calls (0.3 - 1.5 seconds)
    ideal_min, ideal_max = 0.3, 1.5

    if ideal_min <= duration <= ideal_max:
        return 0.9          # High confidence for ideal duration
    elif 0.1 <= duration < ideal_min or ideal_max < duration <= 3.0:
        return 0.7          # Medium confidence for acceptable duration
    else:
        return 0.5          # Low confidence for unusual durations