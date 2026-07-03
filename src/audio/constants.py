"""Audio processing constants for BioSeg."""

# ========================
# FILE LOADER THREAD
# ========================
TARGET_SAMPLE_RATE = 22050          # Hz, covers 0-11kHz for bird vocalizations
MAX_FILE_SIZE_MB = 500              # Maximum file size limit (MB)

# ========================
# AUDIO PROCESSING THREAD
# ========================
TARGET_LOUDNESS_LUFS = -16.0        # Target loudness for LUFS normalization
NOISE_REDUCTION_N_FFT = 2048        # FFT size for noise reduction (spectral gating)
NOISE_REDUCTION_WIN_LENGTH = 1024   # Window length for noise reduction

# ========================
# BIRD CALL DETECTION THREAD
# ========================
DEFAULT_SENSITIVITY = 0.6           # Detection sensitivity (0.1-1.0, higher = more detections)
DEFAULT_MIN_DURATION = 0.1          # Minimum call duration (seconds) - excludes short noises
DEFAULT_MAX_DURATION = 3.0          # Maximum call duration (seconds) - typical bird song length
DEFAULT_MIN_SILENCE = 0.05          # Minimum silence between calls (seconds)
DEFAULT_FRAME_LEN = 2048            # Frame length for STFT analysis
DEFAULT_HOP_LEN = 512               # Hop length for STFT analysis (25% overlap)

# ========================
# SPECTROGRAM FUNCTIONS
# ========================
N_FFT = 2048                        # FFT size for spectrogram calculation
HOP_LENGTH = 512                    # Hop length for spectrogram (50% overlap)
SPEC_MAX_TIME_COLS = 4000           # Maximum columns for performance

# ========================
# MAIN WINDOW (UI) EXPECTED NAMES
# ========================
DEFAULT_N_FFT = 2048                # Alias for N_FFT used in UI
DEFAULT_HOP = 512                   # Alias for HOP_LENGTH used in UI
SPEC_MAX_COLS = 4000                # Alias for SPEC_MAX_TIME_COLS used in UI