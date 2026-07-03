# ========================
# DATA LOADING AND CONFIGURATION
# ========================

import os
import json


def get_project_root():
    """Get the project root directory (where JSON files are located)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # src/data/config.py -> src/data -> src (files are in src/)
    return os.path.dirname(current_dir)


def load_json_data(filename, default_data=None):
    """Load data from a JSON file located in the project root."""
    try:
        filepath = os.path.join(get_project_root(), filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}, using default")
            return default_data if default_data is not None else []
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return default_data if default_data is not None else []


# Load habitat and species data from JSON files
HABITAT_DATA = load_json_data("habitat_data.json", [{
    "habitat_ID": 1,
    "habitat_type": "Forest clearing",
    "code": "oia001",
    "utm_zone": "36N",
    "utm_easting": 307408.0,
    "utm_northing": 4177163.0,
    "lat": 37.721497806152726,
    "lon": 30.814775682616677,
    "google_maps": "https://www.google.com/maps?q=37.721498,30.814776",
    "location": "Isparta"
}])

BIRD_SPECIES_DATA = load_json_data("bird_species_data.json", [{
    "order": "Passeriformes",
    "family": "Turdidae",
    "scientific_name": "Turdus merula",
    "turkish_name": "Karatavuk"
}])


def load_config(filepath="bird_sound_config.json"):
    """Load configuration from JSON file or create default if missing."""
    default_config = {
        "call_types": ["Song", "Alarm", "Call", "Communication", "Other"],
        "call_typesENG": ["Song", "Alarm", "Call", "Communication", "Other"],
        "background_noise": ["Plane", "Engine", "Human Voice", "Wind", "Rain", "Mixed", "Other"],
        "background_noiseENG": ["Plane", "Engine", "Human Voice", "Wind", "Rain", "Mixed", "Other"],
        "locations": ["Burdur, Turkey", "Isparta, Turkey", "Antalya, Turkey", "Other"],
        "locationsENG": ["Burdur, Turkey", "Isparta, Turkey", "Antalya, Turkey", "Other"],
        "recordists": ["Gökhan TURAN", "Other"],
        "recordistsENG": ["Gökhan TURAN", "Other"],
        "ornithologists": ["Expert", "Labeler", "Other", "Not Verified"],
        "ornithologistsENG": ["Expert", "Labeler", "Other", "Not Verified"],
        "verification_statuses": ["Expert Verified", "Labeler Verified", "Not Verified"],
        "verification_statusesENG": ["Expert Verified", "Labeler Verified", "Not Verified"],
        "microphones": ["Zoom H4n Pro Internal", "Sennheiser MKE 600", "Audio-Technica AT897", "Other"],
        "microphonesENG": ["Zoom H4n Pro Internal", "Sennheiser MKE 600", "Audio-Technica AT897", "Other"],
        "recorders": ["Zoom H4n Pro", "Zoom H1n", "Tascam DR-40X", "Other"],
        "recordersENG": ["Zoom H4n Pro", "Zoom H1n", "Tascam DR-40X", "Other"],
        "project": "BioSeg Project 2025",
        "projectENG": "BioSeg Project 2025"
    }

    try:
        # Look for config file in project root
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                # Merge with defaults (add missing keys)
                for key, value in default_config.items():
                    if key not in config_data:
                        config_data[key] = value
                return config_data
        else:
            # Create default config file if it doesn't exist
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config


def save_config(cfg, filepath="bird_sound_config.json"):
    """Save configuration to JSON file."""
    try:
        filepath_full = os.path.join(get_project_root(), filepath)
        with open(filepath_full, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Save config error: {e}")
        return False


def get_turkish_names(bird_data):
    """Extract and return sorted list of unique Turkish bird names."""
    names = sorted(set(b.get("turkish_name", "") for b in bird_data))
    return [n for n in names if n]


def get_locations(habitat_data):
    """Extract and return sorted list of unique locations from habitat data."""
    locs = sorted(set(h.get("location", "") for h in habitat_data if h.get("location", "")))
    return locs
