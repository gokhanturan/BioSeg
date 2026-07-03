"""CSV file handling for dataset export."""

import os
import csv


def get_project_root():
    """Get the project root directory (where CSV files are stored)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # src/data/csv_handler.py -> src/data -> src -> project root
    return os.path.dirname(os.path.dirname(current_dir))


def init_csv(filepath="bird_sound_dataset.csv"):
    """Initialize CSV file with headers if it doesn't exist."""
    full_path = os.path.join(get_project_root(), filepath)

    if not os.path.exists(full_path):
        with open(full_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                "Original_Recording_Name", "Original_File_Path", "Original_Duration",
                "Channels", "File_Format", "Bit_Depth", "Recording_Date", "Recording_Time",
                "Filename", "File_Path", "Start_Time", "End_Time", "Duration",
                "Call_Type", "type_Count", "Order", "Family", "Scientific_Name", "English_Name",
                "Background_Order", "Background_Family", "Background_Scientific_Name",
                "Background_English_Name", "Background_Noise_Type",
                "Notes_For_Other_Species", "Location",
                "Recordist", "Ornithologist", "Verification_Status", "Confidence_Level",
                "Microphone", "Recorder",
                "Habitat_ID", "Habitat_Type", "Code", "UTM_Zone", "UTM_Easting", "UTM_Northing",
                "Lat", "Lon", "Google_Maps", "Notes", "Project"
            ])

    return full_path


def append_segment_to_csv(filepath, row_data):
    """Append a single segment record to the CSV file."""
    full_path = os.path.join(get_project_root(), filepath)

    with open(full_path, mode='a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f, delimiter=';').writerow(row_data)


def get_csv_filepath():
    """Return the default CSV filename."""
    return "bird_sound_dataset.csv"