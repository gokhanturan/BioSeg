
import os
import csv
import sys
import numpy as np
import tempfile
import warnings
import json
import wave
from datetime import datetime
from scipy.io.wavfile import read as wavread, write as wavwrite, WavFileWarning
import scipy.signal as sps
from scipy import signal
import traceback
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFileDialog, QComboBox, QTextEdit, QGroupBox,
    QProgressDialog, QMessageBox, QTabWidget, QSplitter, QSizePolicy, QFormLayout, QCheckBox, QMenu,
    QLineEdit, QSpinBox, QAction, QTableWidget, QTableWidgetItem, QInputDialog, QTableView, QAbstractItemView,
    QDialog, QHeaderView, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QUrl, QAbstractTableModel, QModelIndex, QThread
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QGuiApplication, QCursor, QDesktopServices, QColor

# Matplotlib
import matplotlib.pyplot as plt
try:
    plt.switch_backend('Qt5Agg')
except ImportError:
    pass
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle


from ..audio.constants import (
    TARGET_SAMPLE_RATE, MAX_FILE_SIZE_MB, TARGET_LOUDNESS_LUFS,
    NOISE_REDUCTION_N_FFT, NOISE_REDUCTION_WIN_LENGTH,
    DEFAULT_SENSITIVITY, DEFAULT_MIN_DURATION, DEFAULT_MAX_DURATION,
    DEFAULT_MIN_SILENCE, DEFAULT_FRAME_LEN, DEFAULT_HOP_LEN,
    DEFAULT_N_FFT, DEFAULT_HOP, SPEC_MAX_COLS
)
from ..audio.utils import get_audio_file_info, write_wav_safe
from ..audio.loader import FileLoaderThread
from ..audio.processor import AudioProcessThread
from ..detection.detector import AdvancedBirdCallDetectionThread
from ..visualization.spectrogram import SpectrogramComputeThread
from ..data.config import HABITAT_DATA, BIRD_SPECIES_DATA, load_config, save_config


try:
    import librosa
except ImportError:
    librosa = None
try:
    import noisereduce as nr
except ImportError:
    nr = None
try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None
try:
    import umap
    import hdbscan
except ImportError:
    umap = None
    hdbscan = None
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import RobustScaler
except ImportError:
    DBSCAN = None
    RobustScaler = None

class BirdSoundApp(QMainWindow):
    """Main class for bird sound analysis and labeling app (manages PyQt5 UI)"""

    def __init__(self):
        super().__init__()
        self.ses_dosyasi = None
        self.current_full_media_path = None
        self.modified_temp_full = None
        self.y = None
        self.sr = None
        self.duration = 0.0
        self.view_mode = 'waveform'
        self.spec_cache = None
        self.melspec_cache = None
        self.selection_start = 0.0
        self.selection_end = 0.0
        self.is_selecting = False
        self.current_position = 0.0
        self.position_line = None
        self.selection_position_line = None
        self.selection_rect = None
        self.is_playing = False
        self.temp_file = None
        self._is_looping = False
        self._selection_dirty = True
        self._selection_sig = None
        self.current_xlim = None
        self.last_pan_x = None
        self.csv_file = "bird_sound_dataset.csv"
        self.controls_panel = None
        self.pan_slider = None
        self.file_info = None
        self.original_file_info = None
        self.segment_save_message = ""
        self.right_click_start_pos = None
        self.bird_calls = []  # Holds detected segments
        self.bird_call_rects = []
        self.detected_bird_calls_table = None

        self.habitat_data = HABITAT_DATA
        self.bird_species_data = BIRD_SPECIES_DATA
        self.config_file = "bird_sound_config.json"
        self.config_data = load_config()

        self.group_label_map = {}

        # References for threads
        self.loader_thread = None
        self.proc_thread = None
        self.detection_thread = None
        self.spec_thread = None
        self.feature_thread = None

        # Memory management
        self.memory_watchdog_timer = QTimer()
        self.memory_watchdog_timer.timeout.connect(self.check_memory_usage)

        self.setWindowIcon(self.create_icon())

        self.load_config()
        self.initUI()
        self.init_csv()

        # Start Memory watchdog (check every 30 seconds)
        self.memory_watchdog_timer.start(30000)

    def cleanup_memory(self):
        """Performs memory cleanup"""
        try:
            if hasattr(self, 'y') and self.y is not None:
                del self.y
                self.y = None

            if hasattr(self, 'spec_cache'):
                del self.spec_cache
                self.spec_cache = None

            if hasattr(self, 'melspec_cache'):
                del self.melspec_cache
                self.melspec_cache = None

            # Trigger garbage collection
            import gc
            gc.collect()

            print("Memory cleanup completed")
        except Exception as e:
            print(f"Memory cleanup error: {e}")

    def check_memory_usage(self):
        """Monitor memory usage and clean if necessary"""
        try:
            import psutil
            import gc

            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > 1000:  # If using more than 1GB
                print(f"High memory usage: {memory_mb:.1f} MB")
                gc.collect()

                # User warning (optional)
                if memory_mb > 2000:  # If more than 2GB
                    if hasattr(self, 'lbl_file_info'):
                        self.lbl_file_info.setText(
                            f"⚠️ High memory usage ({memory_mb:.0f} MB). "
                            f"Please close and reopen the file."
                        )
        except ImportError:
            pass
        except Exception as e:
            print(f"Memory check error: {e}")

    def create_menu_bar(self):
        """Creates the application menu bar"""
        menubar = self.menuBar()

        # --------------------
        # 📁 File Menu
        # --------------------
        file_menu = menubar.addMenu('📁 File')

        open_action = QAction('📂 Open Audio File...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        view_csv_action = QAction('📊 View CSV', self)
        view_csv_action.setShortcut('Ctrl+V')
        view_csv_action.triggered.connect(self.view_csv)
        file_menu.addAction(view_csv_action)

        export_csv_action = QAction('📤 Export CSV...', self)
        export_csv_action.setShortcut('Ctrl+E')
        export_csv_action.triggered.connect(self.export_csv)
        file_menu.addAction(export_csv_action)

        file_menu.addSeparator()

        exit_action = QAction('🚪 Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --------------------
        # 👁️ View Menu
        # --------------------
        view_menu = menubar.addMenu('👁️ View')

        waveform_action = QAction('📈 Waveform', self)
        waveform_action.setShortcut('Ctrl+1')
        waveform_action.triggered.connect(lambda: self.set_view_mode('waveform'))
        view_menu.addAction(waveform_action)

        spec_action = QAction('📊 Spectrogram (Linear)', self)
        spec_action.setShortcut('Ctrl+2')
        spec_action.triggered.connect(lambda: self.set_view_mode('spec'))
        view_menu.addAction(spec_action)

        melspec_action = QAction('🎵 Spectrogram (Mel)', self)
        melspec_action.setShortcut('Ctrl+3')
        melspec_action.triggered.connect(lambda: self.set_view_mode('melspec'))
        view_menu.addAction(melspec_action)

        view_menu.addSeparator()

        zoom_in_action = QAction('🔍 Zoom In', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(lambda: self.zoom_buttons(1 / 1.2))
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction('🔍 Zoom Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(lambda: self.zoom_buttons(1.2))
        view_menu.addAction(zoom_out_action)

        zoom_reset_action = QAction('🔍 Reset Zoom', self)
        zoom_reset_action.setShortcut('Ctrl+0')
        zoom_reset_action.triggered.connect(self.reset_zoom)
        view_menu.addAction(zoom_reset_action)

        # --------------------
        # 🎵 Playback Menu
        # --------------------
        play_menu = menubar.addMenu('🎵 Playback')

        play_action = QAction('▶ Play/Pause (Full)', self)
        play_action.setShortcut('Space')
        play_action.triggered.connect(self.toggle_play)
        play_menu.addAction(play_action)

        stop_action = QAction('⏹ Stop (Full)', self)
        stop_action.setShortcut('Ctrl+Space')
        stop_action.triggered.connect(self.stop_audio)
        play_menu.addAction(stop_action)

        rewind_action = QAction('⏪ Rewind 5s', self)
        rewind_action.setShortcut('Ctrl+Left')
        rewind_action.triggered.connect(self.rewind)
        play_menu.addAction(rewind_action)

        forward_action = QAction('⏩ Forward 5s', self)
        forward_action.setShortcut('Ctrl+Right')
        forward_action.triggered.connect(self.forward)
        play_menu.addAction(forward_action)

        play_menu.addSeparator()

        play_selection_action = QAction('▶ Play Selection', self)
        play_selection_action.setShortcut('Ctrl+P')
        play_selection_action.triggered.connect(self.toggle_selection_play)
        play_menu.addAction(play_selection_action)

        loop_selection_action = QAction('🔁 Loop Selection', self)
        loop_selection_action.setShortcut('Ctrl+L')
        loop_selection_action.triggered.connect(self.loop_selection)
        play_menu.addAction(loop_selection_action)

        stop_selection_action = QAction('⏹ Stop Selection', self)
        stop_selection_action.setShortcut('Ctrl+S')
        stop_selection_action.triggered.connect(self.stop_selection)
        play_menu.addAction(stop_selection_action)

        clear_selection_action = QAction('🗑️ Clear Selection', self)
        clear_selection_action.setShortcut('Ctrl+D')
        clear_selection_action.triggered.connect(self.clear_selection)
        play_menu.addAction(clear_selection_action)

        # --------------------
        # ⚙️ Process Menu
        # --------------------
        process_menu = menubar.addMenu('⚙️ Process')

        process_action = QAction('🧹 Run Noise Reduction', self)
        process_action.setShortcut('Ctrl+R')
        process_action.triggered.connect(self.run_processing_pipeline)
        process_menu.addAction(process_action)

        process_menu.addSeparator()

        detect_action = QAction('🐦 Detect Bird Calls', self)
        detect_action.setShortcut('Ctrl+B')
        detect_action.triggered.connect(self.detect_bird_calls)
        process_menu.addAction(detect_action)

        clear_detect_action = QAction('🗑️ Clear Detection', self)
        clear_detect_action.setShortcut('Ctrl+Shift+B')
        clear_detect_action.triggered.connect(self.clear_bird_call_detection)
        process_menu.addAction(clear_detect_action)

        process_menu.addSeparator()

        save_segment_action = QAction('💾 Add Segment to Dataset', self)
        save_segment_action.setShortcut('Ctrl+Shift+S')
        save_segment_action.triggered.connect(self.save_segment)
        process_menu.addAction(save_segment_action)

        save_full_action = QAction('💾 Save Full Audio as WAV...', self)
        save_full_action.setShortcut('Ctrl+W')
        save_full_action.triggered.connect(self.save_full_audio_to_wav)
        process_menu.addAction(save_full_action)

        # --------------------
        # ❓ Help Menu
        # --------------------
        help_menu = menubar.addMenu('❓ Help')

        about_action = QAction('ℹ️ About', self)
        about_action.setShortcut('F1')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        shortcuts_action = QAction('⌨️ Shortcuts', self)
        shortcuts_action.setShortcut('F2')
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts_action)

    def load_config(self):
        """Loads configuration from JSON file or creates default."""
        default_config = {
            "call_types": ["Song", "Alarm", "Call", "Communication", "Other"],
            "bird_species": self.bird_species_data,
            "locations": ["Burdur, Turkey", "Isparta, Turkey", "Antalya, Turkey", "Other"],
            "habitats": ["Forest", "Wetland", "Field", "Meadow", "Mountain", "Coast", "Urban", "Rural", "Other"],
            "background_species": ["Sparrow", "Chaffinch", "Goldfinch", "Jay", "Blackbird", "Robin",
                                   "Blackcap", "Nightingale"],
            "background_noise": ["Plane", "Engine", "Human Voice", "Wind", "Rain", "Mixed", "Other"],
            "recordists": ["Gökhan TURAN", "Other"],
            "ornithologists": ["Expert", "Labeler", "Other", "Not Verified"],
            "verification_statuses": ["Expert Verified", "Labeler Verified", "Not Verified"],
            "microphones": ["Zoom H4n Pro Internal", "Sennheiser MKE 600", "Audio-Technica AT897", "Other"],
            "recorders": ["Zoom H4n Pro", "Zoom H1n", "Tascam DR-40X", "Other"],
            "project": "BioSeg Project 2025"
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
            else:
                self.config_data = default_config
                self.save_config()
        except Exception as e:
            print(f"Error loading config file: {e}")
            self.config_data = default_config

    def save_config(self):
        """Saves configuration file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving config file: {e}")

    def update_project_info(self, new_project_name):
        """Updates project name in config."""
        if new_project_name and new_project_name != self.config_data.get("project", ""):
            self.config_data["project"] = new_project_name
            self.save_config()
            self.txt_project.setText(new_project_name)
            QMessageBox.information(self, "Success", f"Project info updated: {new_project_name}")

    def init_csv(self):
        """Initializes CSV file with new headers (if it doesn't exist)."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
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
                    "Habitat_ID", "Habitat_Type", "Code", "UTM_Zone", "UTM_Easting", "UTM_Northing", "Lat", "Lon",
                    "Google_Maps", "Notes", "Project"
                ])

    def clear_feature_analysis(self):
        """Clears feature analysis results and related threads."""
        pass

    def toggle_play(self):
        """Changes play/stop status (Full Recording)."""
        if self.selection_player.state() == QMediaPlayer.PlayingState:
            self.stop_selection()

        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_play.setText("▶")
            self.is_playing = False
        else:
            self.player.play()
            self.btn_play.setText("⏸")
            self.is_playing = True

    def stop_audio(self):
        """Stops audio (Full Recording)."""
        self.player.stop()
        self.btn_play.setText("▶")
        self.is_playing = False
        self.current_position = 0
        self.plot_waveform()

    def rewind(self):
        """Rewinds 5 seconds (Full Recording)."""
        new_pos = max(0, self.player.position() - 5000)
        self.player.setPosition(new_pos)

    def forward(self):
        """Forwards 5 seconds (Full Recording)."""
        new_pos = min(self.player.duration(), self.player.position() + 5000)
        self.player.setPosition(new_pos)

    def change_speed(self):
        """Changes playback speed."""
        speed = self.sld_speed.value() / 10.0
        self.lbl_speed.setText(f"{speed:.1f}x")
        self.player.setPlaybackRate(speed)

    def clear_selection(self):
        """Clears selection."""
        self.selection_start = 0.0
        self.selection_end = 0.0
        self.mark_selection_dirty(stop_current=True)
        self.update_sliders_from_selection()
        self.update_selection_duration()
        self.plot_waveform()

    def ensure_selection_media_ready(self):
        """Ensures selection media is ready (creates temp file if needed)."""
        need_rebuild = self._selection_dirty
        cur_sig = self.current_selection_signature()

        if self._selection_sig != cur_sig:
            need_rebuild = True

        if self.temp_file is None or not os.path.exists(self.temp_file):
            need_rebuild = True

        if need_rebuild:
            self.create_selection_temp_file()
            self._selection_sig = cur_sig
            self._selection_dirty = False
            self.set_selection_media()

    def toggle_selection_play(self):
        """Plays/pauses selected part."""
        if self.y is None or abs(self.selection_end - self.selection_start) < 0.01:
            return

        if self.player.state() == QMediaPlayer.PlayingState:
            self.stop_audio()

        state = self.selection_player.state()

        if state == QMediaPlayer.PlayingState:
            self.selection_player.pause()
            self.btn_play_selection.setText("▶ Play")
        elif state == QMediaPlayer.PausedState:
            self.selection_player.play()
            self.btn_play_selection.setText("⏸ Pause")
        else:
            self.ensure_selection_media_ready()

            if self.temp_file and os.path.exists(self.temp_file):
                self.selection_player.setPlaybackRate(1.0)
                self.selection_player.play()
                self.btn_play_selection.setText("⏸ Pause")
                self.btn_stop_selection.setEnabled(True)
            else:
                QMessageBox.critical(self, "Error", "Temporary media file for the selection could not be created/found.")

    def create_selection_temp_file(self):
        """Creates temporary WAV file for selection."""
        if self.y is None or self.sr is None:
            return

        try:
            start_time = min(self.selection_start, self.selection_end)
            end_time = max(self.selection_start, self.selection_end)

            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)

            start_sample = max(0, min(start_sample, len(self.y)))
            end_sample = max(start_sample, min(end_sample, len(self.y)))

            if start_sample >= end_sample:
                return

            y_selected = self.y[start_sample:end_sample].copy()

            if self.temp_file and os.path.exists(self.temp_file):
                try:
                    os.unlink(self.temp_file)
                except Exception:
                    pass

            self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            write_wav_safe(self.temp_file, y_selected, self.sr)

        except Exception as e:
            print(f"❌ Selection file creation error: {e}")
            self.temp_file = None
            QMessageBox.critical(self, "Error", f"Could not create temporary file for selected range: {str(e)}")

    def set_selection_media(self):
        """Binds selection media to the player."""
        try:
            if self.temp_file and os.path.exists(self.temp_file):
                self.selection_player.setMedia(QMediaContent(QUrl()))
                media_content = QMediaContent(QUrl.fromLocalFile(self.temp_file))
                self.selection_player.setMedia(media_content)
            else:
                self.selection_player.setMedia(QMediaContent(QUrl()))
        except Exception as e:
            print(f"Media set error: {e}")

    def loop_selection(self):
        """Loops selected part."""
        if self.y is None or abs(self.selection_end - self.selection_start) < 0.01:
            return

        if self.player.state() == QMediaPlayer.PlayingState:
            self.stop_audio()

        self._is_looping = True
        self.ensure_selection_media_ready()

        try:
            self.selection_player.mediaStatusChanged.disconnect(self.on_selection_media_status_changed)
        except Exception:
            pass

        self.selection_player.mediaStatusChanged.connect(self.on_selection_media_status_changed)

        if self.temp_file and os.path.exists(self.temp_file):
            self.selection_player.setPlaybackRate(1.0)
            self.selection_player.play()
            self.btn_play_selection.setText("⏸ Pause")
            self.btn_stop_selection.setEnabled(True)
        else:
            QMessageBox.critical(self, "Error", "Selection media content could not be loaded.")
            self._is_looping = False

    def stop_selection(self):
        """Stops selection playback."""
        self.selection_player.stop()
        self.btn_play_selection.setText("▶ Play")
        self.btn_stop_selection.setEnabled(False)
        self._is_looping = False

        if self.selection_position_line:
            try:
                self.selection_position_line.remove()
            except Exception:
                pass
            self.selection_position_line = None
            self.canvas.draw_idle()

        try:
            self.selection_player.mediaStatusChanged.disconnect(self.on_selection_media_status_changed)
        except Exception:
            pass

    def on_selection_player_state_changed(self, state):
        """Called when selection player state changes."""
        if state == QMediaPlayer.PlayingState:
            self.btn_play_selection.setText("⏸ Pause")
            self.btn_stop_selection.setEnabled(True)
        elif state == QMediaPlayer.PausedState:
            self.btn_play_selection.setText("▶ Play")
            self.btn_stop_selection.setEnabled(True)
        else:
            self.btn_play_selection.setText("▶ Play")
            self.btn_stop_selection.setEnabled(False)

            if self.selection_position_line:
                try:
                    self.selection_position_line.remove()
                except Exception:
                    pass
                self.selection_position_line = None
                self.canvas.draw_idle()

            if self._is_looping:
                self._is_looping = False

    def on_file_load_error(self, error_msg: str):
        """Called on file loading error."""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        QMessageBox.critical(self, "Error", error_msg)
        self.lbl_file_info.setText("File loading error")

    def enable_controls(self, enabled: bool):
        """Enables/disables control elements."""
        for b in (self.btn_play, self.btn_stop, self.btn_rewind, self.btn_forward,
                  self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_reset):
            b.setEnabled(enabled)

        has_selection = self.selection_end > self.selection_start
        for b in (self.btn_save, self.btn_play_selection, self.btn_loop_selection):
            b.setEnabled(enabled and has_selection)

        self.btn_stop_selection.setEnabled(False)
        if hasattr(self, 'btn_run_process'):
            self.btn_run_process.setEnabled(enabled)

    def on_turkish_name_changed(self, turkish_name):
        if not turkish_name:
            self.txt_scientific_name.clear()
            self.txt_family.clear()
            self.txt_order.clear()
        else:
            for bird in self.bird_species_data:
                if bird.get("turkish_name") == turkish_name:  # Yeni anahtar
                    self.txt_scientific_name.setText(bird.get("scientific_name", ""))
                    self.txt_family.setText(bird.get("family", ""))
                    self.txt_order.setText(bird.get("order", ""))
                    break
            else:
                self.txt_scientific_name.clear()
                self.txt_family.clear()
                self.txt_order.clear()

    def on_background_turkish_name_changed(self, background_turkish_name):
        if not background_turkish_name:
            self.txt_background_order.clear()
            self.txt_background_family.clear()
            self.txt_background_scientific_name.clear()
            return

        for bird in self.bird_species_data:
            if bird.get("turkish_name") == background_turkish_name:
                self.txt_background_order.setText(bird.get("order", ""))
                self.txt_background_family.setText(bird.get("family", ""))
                self.txt_background_scientific_name.setText(bird.get("scientific_name", ""))
                return

        self.txt_background_order.clear()
        self.txt_background_family.clear()
        self.txt_background_scientific_name.clear()

    def on_konum_changed(self, konum):
        """Updates Habitat ID ComboBox when Location is changed."""
        if not konum:
            self.cmb_habitat_id.clear()
            self.cmb_habitat_id.addItem("")
            for habitat in self.habitat_data:
                self.cmb_habitat_id.addItem(str(habitat["habitat_ID"]), habitat)
            return

        self.cmb_habitat_id.clear()
        self.cmb_habitat_id.addItem("")

        for habitat in self.habitat_data:
            if habitat.get("konum", "") == konum:
                self.cmb_habitat_id.addItem(str(habitat["habitat_ID"]), habitat)

    def on_habitat_id_changed(self, index):
        """Auto-fills location and habitat info when Habitat ID is selected."""
        try:
            if index <= 0:
                self.txt_habitat.clear()
                self.txt_kod.clear()
                self.txt_utm_zone.clear()
                self.txt_utm_easting.clear()
                self.txt_utm_northing.clear()
                self.txt_lat.clear()
                self.txt_lon.clear()
                self.txt_google_maps.clear()
                self.btn_open_maps.setEnabled(False)
                return

            habitat = self.cmb_habitat_id.currentData()
            if habitat:
                self.txt_habitat.setText(habitat.get("habitat_tipi", ""))
                self.txt_kod.setText(habitat.get("kodu", ""))
                self.txt_utm_zone.setText(str(habitat.get("utm_zone", "")))
                self.txt_utm_easting.setText(str(habitat.get("utm_easting", "")))
                self.txt_utm_northing.setText(str(habitat.get("utm_northing", "")))
                self.txt_lat.setText(str(habitat.get("lat", "")))
                self.txt_lon.setText(str(habitat.get("lon", "")))

                konum = habitat.get("konum", "")
                if konum and self.cmb_location.currentText() != konum:
                    self.cmb_location.setCurrentText(konum)

                maps_link = habitat.get("google_maps", "")
                self.txt_google_maps.setText(maps_link)
                self.btn_open_maps.setEnabled(bool(maps_link and maps_link.startswith('http')))

        except Exception as e:
            print(f"Error on habitat ID change: {e}")

    def create_context_menu(self):
        """Creates right-click context menu."""
        self.context_menu = QMenu(self)

        self.action_play_selection = QAction("▶ Play Selection", self)
        self.action_pause_selection = QAction("⏸ Pause Selection", self)
        self.action_stop_selection = QAction("⏹ Stop Selection", self)
        self.action_clear_selection = QAction("🗑️ Clear Selection", self)

        self.action_play_full = QAction("▶ Play/Pause Full Audio", self)
        self.action_stop_full = QAction("⏹ Stop Full Audio", self)

        self.action_zoom_in = QAction("🔍 Zoom In", self)
        self.action_zoom_out = QAction("🔍 Zoom Out", self)
        self.action_zoom_reset = QAction("🔍 Reset Zoom", self)

        self.context_menu.addAction(self.action_play_selection)
        self.context_menu.addAction(self.action_pause_selection)
        self.context_menu.addAction(self.action_stop_selection)
        self.context_menu.addAction(self.action_clear_selection)

        self.context_menu.addSeparator()
        self.context_menu.addAction(self.action_play_full)
        self.context_menu.addAction(self.action_stop_full)

        self.context_menu.addSeparator()
        self.context_menu.addAction(self.action_zoom_in)
        self.context_menu.addAction(self.action_zoom_out)
        self.context_menu.addAction(self.action_zoom_reset)

        self.action_play_selection.triggered.connect(self.toggle_selection_play)
        self.action_pause_selection.triggered.connect(self.selection_player.pause)
        self.action_stop_selection.triggered.connect(self.stop_selection)
        self.action_clear_selection.triggered.connect(self.clear_selection)

        self.action_play_full.triggered.connect(self.toggle_play)
        self.action_stop_full.triggered.connect(self.stop_audio)

        self.action_zoom_in.triggered.connect(lambda: self.zoom_buttons(1 / 1.2))
        self.action_zoom_out.triggered.connect(lambda: self.zoom_buttons(1.2))
        self.action_zoom_reset.triggered.connect(self.reset_zoom)

    def show_context_menu(self, event):
        """Shows right-click menu and updates action status."""
        if event.inaxes == self.ax:
            has_selection = abs(self.selection_end - self.selection_start) > 0.01
            has_audio = self.y is not None

            self.action_play_selection.setEnabled(has_selection and has_audio)

            is_selection_playing = self.selection_player.state() == QMediaPlayer.PlayingState
            is_selection_loaded = self.selection_player.mediaStatus() != QMediaPlayer.NoMedia

            self.action_pause_selection.setEnabled(is_selection_playing)
            self.action_stop_selection.setEnabled(is_selection_playing or is_selection_loaded)

            self.action_clear_selection.setEnabled(has_selection)

            is_full_playing = self.player.state() == QMediaPlayer.PlayingState
            is_full_loaded = self.player.mediaStatus() != QMediaPlayer.NoMedia

            self.action_play_full.setText("⏸ Pause Full" if is_full_playing else "▶ Play Full")

            try:
                self.action_play_full.triggered.disconnect()
            except Exception:
                pass
            self.action_play_full.triggered.connect(self.toggle_play)
            self.action_play_full.setEnabled(has_audio and is_full_loaded)

            self.action_stop_full.setEnabled(has_audio and is_full_loaded)

            self.context_menu.exec_(QCursor.pos())

    def open_google_maps(self):
        """Opens Google Maps link in browser."""
        maps_url = self.txt_google_maps.text().strip()
        if maps_url and maps_url.startswith('http'):
            QDesktopServices.openUrl(QUrl(maps_url))
        else:
            QMessageBox.information(self, "Info", "Valid Google Maps link not found.")

    def set_view_mode(self, mode):
        """Sets view mode (Waveform, Linear Spectrogram, Mel Spectrogram)."""
        modes = {'waveform': 0, 'spec': 1, 'melspec': 2}
        if mode in modes:
            self.cmb_view.setCurrentIndex(modes[mode])
            self.change_view()

    def show_about(self):
        """Shows about info."""
        about_text = """
        <h3> 🐦 BioSeg Labeling Studio v1.0</h3>
        <p><b>UMAP & HDBSCAN Supported Bioacoustic Segmentation and Labeling Interface</b></p>
        <p>© 2025 | Gökhan TURAN</p>
        <p><b>Advanced Features:</b></p>
        <ul>
            <li>🎵 Advanced preprocessing and band filtering</li>
            <li>🔍 Multi-resolution bird call detection</li>
            <li>📊 Adaptive weighting score algorithm</li>
            <li>🎯 Signal quality filtering (SNR and Spectral Density)</li>
            <li>👥 Advanced grouping (UMAP + HDBSCAN/DBSCAN)</li>
            <li>📈 Detailed confidence scores</li>
            <li>🚀 Background Spectrogram Computation (Performance Optimization)</li>
            <li>🔊 Advanced Background Species & Noise Recording System</li>
        </ul>
        """
        QMessageBox.about(self, "About", about_text)

    def show_shortcuts(self):
        """Shows shortcut keys."""
        shortcuts_text = """
        <h3>⌨️ Shortcut Keys</h3>

        <h4>📁 File Operations</h4>
        <table>
        <tr><td><b>Ctrl+O</b></td><td>Open Audio File</td></tr>
        <tr><td><b>Ctrl+V</b></td><td>View CSV</td></tr>
        <tr><td><b>Ctrl+E</b></td><td>Export CSV</td></tr>
        <tr><td><b>Ctrl+Q</b></td><td>Exit</td></tr>
        </table>

        <h4>👁️ View</h4>
        <table>
        <tr><td><b>Ctrl+1</b></td><td>Waveform</td></tr>
        <tr><td><b>Ctrl+2</b></td><td>Spectrogram (Linear)</td></tr>
        <tr><td><b>Ctrl+3</b></td><td>Spectrogram (Mel)</td></tr>
        <tr><td><b>Ctrl++</b></td><td>Zoom In</td></tr>
        <tr><td><b>Ctrl+-</b></td><td>Zoom Out</td></tr>
        <tr><td><b>Ctrl+0</b></td><td>Reset Zoom</td></tr>
        </table>

        <h4>🎵 Playback Controls</h4>
        <table>
        <tr><td><b>Space</b></td><td>Play/Pause (Full)</td></tr>
        <tr><td><b>Ctrl+Space</b></td><td>Stop (Full)</td></tr>
        <tr><td><b>Ctrl+←</b></td><td>Rewind 5s</td></tr>
        <tr><td><b>Ctrl+→</b></td><td>Forward 5s</td></tr>
        <tr><td><b>Ctrl+P</b></td><td>Play Selection</td></tr>
        <tr><td><b>Ctrl+L</b></td><td>Loop Selection</td></tr>
        <tr><td><b>Ctrl+S</b></td><td>Stop Selection</td></tr>
        <tr><td><b>Ctrl+D</b></td><td>Clear Selection</td></tr>
        </table>

        <h4>⚙️ Actions</h4>
        <table>
        <tr><td><b>Ctrl+R</b></td><td>Noise Reduction</td></tr>
        <tr><td><b>Ctrl+B</b></td><td>Detect Bird Calls</td></tr>
        <tr><td><b>Ctrl+Shift+B</b></td><td>Clear Detection</td></tr>
        <tr><td><b>Ctrl+Shift+S</b></td><td>Save Segment</td></tr>
        <tr><td><b>Ctrl+W</b></td><td>Save Full Recording</td></tr>
        </table>

        <h4>🎮 Plot Controls</h4>
        <table>
        <tr><td><b>Mouse Wheel</b></td><td>Zoom</td></tr>
        <tr><td><b>Shift + Wheel</b></td><td>Horizontal Pan</td></tr>
        <tr><td><b>Right Click + Drag</b></td><td>Pan</td></tr>
        <tr><td><b>Left Click + Drag</b></td><td>Selection</td></tr>
        <tr><td><b>A/D</b></td><td>Horizontal Pan</td></tr>
        </table>
        """
        msg = QMessageBox()
        msg.setWindowTitle("Shortcuts")
        msg.setTextFormat(Qt.RichText)
        msg.setText(shortcuts_text)
        msg.exec_()

    def change_view(self):
        """Changes view mode and updates plot."""
        if not hasattr(self, "cmb_view"):
            return

        idx = self.cmb_view.currentIndex()
        modes = ['waveform', 'spec', 'melspec']
        new_mode = modes[idx] if 0 <= idx < len(modes) else 'waveform'

        if new_mode != self.view_mode:
            self.view_mode = new_mode
            if self.y is not None and self.sr is not None and \
                    (self.view_mode == 'spec' and self.spec_cache is None or self.view_mode == 'melspec' and self.melspec_cache is None):
                self.start_spectrogram_compute()
            else:
                self.plot_waveform()

    def start_spectrogram_compute(self):
        """Starts spectrogram computation in background thread."""
        if self.y is None or self.sr is None:
            return

        if self.spec_thread and self.spec_thread.isRunning():
            self.spec_thread.quit()
            self.spec_thread.wait(500)

        self.lbl_file_info.setText("Computing spectrograms in background... Please wait.")
        QApplication.setOverrideCursor(Qt.BusyCursor)

        self.spec_thread = SpectrogramComputeThread(self.y, self.sr)
        self.spec_thread.finished.connect(self.on_spectrogram_computed)
        self.spec_thread.error.connect(self.on_spectrogram_error)
        self.spec_thread.start()

    def on_spectrogram_computed(self, mode, S_db, extent):
        """Called when spectrogram computation is finished."""
        if mode == 'spec':
            self.spec_cache = (S_db, extent)
        elif mode == 'melspec':
            self.melspec_cache = (S_db, extent)

        if self.view_mode == mode:
            QApplication.restoreOverrideCursor()
            self.plot_waveform()

        if self.spec_cache is not None and self.melspec_cache is not None:
            QApplication.restoreOverrideCursor()
            self.lbl_file_info.setText(self.create_file_info_text())

    def on_spectrogram_error(self, msg):
        QApplication.restoreOverrideCursor()
        print(f"Spectrogram Error: {msg}")

    def apply_compact_mode(self, on: bool = True):
        """Applies compact mode style settings."""
        base = 12 if on else 14
        pad = "4px 8px" if on else "8px 12px"
        self.setStyleSheet(f"""
        QGroupBox {{ font-size: {base + 1}px; margin-top: 8px; }}
        QPushButton {{ font-size: {base}px; padding: {pad}; }}
        QLabel, QComboBox, QSlider, QTextEdit, QLineEdit, QTableWidget, QTableView {{ font-size: {base}px; }}
        QTabWidget::pane {{ border: 0; }}
        """)

    def load_file(self):
        """Prompts user to select audio file and starts loading."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "WAV Files (*.wav)")
        if not file_path:
            return

        if os.path.splitext(file_path)[1].lower() != ".wav":
            QMessageBox.information(self, "Warning", "Only .wav files are supported.")
            return

        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 500:
                reply = QMessageBox.question(
                    self,
                    "Large File Warning",
                    f"File size is {file_size_mb:.1f} MB. Files of this size may cause memory issues.\n"
                    f"Do you want to continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
        except Exception as e:
            print(f"File size check error: {e}")

        self.ses_dosyasi = file_path

        threads_to_stop = [self.loader_thread, self.proc_thread, self.detection_thread, self.spec_thread,
                           self.feature_thread]
        for thread in threads_to_stop:
            if thread and thread.isRunning():
                thread.quit()
                thread.wait(500)

        self.progress_dialog = QProgressDialog("Loading file...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Please Wait")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.show()

        self.loader_thread = FileLoaderThread(file_path)
        self.progress_dialog.canceled.connect(self.on_load_canceled)
        self.loader_thread.progress.connect(self.update_progress_dialog)
        self.loader_thread.finished.connect(self.on_file_loaded)
        self.loader_thread.error.connect(self.on_file_load_error)
        self.loader_thread.start()

    def on_load_canceled(self):
        threads_to_stop = [self.loader_thread, self.spec_thread, self.feature_thread]
        for thread in threads_to_stop:
            if thread and thread.isRunning():
                thread.quit()
                thread.wait(2000)

        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        self.lbl_file_info.setText("Loading canceled")

    def update_progress_dialog(self, value: int):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(value)

    def on_file_loaded(self, y, sr, duration: float, file_info: dict):
        self.cleanup_memory()

        self.y, self.sr, self.duration = y, sr, duration
        self.file_info = file_info
        self.original_file_info = file_info.copy()

        self.file_info['processed_duration'] = duration
        self.file_info['processed_sample_rate'] = sr

        if 'original_sample_rate' not in self.file_info:
            self.file_info['original_sample_rate'] = file_info['sample_rate']

        self.current_xlim = (0, self.duration)
        self.spec_cache = None
        self.melspec_cache = None

        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        self.clear_feature_analysis()
        self.clear_bird_call_detection()

        self.lbl_segment_message.setText("")
        self.update_file_info_display()
        self.lbl_total_time.setText(self.format_time(self.duration))

        max_slider_value = int(self.duration * 100)
        self.sld_start.setRange(0, max_slider_value)
        self.sld_end.setRange(0, max_slider_value)
        self.sld_end.setValue(max_slider_value)

        self.selection_start = 0.0
        self.selection_end = self.duration
        self.mark_selection_dirty(stop_current=True)
        self.update_sliders_from_selection()
        self.update_selection_duration()
        self.plot_waveform()
        self.enable_controls(True)
        self.btn_clear.setEnabled(True)

        if librosa is not None:
            self.btn_detect_bird_calls.setEnabled(True)

        self.current_full_media_path = self.ses_dosyasi
        media_content = QMediaContent(QUrl.fromLocalFile(self.current_full_media_path))
        self.player.setMedia(media_content)
        self._clear_modified_full_temp()
        self.btn_save_full.setEnabled(True)

        self.start_spectrogram_compute()

    @staticmethod
    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def current_selection_signature(self):
        return (
            round(self.selection_start, 6),
            round(self.selection_end, 6),
            self.sr,
            len(self.y) if self.y is not None else 0,
        )

    def mark_selection_dirty(self, stop_current: bool = False):
        self._selection_dirty = True
        if stop_current and self.selection_player.state() == QMediaPlayer.PlayingState:
            self.stop_selection()

        if self.selection_position_line:
            try:
                self.selection_position_line.remove()
            except Exception:
                pass
            self.selection_position_line = None
            self.canvas.draw_idle()

    def plot_waveform(self):
        self.ax.clear()
        if self.y is None or self.y.size == 0:
            self.ax.text(0.5, 0.5, "File Not Loaded", ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=16)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude / Freq')
            self.canvas.draw()
            return

        if self.view_mode == 'waveform':
            if len(self.y) > 10000:
                step = len(self.y) // 10000
                time_arr = np.linspace(0, self.duration, len(self.y))[::step]
                y_down = self.y[::step]
            else:
                time_arr = np.linspace(0, self.duration, len(self.y))
                y_down = self.y
            self.ax.plot(time_arr, y_down, 'b-', alpha=0.7, linewidth=0.5)
            self.ax.set_ylabel('Amplitude')
        else:
            cache = self.spec_cache if self.view_mode == 'spec' else self.melspec_cache
            if cache is None:
                self.ax.text(0.5, 0.5, f"Computing {self.view_mode.capitalize()} Spectrogram...",
                             ha='center', va='center', transform=self.ax.transAxes, color='orange', fontsize=12)
                self.ax.set_xlabel('Time (s)')
                self.ax.set_ylabel('Frequency (Hz)')
                self.canvas.draw()
                self.start_spectrogram_compute()
                return

            S_db, extent = cache
            try:
                if S_db.size == 0:
                    self.ax.text(0.5, 0.5, "Audio too short, spectrogram could not be created.",
                                 ha='center', va='center', transform=self.ax.transAxes, color='red', fontsize=10)
                else:
                    self.ax.imshow(S_db, origin='lower', aspect='auto', extent=extent, cmap='magma')
                self.ax.set_ylabel('Frequency (Hz)' if self.view_mode == 'spec' else 'Mel Bands')
            except Exception as e:
                self.ax.text(0.5, 0.5, f"Spectrogram error:\n{e}",
                             ha='center', va='center', transform=self.ax.transAxes, color='red', fontsize=10)
                self.ax.set_ylabel('')
                self.canvas.draw()
                return

        self.ax.set_xlabel('Time (s)')
        title_extra = {'waveform': 'Waveform', 'spec': 'Spectrogram (Linear)', 'melspec': 'Spectrogram (Mel)'}[self.view_mode]
        self.ax.set_title(f'{title_extra} — Click & Drag to select (Right-click: Menu)')
        self.ax.grid(True, alpha=0.25 if self.view_mode == 'waveform' else False)

        for call_rect in self.bird_call_rects:
            try:
                call_rect.remove()
            except Exception:
                pass
        self.bird_call_rects = []

        y_min, y_max = self.ax.get_ylim()
        height = y_max - y_min

        for bird_call in self.bird_calls:
            start = bird_call['start']
            end = bird_call['end']
            color = bird_call['color']

            rect = Rectangle(
                (start, y_min),
                end - start,
                height,
                alpha=0.3, color=color,
                label=f"Group {bird_call['group']}"
            )
            self.ax.add_patch(rect)
            self.bird_call_rects.append(rect)

        start_val = min(self.selection_start, self.selection_end)
        width = abs(self.selection_end - self.selection_start)

        if width > 1e-3:
            if not self.selection_rect or self.selection_rect not in self.ax.patches:
                if self.selection_rect:
                    try:
                        self.selection_rect.remove()
                    except Exception:
                        pass

                self.selection_rect = Rectangle(
                    (start_val, y_min),
                    width,
                    height,
                    alpha=0.3, color='red',
                    zorder=10
                )
                self.ax.add_patch(self.selection_rect)
            else:
                self.selection_rect.set_xy((start_val, y_min))
                self.selection_rect.set_width(width)
                self.selection_rect.set_height(height)
                self.selection_rect.set_visible(True)
        else:
            if self.selection_rect:
                self.selection_rect.set_visible(False)

        if not self.position_line or self.position_line not in self.ax.lines:
            self.position_line = self.ax.axvline(x=self.current_position, color='r', linestyle='-', linewidth=2)
        else:
            self.position_line.set_xdata([self.current_position, self.current_position])

        if self.selection_position_line and self.selection_player.state() != QMediaPlayer.PlayingState:
            try:
                self.selection_position_line.remove()
            except Exception:
                pass
            self.selection_position_line = None

        if self.current_xlim is not None:
            self.ax.set_xlim(self.current_xlim)
        else:
            self.ax.set_xlim(0, max(self.duration, 1e-6))

        self.update_pan_slider_state()
        self.canvas.draw()

    def zoom_buttons(self, scale_factor: float):
        if self.duration <= 0:
            return

        x_min, x_max = self.ax.get_xlim()
        cur_width = x_max - x_min
        xdata = (x_min + x_max) / 2
        new_width = max((cur_width * scale_factor), 1e-3)
        left = xdata - new_width / 2
        right = xdata + new_width / 2
        left = max(0.0, left)
        right = min(self.duration, right)
        if right - left < 1e-3:
            right = left + 1e-3

        if right == self.duration and (right - left) < new_width:
            left = max(0.0, right - new_width)

        self.current_xlim = (left, right)
        self.ax.set_xlim(self.current_xlim)
        self.update_pan_slider_state()
        self.canvas.draw()

    def reset_zoom(self):
        self.current_xlim = (0, self.duration)
        self.ax.set_xlim(self.current_xlim)
        self.update_pan_slider_state()
        self.canvas.draw()

    def on_scroll(self, event):
        if self.duration <= 0 or event.inaxes != self.ax:
            return

        if event.key == 'shift':
            x_min, x_max = self.ax.get_xlim()
            cur_width = x_max - x_min
            step = (cur_width) * (0.1 if event.button == 'up' else -0.1)
            left = max(0.0, x_min - step)
            right = left + cur_width

            if right > self.duration:
                right = self.duration
                left = max(0.0, right - cur_width)

            self.current_xlim = (left, right)
            self.ax.set_xlim(self.current_xlim)
            self.update_pan_slider_state()
            self.canvas.draw()
            return

        x_min, x_max = self.ax.get_xlim()
        cur_width = x_max - x_min
        xdata = event.xdata if event.xdata is not None else (x_min + x_max) / 2
        base_scale = 1.2
        scale_factor = (1 / base_scale) if event.button == 'up' else base_scale
        new_width = cur_width * scale_factor
        left = xdata - (xdata - x_min) * (new_width / cur_width)
        right = left + new_width

        left = max(0.0, left)
        right = min(self.duration, right)

        if right - left < 1e-3:
            right = left + 1e-3

        if right == self.duration and (right - left) < new_width:
            left = max(0.0, right - new_width)

        self.current_xlim = (left, right)
        self.ax.set_xlim(self.current_xlim)
        self.update_pan_slider_state()
        self.canvas.draw()

    def on_pan_slider(self, value: int):
        if self.duration <= 0 or self.current_xlim is None:
            return

        view_width = self.current_xlim[1] - self.current_xlim[0]
        if view_width >= self.duration:
            return

        max_left = max(0.0, self.duration - view_width)
        left = (value / 1000.0) * max_left
        right = left + view_width
        self.current_xlim = (left, right)
        self.ax.set_xlim(self.current_xlim)
        self.canvas.draw()

    def update_pan_slider_state(self):
        if self.duration <= 0 or self.current_xlim is None:
            self.pan_slider.setEnabled(False)
            return

        view_width = self.current_xlim[1] - self.current_xlim[0]
        need_pan = bool(view_width < float(self.duration) - 1e-6)
        self.pan_slider.setEnabled(need_pan)

        if need_pan:
            max_left = max(0.0, self.duration - view_width)
            pos = int((self.current_xlim[0] / max_left) * 1000) if max_left > 0 else 0
            pos = min(max(pos, 0), 1000)
            if self.pan_slider.value() != pos:
                self.pan_slider.blockSignals(True)
                self.pan_slider.setValue(pos)
                self.pan_slider.blockSignals(False)

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        if event.button == 3:
            self.last_pan_x = event.xdata
            self.right_click_start_pos = (event.x, event.y)
            return

        if event.button == 1:
            self.is_selecting = True
            self.selection_start = float(event.xdata)
            self.selection_end = float(event.xdata)
            self.mark_selection_dirty(stop_current=False)
            self.update_sliders_from_selection()
            self.update_selection_duration()
            self.plot_waveform()

    def on_release(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 3:
            if (self.right_click_start_pos is not None and
                    abs(event.x - self.right_click_start_pos[0]) < 3 and
                    abs(event.y - self.right_click_start_pos[1]) < 3):
                self.show_context_menu(event)

            self.last_pan_x = None
            self.right_click_start_pos = None
            return

        if event.xdata is None:
            self.is_selecting = False
            return

        if event.button == 1:
            self.is_selecting = False
            self.selection_end = float(event.xdata)

            if self.selection_start > self.selection_end:
                self.selection_start, self.selection_end = self.selection_end, self.selection_start

            self.selection_start = max(0.0, min(self.selection_start, self.duration))
            self.selection_end = max(0.0, min(self.selection_end, self.duration))

            self.mark_selection_dirty(stop_current=True)
            self.update_sliders_from_selection()
            self.update_selection_duration()
            self.plot_waveform()

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return

        if self.last_pan_x is not None and event.xdata is not None:
            x_min, x_max = self.ax.get_xlim()
            cur_width = x_max - x_min
            dx = self.last_pan_x - event.xdata
            left = max(0.0, x_min + dx)
            right = left + cur_width

            if right > self.duration:
                right = self.duration
                left = max(0.0, right - cur_width)

            self.current_xlim = (left, right)
            self.ax.set_xlim(self.current_xlim)
            self.update_pan_slider_state()
            self.canvas.draw()
            self.last_pan_x = event.xdata
            return

        if not self.is_selecting or event.xdata is None:
            return

        self.selection_end = float(event.xdata)

        temp_start = min(self.selection_start, self.selection_end)
        temp_end = max(self.selection_start, self.selection_end)

        temp_start = max(0.0, min(temp_start, self.duration))
        temp_end = max(0.0, min(temp_end, self.duration))

        self.lbl_start.setText(f"Start: {temp_start:.3f}s")
        self.lbl_end.setText(f"End: {temp_end:.3f}s")

        if self.selection_rect:
            current_y_min, current_y_max = self.ax.get_ylim()
            self.selection_rect.set_width(temp_end - temp_start)
            self.selection_rect.set_xy((temp_start, current_y_min))
            self.selection_rect.set_visible(True)
            self.canvas.draw_idle()

    def keyPressEvent(self, event):
        if self.duration <= 0:
            super().keyPressEvent(event)
            return

        key = event.key()
        mods = event.modifiers()

        if mods & Qt.ControlModifier:
            if key == Qt.Key_O:
                self.load_file()
                return
            if key == Qt.Key_V:
                self.view_csv()
                return
            if key == Qt.Key_E:
                self.export_csv()
                return
            if key == Qt.Key_Q:
                self.close()
                return
            if key == Qt.Key_1:
                self.set_view_mode('waveform')
                return
            if key == Qt.Key_2:
                self.set_view_mode('spec')
                return
            if key == Qt.Key_3:
                self.set_view_mode('melspec')
                return
            if key == Qt.Key_R:
                self.run_processing_pipeline()
                return
            if key == Qt.Key_B:
                self.detect_bird_calls()
                return
            if key == Qt.Key_P:
                self.toggle_selection_play()
                return
            if key == Qt.Key_L:
                self.loop_selection()
                return
            if key == Qt.Key_S:
                self.stop_selection()
                return
            if key == Qt.Key_D:
                self.clear_selection()
                return
            if key == Qt.Key_W:
                self.save_full_audio_to_wav()
                return
            if key == Qt.Key_Plus or key == Qt.Key_Equal:
                self.zoom_buttons(1 / 1.2)
                return
            if key == Qt.Key_Minus:
                self.zoom_buttons(1.2)
                return
            if key == Qt.Key_0:
                self.reset_zoom()
                return
            if key == Qt.Key_Space:
                self.stop_audio()
                return

            if mods & Qt.ShiftModifier and key == Qt.Key_S:
                self.save_segment()
                return
            if mods & Qt.ShiftModifier and key == Qt.Key_B:
                self.clear_bird_call_detection()
                return

        if key in (Qt.Key_F1, Qt.Key_F2):
            if key == Qt.Key_F1:
                self.show_about()
            elif key == Qt.Key_F2:
                self.show_shortcuts()
            return

        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_buttons(1 / 1.2)
            return
        if key in (Qt.Key_Minus,):
            self.zoom_buttons(1.2)
            return
        if key == Qt.Key_0:
            self.reset_zoom()
            return

        base = 0.01
        if mods & Qt.ControlModifier:
            base = 5.0
        elif mods & Qt.AltModifier:
            base = 0.005

        if key in (Qt.Key_Left, Qt.Key_Right):
            sign = -1 if key == Qt.Key_Left else 1

            if mods & Qt.ControlModifier:
                new_pos = self.player.position() + sign * 5000
                self.player.setPosition(new_pos)

            else:
                if (mods & Qt.ShiftModifier):
                    self.selection_end = np.clip(self.selection_end + sign * base, 0, self.duration)
                    if self.selection_end < self.selection_start:
                        self.selection_end = self.selection_start
                else:
                    width = max(0.0, self.selection_end - self.selection_start)
                    new_start = np.clip(self.selection_start + sign * base, 0, max(0, self.duration - width))
                    self.selection_start = new_start
                    self.selection_end = new_start + width

                self.mark_selection_dirty(stop_current=True)
                self.update_sliders_from_selection()
                self.update_selection_duration()
                self.plot_waveform()
            return

        if key in (Qt.Key_A, Qt.Key_D):
            x_min, x_max = self.ax.get_xlim()
            cur_width = x_max - x_min
            step = cur_width * (0.1 if key == Qt.Key_D else -0.1)
            left = max(0.0, x_min + step)
            right = left + cur_width

            if right > self.duration:
                right = self.duration
                left = max(0.0, right - cur_width)

            self.current_xlim = (left, right)
            self.ax.set_xlim(self.current_xlim)
            self.update_pan_slider_state()
            self.canvas.draw()
            return

        if key == Qt.Key_Space:
            self.toggle_play()
            return

        super().keyPressEvent(event)

    def update_selection_duration(self):
        start = min(self.selection_start, self.selection_end)
        end = max(self.selection_start, self.selection_end)
        duration = end - start

        if duration <= 0.01:
            self.lbl_selection_duration.setText("No selection")
            for b in (self.btn_play_selection, self.btn_loop_selection, self.btn_stop_selection, self.btn_save):
                b.setEnabled(False)
        else:
            self.lbl_selection_duration.setText(f"Selected duration: {duration:.3f} seconds")
            for b in (self.btn_play_selection, self.btn_loop_selection, self.btn_save):
                b.setEnabled(True)
            self.btn_stop_selection.setEnabled(False)

    def update_sliders_from_selection(self):
        if self.duration > 0:
            start = min(self.selection_start, self.selection_end)
            end = max(self.selection_start, self.selection_end)

            start_value = int(start * 100)
            end_value = int(end * 100)

            self.sld_start.blockSignals(True)
            self.sld_end.blockSignals(True)

            self.sld_start.setValue(start_value)
            self.sld_end.setValue(end_value)

            self.sld_start.blockSignals(False)
            self.sld_end.blockSignals(False)

            self.lbl_start.setText(f"Start: {start:.3f}s")
            self.lbl_end.setText(f"End: {end:.3f}s")

    def update_time_from_slider(self):
        self.selection_start = self.sld_start.value() / 100.0
        self.selection_end = self.sld_end.value() / 100.0

        self.selection_start = max(0.0, min(self.selection_start, self.duration))
        self.selection_end = max(0.0, min(self.selection_end, self.duration))

        if self.sld_start.sender() == self.sld_start and self.selection_start > self.selection_end:
            self.selection_end = self.selection_start
            self.sld_end.blockSignals(True)
            self.sld_end.setValue(int(self.selection_end * 100))
            self.sld_end.blockSignals(False)
        elif self.sld_end.sender() == self.sld_end and self.selection_end < self.selection_start:
            self.selection_start = self.selection_end
            self.sld_start.blockSignals(True)
            self.sld_start.setValue(int(self.selection_start * 100))
            self.sld_start.blockSignals(False)

        self.lbl_start.setText(f"Start: {self.selection_start:.3f}s")
        self.lbl_end.setText(f"End: {self.selection_end:.3f}s")

        self.mark_selection_dirty(stop_current=True)
        self.update_selection_duration()
        self.plot_waveform()

    def update_position(self, position_ms: int):
        self.current_position = position_ms / 1000
        self.lbl_current_time.setText(self.format_time(self.current_position))

        if self.position_line:
            self.position_line.set_xdata([self.current_position, self.current_position])
            self.canvas.draw_idle()

    def update_selection_position(self, position_ms: int):
        if self.selection_player.state() == QMediaPlayer.PlayingState:
            selection_start_time = min(self.selection_start, self.selection_end)
            selection_current_pos = selection_start_time + (position_ms / 1000.0)

            selection_end_time = max(self.selection_start, self.selection_end)

            if self._is_looping and position_ms >= self.selection_player.duration() - 50:
                self.selection_player.setPosition(0)
                return

            if selection_current_pos >= selection_end_time - 0.01:
                self.stop_selection()
                return

            if self.selection_position_line and self.selection_position_line in self.ax.lines:
                self.selection_position_line.set_xdata([selection_current_pos, selection_current_pos])
            else:
                self.selection_position_line = self.ax.axvline(
                    x=selection_current_pos,
                    color='green',
                    linestyle='-',
                    linewidth=2,
                    zorder=11
                )

            self.canvas.draw_idle()

    def update_duration(self, duration_ms: int):
        if duration_ms > 0:
            self.sld_progress.setRange(0, duration_ms)

    def update_progress(self):
        try:
            if self.player and self.player.duration() > 0:
                self.sld_progress.setValue(self.player.position())
        except Exception:
            pass

    def seek_audio(self, position: int):
        self.player.setPosition(position)

    def run_processing_pipeline(self):
        if self.y is None or self.sr is None:
            QMessageBox.information(self, "Info", "Load an audio file first.")
            return

        if self.proc_thread and self.proc_thread.isRunning():
            self.proc_thread.quit()
            self.proc_thread.wait(500)

        use_selection = (self.cmb_proc_source.currentIndex() == 1)
        t0 = min(self.selection_start, self.selection_end) if use_selection else 0.0
        t1 = max(self.selection_start, self.selection_end) if use_selection else self.duration

        if use_selection and not (t1 > t0 + 0.01):
            QMessageBox.information(self, "Info",
                                    "Selected interval is too short or invalid. Please select a range or use 'Full recording'.")
            return

        do_denoise = self.chk_denoise.isChecked()
        do_lufs = self.chk_lufs.isChecked()

        if do_denoise and nr is None:
            QMessageBox.information(self, "Info", "noisereduce not installed. install: pip install noisereduce")
            return
        if do_lufs and pyln is None:
            QMessageBox.information(self, "Info", "pyloudnorm not installed. install: pip install pyloudnorm")
            return

        self.lbl_proc_status.setText("Processing started…")
        self.btn_run_process.setEnabled(False)
        QApplication.setOverrideCursor(Qt.BusyCursor)

        self.proc_thread = AudioProcessThread(
            y=self.y, sr=self.sr, t0=t0, t1=t1,
            do_denoise=do_denoise, do_lufs=do_lufs, target_lufs=-16.0
        )
        self.proc_thread.progress.connect(lambda p: self.lbl_proc_status.setText(f"Processing… {p}%"))
        self.proc_thread.finished.connect(self.on_processing_finished)
        self.proc_thread.error.connect(self.on_processing_error)
        self.proc_thread.start()

    def on_processing_finished(self, result: dict):
        QApplication.restoreOverrideCursor()
        self.btn_run_process.setEnabled(True)

        y_out = result.get('y_out', None)
        t0 = float(result.get('t0', 0.0))
        t1 = float(result.get('t1', self.duration))

        if y_out is None:
            self.lbl_proc_status.setText("Processing finished but no output.")
            return

        start = int(t0 * self.sr)
        end = int(t1 * self.sr)

        start = max(0, min(start, len(self.y)))
        end = max(start, min(end, len(self.y)))

        target_length = end - start

        if len(y_out) != target_length:
            if len(y_out) > target_length:
                y_out = y_out[:target_length]
            else:
                y_out = np.pad(y_out, (0, target_length - len(y_out)), mode='constant')

        if start >= 0 and end <= len(self.y) and target_length > 0:
            self.y[start:end] = y_out

        self.spec_cache = None
        self.melspec_cache = None
        self.plot_waveform()
        self.lbl_proc_status.setText("Processing complete. Waveform updated. Spectrogram will be re-computed.")

        self.start_spectrogram_compute()

        self._rebuild_full_temp_and_bind_player()
        self.mark_selection_dirty(stop_current=True)
        self.btn_save_full.setEnabled(True)

    def on_processing_error(self, msg: str):
        QApplication.restoreOverrideCursor()
        self.btn_run_process.setEnabled(True)
        self.lbl_proc_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _rebuild_full_temp_and_bind_player(self):
        if self.y is None or self.sr is None:
            return

        try:
            self._clear_modified_full_temp()
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            self.modified_temp_full = temp_file.name

            write_wav_safe(self.modified_temp_full, self.y, self.sr)
            self.current_full_media_path = self.modified_temp_full

            media_content = QMediaContent(QUrl.fromLocalFile(self.current_full_media_path))
            self.player.setMedia(media_content)

            self.lbl_total_time.setText(self.format_time(self.duration))
        except Exception as e:
            self.lbl_proc_status.setText(f"Error in full recording preparation: {e}")

    def _clear_modified_full_temp(self):
        try:
            if self.modified_temp_full and os.path.exists(self.modified_temp_full):
                os.unlink(self.modified_temp_full)
        except Exception as e:
            print(f"Error deleting temp file: {e}")
        finally:
            self.modified_temp_full = None

    def save_full_audio_to_wav(self):
        if self.y is None or self.sr is None:
            QMessageBox.information(self, "Info", "No audio found to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Current Recording as WAV", "", "WAV (*.wav)")
        if not file_path:
            return

        try:
            write_wav_safe(file_path, self.y, self.sr)
            QMessageBox.information(self, "Success", f"Saved: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save: {str(e)}")

    def update_file_info_display(self):
        if self.file_info is None:
            return

        file_info_text = self.create_file_info_text()
        self.lbl_file_info.setText(file_info_text)

    def create_file_info_text(self):
        if self.file_info is None:
            return "File info not found"

        info = self.file_info

        duration = info.get('processed_duration', info.get('duration', 0.0))
        sample_rate = info.get('processed_sample_rate', info.get('sample_rate', 'Unknown'))
        original_sample_rate = info.get('original_sample_rate', 'Unknown')

        sr_info = f"{sample_rate}Hz"
        if sample_rate != original_sample_rate:
            sr_info += f" (Original: {original_sample_rate}Hz)"

        file_info_text = (
            f"<table width='100%' border='0' cellspacing='0' cellpadding='5' style='font-size: 15px;'>"
            f"<tr>"
            f"<td width='50%' valign='top' style='border-right: 1px solid #ccc; padding-right: 15px;'>"
            f"<b>📁 Basic Information</b><br>"
            f"═══════════════════════════════<br>"
            f"• 📄 <b>File:</b> {os.path.basename(self.ses_dosyasi)}<br>"
            f"• 📂 <b>Path:</b> {self.ses_dosyasi}<br>"
            f"• 💾 <b>Size:</b> {info['file_size_mb']:.2f} MB<br>"
            f"• ⏱️ <b>Duration:</b> {duration:.2f}s<br>"
            f"• 📅 <b>Recording Date:</b> {info['recording_date']}<br>"
            f"• 🕒 <b>Recording Time:</b> {info['recording_time']}<br>"
            f"</td>"
            f"<td width='50%' valign='top'>"
            f"<b>🔧 Technical Specifications</b><br>"
            f"═══════════════════════════════<br>"
            f"• 🔊 <b>Channels:</b> {info['channels']}<br>"
            f"• 📊 <b>Sampling:</b> {sr_info}<br>"
            f"• 🎚️ <b>Bit Depth:</b> {info['bit_depth']}<br>"
            f"• 📁 <b>Format:</b> {info['file_format']}<br>"
            f"• 🔄 <b>Frame Count:</b> {len(self.y):,}<br>"
            f"</td>"
            f"</tr>"
            f"</table>"
        )

        return file_info_text

    def update_detection_params(self):
        sensitivity = self.sld_sensitivity.value() / 10.0
        min_duration = self.sld_min_duration.value() / 10.0

        self.lbl_detection_status.setText(f"Parameters: Sensitivity={sensitivity:.1f}, Min. Duration={min_duration:.1f}s")

    def detect_bird_calls(self):
        if self.y is None or self.sr is None:
            QMessageBox.information(self, "Info", "Load an audio file first.")
            return

        if librosa is None:
            QMessageBox.information(self, "Info", "librosa not installed. install: pip install librosa")
            return

        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.quit()
            self.detection_thread.wait(500)

        sensitivity = self.sld_sensitivity.value() / 10.0
        min_duration = self.sld_min_duration.value() / 10.0

        if hdbscan is None or umap is None:
            if DBSCAN is None:
                QMessageBox.information(self, "Info",
                                        "Required libraries (HDBSCAN/UMAP/DBSCAN) for grouping not installed. Basic detection will be performed.")

        self.lbl_detection_status.setText("Detecting bird calls...")
        self.btn_detect_bird_calls.setEnabled(False)
        QApplication.setOverrideCursor(Qt.BusyCursor)

        self.detection_thread = AdvancedBirdCallDetectionThread(
            self.y, self.sr,
            sensitivity=sensitivity,
            min_duration=min_duration
        )
        self.detection_thread.progress.connect(lambda p: self.lbl_detection_status.setText(f"Detecting… {p}%"))
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error.connect(self.on_detection_error)
        self.detection_thread.start()

    def on_detection_finished(self, bird_calls):
        QApplication.restoreOverrideCursor()
        self.btn_detect_bird_calls.setEnabled(True)
        self.btn_clear_detection.setEnabled(True)

        self.bird_calls = bird_calls
        self.detected_bird_calls_table.setRowCount(0)

        if len(bird_calls) == 0:
            self.lbl_detection_status.setText(
                "❌ **Not Detected:** Please increase sensitivity and decrease Min Duration, then try again.")
            self.plot_waveform()
            return

        self.lbl_detection_status.setText(f"{len(bird_calls)} bird calls detected")

        self.detected_bird_calls_table.setColumnCount(6)
        self.detected_bird_calls_table.setHorizontalHeaderLabels([
            "Seg. ID", "Group ID", "Start (s)", "End (s)", "Duration (s)", "Avg. Conf."
        ])

        self.detected_bird_calls_table.setRowCount(len(self.bird_calls))

        for row, call in enumerate(self.bird_calls):
            group_id = call['group']
            color_str = call['color']
            avg_confidence = call['confidence']

            item_seg_id = QTableWidgetItem(str(call['segment_id']))
            self.detected_bird_calls_table.setItem(row, 0, item_seg_id)

            group_text = "Noise (-1)" if group_id == -1 else f"Group {group_id}"
            item_group = QTableWidgetItem(group_text)
            item_group.setData(Qt.UserRole, group_id)
            self.detected_bird_calls_table.setItem(row, 1, item_group)

            item_start = QTableWidgetItem(f"{call['start']:.3f}")
            self.detected_bird_calls_table.setItem(row, 2, item_start)

            item_end = QTableWidgetItem(f"{call['end']:.3f}")
            self.detected_bird_calls_table.setItem(row, 3, item_end)

            item_duration = QTableWidgetItem(f"{call['duration']:.3f}")
            self.detected_bird_calls_table.setItem(row, 4, item_duration)

            item_conf = QTableWidgetItem(f"{avg_confidence:.2f}")
            self.detected_bird_calls_table.setItem(row, 5, item_conf)

            background_color = QColor(color_str).lighter(150)

            for col in range(self.detected_bird_calls_table.columnCount()):
                item = self.detected_bird_calls_table.item(row, col)
                if item:
                    item.setBackground(background_color)
                    item.setForeground(QColor(Qt.black))

        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)

        self.plot_waveform()

    def on_detection_error(self, msg):
        QApplication.restoreOverrideCursor()
        self.btn_detect_bird_calls.setEnabled(True)
        self.lbl_detection_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def clear_bird_call_detection(self):
        self.bird_calls = []
        self.group_label_map = {}
        for rect in self.bird_call_rects:
            if rect in self.ax.patches:
                rect.remove()
        self.bird_call_rects = []
        self.detected_bird_calls_table.setRowCount(0)
        self.lbl_detection_status.setText("Bird call detection cleared")
        self.btn_clear_detection.setEnabled(False)

        self.clear_feature_analysis()
        self.plot_waveform()

    def on_bird_call_item_clicked(self, row, column):
        if not self.bird_calls or row >= len(self.bird_calls):
            return

        call = self.bird_calls[row]

        group_id = call['group']
        start_time = call['start']
        end_time = call['end']

        if group_id == -1:
            self.lbl_segment_message.setText("Noise/Other (-1) segment selected. No label can be assigned.")
            return

        self.selection_start = start_time
        self.selection_end = end_time

        self.mark_selection_dirty(stop_current=True)
        self.update_sliders_from_selection()
        self.update_selection_duration()
        self.plot_waveform()

        self.toggle_selection_play()

        if group_id in self.group_label_map:
            self.cmb_turkish_name.setCurrentText(self.group_label_map[group_id])
            self.lbl_segment_message.setText(
                f"Group {group_id} label '{self.group_label_map[group_id]}' auto-loaded. Segment {call['segment_id']} playing."
            )
        else:
            self.cmb_turkish_name.setCurrentText("")
            self.lbl_segment_message.setText(
                f"Group {group_id} (Segment {call['segment_id']}) selected and playing. Please assign a label."
            )

    def save_segment(self):
        if self.y is None:
            QMessageBox.warning(self, "Warning", "Load an audio file first.")
            return

        selection_duration = abs(self.selection_end - self.selection_start)

        if selection_duration == 0:
            QMessageBox.warning(self, "Warning", "No area selected. Please select a range.")
            return

        if selection_duration < 0.01:
            QMessageBox.warning(self, "Warning",
                                "Selected area is too short (< 0.01s). Please select a longer area.")
            return

        total_duration = self.duration

        if total_duration > 0 and (selection_duration / total_duration) > 0.95:
            reply = QMessageBox.question(
                self,
                "Full Recording Selected",
                f"You selected the entire recording (Selection: {selection_duration:.2f}s, Total: {total_duration:.2f}s).\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return

        turkish_name = self.cmb_turkish_name.currentText().strip()
        call_type = self.cmb_call_type.currentText().strip()

        if not turkish_name or not call_type:
            QMessageBox.critical(self, "Error", "Please fill 'English Name' and 'Call Type' fields.")
            return

        location = self.cmb_location.currentText().strip()
        if not location:
            QMessageBox.critical(self, "Error", "Please select 'Location' field.")
            return

        habitat_id_index = self.cmb_habitat_id.currentIndex()
        if habitat_id_index <= 0:
            QMessageBox.critical(self, "Error", "Please select 'Habitat ID' field.")
            return

        type_count = self.cmb_type_count.currentText()
        if type_count == "Multiple":
            background_turkish_name = self.cmb_background_turkish_name.currentText().strip()
            if not background_turkish_name:
                QMessageBox.critical(self, "Error",
                                     "When 'Multiple' species count is selected, 'Background English Name' must be filled.")
                return

        try:
            segment_filename = self._write_single_segment_to_disk_and_csv()

            if self.cmb_habitat_id.currentIndex() > 0:
                habitat = self.cmb_habitat_id.currentData()
                habitat_id = str(habitat.get("habitat_ID", ""))
                habitat_tipi = habitat.get("habitat_tipi", "")
                habitat_kodu = habitat.get("kodu", "")
            else:
                habitat_id = ""
                habitat_tipi = self.txt_habitat.text() or ""
                habitat_kodu = self.txt_kod.text() or ""

            if self.file_info:
                kayit_tarihi = self.file_info.get('recording_date', 'Unknown')
                kayit_saati = self.file_info.get('recording_time', 'Unknown')
            else:
                kayit_tarihi = "Unknown"
                kayit_saati = "Unknown"

            message = (
                f"<div style='font-family: Arial, sans-serif;'>"
                f"<h3 style='text-align: center; color: #2e7d32;'>✅ SEGMENT SAVED</h3>"
                f"<hr>"
                f"<div><b>File:</b> {segment_filename}</div>"
                f"<div><b>Species:</b> {turkish_name}</div>"
                f"<div><b>Species Count:</b> {type_count}</div>"
                f"<div><b>Location:</b> {location}</div>"
                f"<div><b>Habitat ID:</b> {habitat_id}</div>"
                f"<div><b>Bg Species:</b> {self.cmb_background_turkish_name.currentText()}</div>"
                f"<div><b>Bg Noise:</b> {self.cmb_background_noise.currentText()}</div>"
                f"<div><b>Segment Duration:</b> {selection_duration:.3f} s</div>"
                f"<hr>"
                f"<div style='font-size: 12px; color: #666;'>"
                f"Saved to 'segments' folder</div>"
                f"</div>"
            )

            msg_box = QMessageBox()
            msg_box.setWindowTitle("BioSeg Labeling Studio v1.0")
            msg_box.setTextFormat(Qt.RichText)
            msg_box.setText(message)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()

            self.lbl_segment_message.setText(f"✅ Single segment saved: {segment_filename}")

        except Exception as e:
            self.lbl_segment_message.setText(f"❌ Saving error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Could not save segment: {str(e)}")

    def _write_single_segment_to_disk_and_csv(self):
        import re

        start_time = min(self.selection_start, self.selection_end)
        end_time = max(self.selection_start, self.selection_end)

        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        start_sample = max(0, min(start_sample, len(self.y)))
        end_sample = max(start_sample, min(end_sample, len(self.y)))

        if start_sample >= end_sample:
            raise ValueError("Invalid or too short audio slice.")

        y_selected = self.y[start_sample:end_sample]

        import os
        segments_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "segments")

        os.makedirs(segments_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.ses_dosyasi or "recording"))[0]

        habitat_id_str = ""
        if self.cmb_habitat_id.currentIndex() > 0:
            habitat = self.cmb_habitat_id.currentData()
            habitat_id_str = str(habitat.get("habitat_ID", ""))
        else:
            numbers = re.findall(r'\d{2,}', base_name)
            if numbers:
                habitat_id_str = numbers[0]

        turkce_kus_turu_adi = self.cmb_turkish_name.currentText().strip()
        if not turkce_kus_turu_adi:
            turkce_kus_turu_adi = "UNKNOWN"
        else:
            turkce_kus_turu_adi = turkce_kus_turu_adi.replace(" ", "_").replace(",", "")

        tarih = ""
        tarih_match = re.search(r'(\d{8})', base_name)
        if tarih_match:
            tarih = tarih_match.group(1)
        else:
            if self.file_info and 'creation_time' in self.file_info:
                tarih = self.file_info['creation_time'].strftime("%Y%m%d")
            else:
                tarih = datetime.now().strftime("%Y%m%d")

        parts = [base_name]
        if habitat_id_str:
            parts.append(f"Habitat{habitat_id_str}")
        parts.append(turkce_kus_turu_adi)
        parts.append(tarih)
        segment_filename_base = "_".join(parts) + ".wav"
        segment_filename_base = re.sub(r'[<>:"/\\|?*]', '', segment_filename_base)

        segment_filename = segment_filename_base
        counter = 1
        while os.path.exists(os.path.join(segments_dir, segment_filename)):
            name_without_ext, ext = os.path.splitext(segment_filename_base)
            segment_filename = f"{name_without_ext}_{counter}{ext}"
            counter += 1

        segment_filepath = os.path.join(segments_dir, segment_filename)
        write_wav_safe(segment_filepath, y_selected, self.sr)

        habitat_id_index = self.cmb_habitat_id.currentIndex()
        if habitat_id_index > 0:
            habitat = self.cmb_habitat_id.currentData()
            habitat_ID = habitat.get("habitat_ID", "")
            habitat_tipi = habitat.get("habitat_tipi", "")
            kod = habitat.get("kodu", "")
            utm_zone = habitat.get("utm_zone", "")
            utm_easting = habitat.get("utm_easting", "")
            utm_northing = habitat.get("utm_northing", "")
            lat = habitat.get("lat", "")
            lon = habitat.get("lon", "")
            google_maps = habitat.get("google_maps", "")
        else:
            habitat_ID = ""
            habitat_tipi = self.txt_habitat.text()
            kod = self.txt_kod.text()
            utm_zone = self.txt_utm_zone.text()
            utm_easting = self.txt_utm_easting.text()
            utm_northing = self.txt_utm_northing.text()
            lat = self.txt_lat.text()
            lon = self.txt_lon.text()
            google_maps = self.txt_google_maps.text()

        with open(self.csv_file, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([
                os.path.basename(self.ses_dosyasi) if self.ses_dosyasi else "(temp)",
                self.ses_dosyasi if self.ses_dosyasi else "",
                round(self.file_info['duration'], 3) if self.file_info else 0,
                self.file_info['channels'] if self.file_info else "",
                self.file_info['file_format'] if self.file_info else "WAV",
                self.file_info['bit_depth'] if self.file_info else "16 bit",
                self.file_info['recording_date'] if self.file_info else "",
                self.file_info['recording_time'] if self.file_info else "",
                segment_filename,
                segment_filepath,
                round(start_time, 3),
                round(end_time, 3),
                round(end_time - start_time, 3),
                self.cmb_call_type.currentText(),
                self.cmb_type_count.currentText(),
                self.txt_order.text(),
                self.txt_family.text(),
                self.txt_scientific_name.text(),
                self.cmb_turkish_name.currentText(),
                self.txt_background_order.text(),
                self.txt_background_family.text(),
                self.txt_background_scientific_name.text(),
                self.cmb_background_turkish_name.currentText(),
                self.cmb_background_noise.currentText(),
                self.txt_other_species.toPlainText(),
                self.cmb_location.currentText(),
                self.cmb_recordist.currentText(),
                self.cmb_ornithologist.currentText(),
                self.cmb_verification_status.currentText(),
                self.spn_confidence.value(),
                self.cmb_microphone.currentText(),
                self.cmb_recorder.currentText(),
                habitat_ID,
                habitat_tipi,
                kod,
                utm_zone,
                utm_easting,
                utm_northing,
                lat,
                lon,
                google_maps,
                self.txt_notes.toPlainText(),
                self.config_data.get("project", "")
            ])

        return segment_filename

    def view_csv(self):
        try:
            if os.path.exists(self.csv_file):
                if sys.platform.startswith("win"):
                    os.startfile(self.csv_file)
                elif sys.platform == "darwin":
                    os.system(f"open '{self.csv_file}'")
                else:
                    os.system(f"xdg-open '{self.csv_file}'")
            else:
                QMessageBox.information(self, "Info", "CSV file not created yet.")
        except Exception:
            QMessageBox.information(self, "Info", f"CSV file: {self.csv_file}")

    def export_csv(self):
        if not os.path.exists(self.csv_file):
            QMessageBox.information(self, "Info", "No CSV file generated yet.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                import shutil
                shutil.copy2(self.csv_file, file_path)
                QMessageBox.information(self, "Success", f"CSV file saved as {file_path}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save CSV file: {str(e)}")

    def create_icon(self):
        try:
            from PyQt5.QtGui import QIcon, QPixmap, QPainter, QFont, QColor
            from PyQt5.QtCore import Qt

            pixmap = QPixmap(256, 256)
            pixmap.fill(Qt.transparent)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)

            font = QFont()
            font.setPointSize(180)

            bird_emoji = "🐦"

            painter.setFont(font)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, bird_emoji)

            painter.end()

            icon = QIcon(pixmap)
            return icon

        except Exception as e:
            print(f"Icon could not be created: {e}")
            return QIcon()

    def closeEvent(self, event):
        threads_to_stop = [
            ('loader_thread', self.loader_thread),
            ('proc_thread', self.proc_thread),
            ('detection_thread', self.detection_thread),
            ('spec_thread', self.spec_thread),
        ]

        for name, thread in threads_to_stop:
            if thread and thread.isRunning():
                print(f"Stopping {name}...")
                thread.quit()
                thread.wait(1000)
                if thread.isRunning():
                    thread.terminate()
                    thread.wait()

        temp_files = [self.temp_file, self.modified_temp_full]
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    print(f"Temp file deleted: {temp_file}")
                except Exception as e:
                    print(f"Temp file could not be deleted: {e}")

        if hasattr(self, 'memory_watchdog_timer'):
            self.memory_watchdog_timer.stop()

        self.cleanup_memory()

        event.accept()

    def initUI(self):
        self.setWindowTitle(
            'BioSeg Labeling Studio v1.0 — UMAP & HDBSCAN Supported Bioacoustic Labeling Interface © 2025 | Gökhan TURAN')
        self.create_menu_bar()

        screen_geo = QGuiApplication.primaryScreen().availableGeometry()
        w = min(1400, screen_geo.width() - 80)
        h = min(900, screen_geo.height() - 80)
        self.resize(w, h)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.controls_panel = QWidget()
        left_layout = QVBoxLayout(self.controls_panel)
        self.controls_panel.setMinimumHeight(400)

        def groupbox(title: str):
            g = QGroupBox(title)
            l = QVBoxLayout()
            l.setContentsMargins(12, 22, 12, 8)
            l.setSpacing(6)
            g.setLayout(l)
            return g, l

        def fixed_button(btn: QPushButton):
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            return btn

        # --- Audio File Group ---
        file_group, file_layout = groupbox("🔊 Audio File")
        top_row = QHBoxLayout()
        self.btn_load = fixed_button(QPushButton("📂 Load Audio File (.wav)"))
        self.btn_load.clicked.connect(self.load_file)
        self.btn_load.setStyleSheet("QPushButton { font-weight: bold; padding:6px 12px; min-height:32px; }")
        top_row.addWidget(self.btn_load)
        top_row.addStretch()
        file_layout.addLayout(top_row)

        self.lbl_file_info = QLabel("No File Selected! Please load a .wav file first.")
        self.lbl_file_info.setWordWrap(True)
        self.lbl_file_info.setStyleSheet(
            "QLabel { background:#f0f0f0; padding:6px 10px; border-radius:4px; font-size: 15px; }")
        file_layout.addWidget(self.lbl_file_info)

        # --- View Group ---
        view_group, v_layout = groupbox("👁️ View")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft)
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["Waveform", "Spectrogram (Linear)", "Spectrogram (Mel)"])
        self.cmb_view.currentIndexChanged.connect(self.change_view)
        self.cmb_view.setMaximumWidth(200)
        form.addRow(QLabel("View:"), self.cmb_view)
        v_layout.addLayout(form)

        # --- Playback & Noise Reduction Group ---
        playproc_group = QWidget()
        pp_layout = QVBoxLayout(playproc_group)

        control_box = QGroupBox("🎵 Full Audio Playback")
        control_layout = QVBoxLayout(control_box)

        progress_col = QVBoxLayout()
        self.sld_progress = QSlider(Qt.Horizontal)
        self.sld_progress.setRange(0, 1000)
        self.sld_progress.sliderMoved.connect(self.seek_audio)
        self.sld_progress.setMinimumHeight(16)
        progress_col.addWidget(self.sld_progress)

        time_row = QHBoxLayout()
        self.lbl_current_time = QLabel("00:00")
        self.lbl_current_time.setStyleSheet("font-weight:bold;")
        self.lbl_total_time = QLabel("00:00")
        time_row.addWidget(self.lbl_current_time)
        time_row.addWidget(QLabel(" / "))
        time_row.addWidget(self.lbl_total_time)
        time_row.addStretch()
        progress_col.addLayout(time_row)
        control_layout.addLayout(progress_col)

        transport = QHBoxLayout()
        transport.setSpacing(4)
        control_btn_style = "QPushButton { font-size:16px; min-width:40px; min-height:32px; padding:4px 10px; }"

        self.btn_rewind = fixed_button(QPushButton("⏪"))
        self.btn_rewind.clicked.connect(self.rewind)
        self.btn_rewind.setEnabled(False)
        self.btn_rewind.setStyleSheet(control_btn_style)

        self.btn_play = fixed_button(QPushButton("▶"))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        self.btn_play.setStyleSheet(control_btn_style)

        self.btn_stop = fixed_button(QPushButton("⏹"))
        self.btn_stop.clicked.connect(self.stop_audio)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(control_btn_style)

        self.btn_forward = fixed_button(QPushButton("⏩"))
        self.btn_forward.clicked.connect(self.forward)
        self.btn_forward.setEnabled(False)
        self.btn_forward.setStyleSheet(control_btn_style)

        for b in (self.btn_rewind, self.btn_play, self.btn_stop, self.btn_forward):
            transport.addWidget(b)
        transport.addStretch()
        control_layout.addLayout(transport)

        speed_row = QHBoxLayout()
        lbl_speed = QLabel("Playback Speed:")
        self.sld_speed = QSlider(Qt.Horizontal)
        self.sld_speed.setRange(5, 20)
        self.sld_speed.setValue(10)
        self.sld_speed.valueChanged.connect(self.change_speed)
        self.sld_speed.setMinimumHeight(16)
        self.lbl_speed = QLabel("1.0x")
        self.lbl_speed.setStyleSheet("font-weight:bold;")
        speed_row.addWidget(lbl_speed)
        speed_row.addWidget(self.sld_speed, 1)
        speed_row.addWidget(self.lbl_speed)
        control_layout.addLayout(speed_row)

        pp_layout.addWidget(control_box)

        denoise_box = QGroupBox("🧹 Noise Reduction / Normalization")
        dg_layout = QVBoxLayout(denoise_box)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self.cmb_proc_source = QComboBox()
        self.cmb_proc_source.addItems(["Full recording", "Selected range only"])
        src_row.addWidget(self.cmb_proc_source)
        src_row.addStretch()
        dg_layout.addLayout(src_row)

        opts = QGroupBox("Operations")
        opts_l = QHBoxLayout(opts)
        self.chk_denoise = QCheckBox("🧹 Denoise")
        self.chk_denoise.setChecked(True)
        self.chk_denoise.setToolTip("Reduces background noise in the recording.")
        self.chk_lufs = QCheckBox("🔊 LUFS Normalization")
        self.chk_lufs.setChecked(True)
        self.chk_lufs.setToolTip("Normalizes audio levels to a standard volume.")
        opts_l.addWidget(self.chk_denoise)
        opts_l.addWidget(self.chk_lufs)
        opts_l.addStretch()
        dg_layout.addWidget(opts)

        btn_row = QHBoxLayout()
        self.btn_run_process = QPushButton("▶ Run Actions")
        self.btn_run_process.clicked.connect(self.run_processing_pipeline)
        self.btn_save_full = QPushButton("💾 Save Current Recording as WAV")
        self.btn_save_full.setEnabled(False)
        btn_row.addWidget(self.btn_run_process)
        btn_row.addWidget(self.btn_save_full)
        btn_row.addStretch()
        dg_layout.addLayout(btn_row)

        self.lbl_proc_status = QLabel("Ready")
        self.lbl_proc_status.setStyleSheet("QLabel{background:#eef; padding:6px; border-radius:4px;}")
        dg_layout.addWidget(self.lbl_proc_status)

        pp_layout.addWidget(denoise_box)

        # --- Zoom Group ---
        def group_zoom():
            g, z_layout = groupbox("🔍 Zoom & Pan")
            hz_top = QHBoxLayout()
            hz_top.setSpacing(4)
            self.btn_zoom_in = fixed_button(QPushButton("+"))
            self.btn_zoom_in.clicked.connect(lambda: self.zoom_buttons(1 / 1.2))
            self.btn_zoom_out = fixed_button(QPushButton("-"))
            self.btn_zoom_out.clicked.connect(lambda: self.zoom_buttons(1.2))
            self.btn_zoom_reset = fixed_button(QPushButton("Reset Zoom"))
            self.btn_zoom_reset.clicked.connect(self.reset_zoom)

            for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_reset):
                b.setStyleSheet("QPushButton { min-height:32px; padding:4px 8px; }")

            hz_top.addWidget(self.btn_zoom_in)
            hz_top.addWidget(self.btn_zoom_out)
            hz_top.addWidget(self.btn_zoom_reset)
            hz_top.addStretch()
            z_layout.addLayout(hz_top)

            self.pan_slider = QSlider(Qt.Horizontal)
            self.pan_slider.setRange(0, 1000)
            self.pan_slider.setEnabled(False)
            self.pan_slider.valueChanged.connect(self.on_pan_slider)
            self.pan_slider.setMinimumHeight(16)
            z_layout.addWidget(QLabel("Horizontal Pan"))
            z_layout.addWidget(self.pan_slider)

            pan_hint = QLabel(
                "Right click + drag: pan | Right click: menu | Mouse wheel: zoom | Shift+scroll: pan")
            pan_hint.setWordWrap(True)
            pan_hint.setStyleSheet("color:#666")
            z_layout.addWidget(pan_hint)
            return g

        zoom_group = group_zoom()

        # --- Range Selection Group ---
        selection_group, selection_layout = groupbox("⏰ Range Selection and Playback")
        sel_form = QFormLayout()
        sel_form.setLabelAlignment(Qt.AlignLeft)
        sel_form.setFormAlignment(Qt.AlignLeft)

        self.sld_start = QSlider(Qt.Horizontal)
        self.sld_start.setRange(0, 0)
        self.sld_start.valueChanged.connect(self.update_time_from_slider)
        self.sld_start.sliderMoved.connect(self.update_time_from_slider)
        self.sld_start.setMinimumHeight(18)

        self.sld_end = QSlider(Qt.Horizontal)
        self.sld_end.setRange(0, 0)
        self.sld_end.valueChanged.connect(self.update_time_from_slider)
        self.sld_end.sliderMoved.connect(self.update_time_from_slider)
        self.sld_end.setMinimumHeight(18)

        sel_form.addRow(QLabel("Start:"), self.sld_start)
        sel_form.addRow(QLabel("End:"), self.sld_end)
        selection_layout.addLayout(sel_form)

        info_row = QHBoxLayout()
        self.lbl_start = QLabel("Start: 0.000s")
        self.lbl_end = QLabel("End: 0.000s")
        for lbl in (self.lbl_start, self.lbl_end):
            lbl.setStyleSheet("font-weight:bold;")
        info_row.addWidget(self.lbl_start)
        info_row.addStretch()
        info_row.addWidget(self.lbl_end)
        selection_layout.addLayout(info_row)

        selection_btn_layout = QHBoxLayout()
        selection_btn_layout.setSpacing(6)
        action_btn_style = "QPushButton { font-size:16px; padding:8px 14px; min-height:32px; font-weight:bold; color:white; }"

        self.btn_play_selection = fixed_button(QPushButton("▶ Play"))
        self.btn_play_selection.clicked.connect(self.toggle_selection_play)
        self.btn_play_selection.setEnabled(False)
        self.btn_play_selection.setStyleSheet(action_btn_style + "QPushButton { background:#4CAF50; }")

        self.btn_loop_selection = fixed_button(QPushButton("🔁 Loop"))
        self.btn_loop_selection.clicked.connect(self.loop_selection)
        self.btn_loop_selection.setEnabled(False)
        self.btn_loop_selection.setStyleSheet(action_btn_style + "QPushButton { background:#2196F3; }")

        self.btn_stop_selection = fixed_button(QPushButton("⏹ Stop"))
        self.btn_stop_selection.clicked.connect(self.stop_selection)
        self.btn_stop_selection.setEnabled(False)
        self.btn_stop_selection.setStyleSheet(action_btn_style + "QPushButton { background:#f44336; }")

        selection_btn_layout.addWidget(self.btn_play_selection)
        selection_btn_layout.addWidget(self.btn_loop_selection)
        selection_btn_layout.addWidget(self.btn_stop_selection)
        selection_btn_layout.addStretch()
        selection_layout.addLayout(selection_btn_layout)

        self.btn_clear = fixed_button(QPushButton("🗑️ Clear Selection"))
        self.btn_clear.clicked.connect(self.clear_selection)
        self.btn_clear.setEnabled(False)
        self.btn_clear.setStyleSheet("QPushButton{min-height:32px;}")
        selection_layout.addWidget(self.btn_clear, alignment=Qt.AlignLeft)

        self.lbl_selection_duration = QLabel("No selection")
        self.lbl_selection_duration.setWordWrap(True)
        selection_layout.addWidget(self.lbl_selection_duration)

        # --- Detection Group ---
        detection_group, detection_layout = groupbox("🔍 Advanced Bird Call Detection")

        param_form = QFormLayout()

        self.sld_sensitivity = QSlider(Qt.Horizontal)
        self.sld_sensitivity.setRange(1, 10)
        self.sld_sensitivity.setValue(2)
        self.sld_sensitivity.valueChanged.connect(self.update_detection_params)
        param_form.addRow(QLabel("Sensitivity (x0.1):"), self.sld_sensitivity)

        self.sld_min_duration = QSlider(Qt.Horizontal)
        self.sld_min_duration.setRange(1, 50)
        self.sld_min_duration.setValue(8)
        self.sld_min_duration.valueChanged.connect(self.update_detection_params)
        param_form.addRow(QLabel("Min. Duration (x0.1s):"), self.sld_min_duration)

        detection_layout.addLayout(param_form)

        detection_btn_layout = QHBoxLayout()
        self.btn_detect_bird_calls = fixed_button(QPushButton("🐦 Detect Bird Calls"))
        self.btn_detect_bird_calls.clicked.connect(self.detect_bird_calls)
        self.btn_detect_bird_calls.setEnabled(False)
        self.btn_detect_bird_calls.setStyleSheet(
            "QPushButton { background:#9C27B0; color:white; font-weight:bold; min-height:32px; }")

        self.btn_clear_detection = fixed_button(QPushButton("🗑️ Clear Detection"))
        self.btn_clear_detection.clicked.connect(self.clear_bird_call_detection)
        self.btn_clear_detection.setEnabled(False)
        self.btn_clear_detection.setStyleSheet("QPushButton{min-height:32px;}")

        detection_btn_layout.addWidget(self.btn_detect_bird_calls)
        detection_btn_layout.addWidget(self.btn_clear_detection)
        detection_btn_layout.addStretch()
        detection_layout.addLayout(detection_btn_layout)

        self.lbl_detection_status = QLabel("No detection performed")
        self.lbl_detection_status.setWordWrap(True)
        self.lbl_detection_status.setStyleSheet("QLabel { background:#f0f0f0; padding:6px 10px; border-radius:4px; }")
        detection_layout.addWidget(self.lbl_detection_status)

        detection_layout.addWidget(QLabel("Detected Bird Calls:"))
        self.detected_bird_calls_table = QTableWidget()

        self.detected_bird_calls_table.setColumnCount(6)
        self.detected_bird_calls_table.setHorizontalHeaderLabels([
            "Seg. ID", "Group ID", "Start (s)", "End (s)", "Duration (s)", "Avg. Conf."
        ])

        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.detected_bird_calls_table.setSortingEnabled(True)
        self.detected_bird_calls_table.cellClicked.connect(self.on_bird_call_item_clicked)
        self.detected_bird_calls_table.setMaximumHeight(150)
        detection_layout.addWidget(self.detected_bird_calls_table)

        # --- Dataset Operations Group ---
        action_group, action_layout = groupbox("🏷️ Dataset Operations")
        tabs = QTabWidget()

        # Species & Behavior Tab
        species_tab = QWidget()
        species_layout = QVBoxLayout(species_tab)
        call_and_type_layout = QHBoxLayout()

        call_and_type_layout.addWidget(QLabel("Call Type:"))
        self.cmb_call_type = QComboBox()
        self.cmb_call_type.addItems(self.config_data.get("call_typesENG", []))
        self.cmb_call_type.setMaximumWidth(200)
        call_and_type_layout.addWidget(self.cmb_call_type)

        call_and_type_layout.addSpacing(30)

        call_and_type_layout.addWidget(QLabel("Species Count:"))
        self.cmb_type_count = QComboBox()
        self.cmb_type_count.addItems(["Single", "Multiple", "Unknown"])
        self.cmb_type_count.setMaximumWidth(200)
        self.cmb_type_count.currentIndexChanged.connect(self.update_background_widgets_state)
        call_and_type_layout.addWidget(self.cmb_type_count)

        call_and_type_layout.addStretch()
        species_layout.addLayout(call_and_type_layout)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #cccccc;")
        species_layout.addWidget(line1)

        primary_title = QLabel("Primary Species")
        primary_title.setStyleSheet("font-weight: bold;")
        species_layout.addWidget(primary_title)

        primary_layout = QHBoxLayout()
        primary_layout.addWidget(QLabel("Turkish Name:"))
        self.cmb_turkish_name = QComboBox()
        turkish_names = sorted(list(set([bird["turkish_name"] for bird in self.bird_species_data])))
        self.cmb_turkish_name.addItems([""] + turkish_names)
        self.cmb_turkish_name.currentTextChanged.connect(self.on_turkish_name_changed)
        self.cmb_turkish_name.setMaximumWidth(200)
        self.cmb_turkish_name.setEditable(True)
        primary_layout.addWidget(self.cmb_turkish_name)

        primary_layout.addWidget(QLabel("Scientific Name:"))
        self.txt_scientific_name = QLineEdit()
        self.txt_scientific_name.setReadOnly(True)
        self.txt_scientific_name.setMaximumWidth(200)
        primary_layout.addWidget(self.txt_scientific_name)

        primary_layout.addWidget(QLabel("Family:"))
        self.txt_family = QLineEdit()
        self.txt_family.setReadOnly(True)
        self.txt_family.setMaximumWidth(200)
        primary_layout.addWidget(self.txt_family)

        primary_layout.addWidget(QLabel("Order:"))
        self.txt_order = QLineEdit()
        self.txt_order.setReadOnly(True)
        self.txt_order.setMaximumWidth(200)
        primary_layout.addWidget(self.txt_order)

        primary_layout.addStretch()
        species_layout.addLayout(primary_layout)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #cccccc;")
        species_layout.addWidget(line2)

        background_title = QLabel("Background Species Information")
        background_title.setStyleSheet("font-weight: bold;")
        species_layout.addWidget(background_title)

        background_layout = QHBoxLayout()
        background_layout.addWidget(QLabel("Bg English Name:"))
        self.cmb_background_turkish_name = QComboBox()
        self.cmb_background_turkish_name.addItems([""] + turkish_names)
        self.cmb_background_turkish_name.currentTextChanged.connect(self.on_background_turkish_name_changed)
        self.cmb_background_turkish_name.setMaximumWidth(200)
        self.cmb_background_turkish_name.setEditable(True)
        background_layout.addWidget(self.cmb_background_turkish_name)

        background_layout.addWidget(QLabel("Bg Scientific:"))
        self.txt_background_scientific_name = QLineEdit()
        self.txt_background_scientific_name.setReadOnly(True)
        self.txt_background_scientific_name.setMaximumWidth(200)
        background_layout.addWidget(self.txt_background_scientific_name)

        background_layout.addWidget(QLabel("Bg Family:"))
        self.txt_background_family = QLineEdit()
        self.txt_background_family.setReadOnly(True)
        self.txt_background_family.setMaximumWidth(200)
        background_layout.addWidget(self.txt_background_family)

        background_layout.addWidget(QLabel("Bg Order:"))
        self.txt_background_order = QLineEdit()
        self.txt_background_order.setReadOnly(True)
        self.txt_background_order.setMaximumWidth(400)
        background_layout.addWidget(self.txt_background_order)

        background_layout.addStretch()
        species_layout.addLayout(background_layout)

        other_species_layout = QHBoxLayout()
        other_species_layout.addWidget(QLabel("Other Species Notes:"))
        self.txt_other_species = QTextEdit()
        self.txt_other_species.setMaximumHeight(25)
        self.txt_other_species.setPlaceholderText("Notes for species not in the list...")
        self.txt_other_species.setMinimumWidth(200)
        other_species_layout.addWidget(self.txt_other_species)
        other_species_layout.addStretch()
        species_layout.addLayout(other_species_layout)

        line3 = QFrame()
        line3.setFrameShape(QFrame.HLine)
        line3.setStyleSheet("background-color: #cccccc;")
        species_layout.addWidget(line3)

        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Background Noise Type:"))
        self.cmb_background_noise = QComboBox()
        self.cmb_background_noise.addItems([""] + self.config_data.get("background_noiseENG", []))
        self.cmb_background_noise.setEditable(True)
        self.cmb_background_noise.setMaximumWidth(200)
        noise_layout.addWidget(self.cmb_background_noise)
        noise_layout.addStretch()
        species_layout.addLayout(noise_layout)

        species_layout.addStretch()

        # Location & Habitat Tab
        location_tab = QWidget()
        location_layout = QVBoxLayout(location_tab)
        location_form = QFormLayout()
        location_form.setLabelAlignment(Qt.AlignLeft)
        location_form.setFormAlignment(Qt.AlignLeft)

        self.cmb_location = QComboBox()
        locations = sorted(list(set([h.get("konum", "") for h in self.habitat_data if h.get("konum", "")])))
        self.cmb_location.addItems([""] + locations)
        self.cmb_location.currentTextChanged.connect(self.on_konum_changed)
        self.cmb_location.setMaximumWidth(250)
        self.cmb_location.setEditable(True)
        location_form.addRow(QLabel("Location:"), self.cmb_location)

        self.cmb_habitat_id = QComboBox()
        self.cmb_habitat_id.addItem("")
        for habitat in self.habitat_data:
            self.cmb_habitat_id.addItem(str(habitat["habitat_ID"]), habitat)
        self.cmb_habitat_id.currentIndexChanged.connect(self.on_habitat_id_changed)
        self.cmb_habitat_id.setMaximumWidth(200)
        location_form.addRow(QLabel("Habitat ID:"), self.cmb_habitat_id)

        self.txt_habitat = QLineEdit()
        self.txt_habitat.setReadOnly(True)
        self.txt_habitat.setPlaceholderText("Habitat type...")
        location_form.addRow(QLabel("Habitat Type:"), self.txt_habitat)

        self.txt_kod = QLineEdit()
        self.txt_kod.setReadOnly(True)
        location_form.addRow(QLabel("Code:"), self.txt_kod)

        self.txt_utm_zone = QLineEdit()
        self.txt_utm_zone.setReadOnly(True)
        location_form.addRow(QLabel("UTM Zone:"), self.txt_utm_zone)

        utm_layout = QHBoxLayout()
        self.txt_utm_easting = QLineEdit()
        self.txt_utm_easting.setReadOnly(True)
        self.txt_utm_northing = QLineEdit()
        self.txt_utm_northing.setReadOnly(True)
        utm_layout.addWidget(QLabel("Easting:"))
        utm_layout.addWidget(self.txt_utm_easting)
        utm_layout.addWidget(QLabel("Northing:"))
        utm_layout.addWidget(self.txt_utm_northing)
        location_form.addRow(QLabel("UTM Coordinates:"), utm_layout)

        lat_lon_layout = QHBoxLayout()
        self.txt_lat = QLineEdit()
        self.txt_lat.setReadOnly(True)
        self.txt_lon = QLineEdit()
        self.txt_lon.setReadOnly(True)
        lat_lon_layout.addWidget(QLabel("Lat:"))
        lat_lon_layout.addWidget(self.txt_lat)
        lat_lon_layout.addWidget(QLabel("Lon:"))
        lat_lon_layout.addWidget(self.txt_lon)
        location_form.addRow(QLabel("Coordinates:"), lat_lon_layout)

        maps_layout = QHBoxLayout()
        self.txt_google_maps = QLineEdit()
        self.txt_google_maps.setReadOnly(True)
        self.btn_open_maps = QPushButton("🌍 Open in Maps")
        self.btn_open_maps.clicked.connect(self.open_google_maps)
        self.btn_open_maps.setEnabled(False)
        maps_layout.addWidget(self.txt_google_maps, 4)
        maps_layout.addWidget(self.btn_open_maps, 1)
        location_form.addRow(QLabel("Google Maps:"), maps_layout)

        location_layout.addLayout(location_form)
        location_layout.addStretch()

        # People & Verification Tab
        verification_tab = QWidget()
        verification_layout = QVBoxLayout(verification_tab)
        verification_form = QFormLayout()
        verification_form.setLabelAlignment(Qt.AlignLeft)
        verification_form.setFormAlignment(Qt.AlignLeft)

        self.cmb_recordist = QComboBox()
        self.cmb_recordist.addItems(self.config_data.get("recordistsENG", []))
        self.cmb_recordist.setEditable(True)
        verification_form.addRow(QLabel("Recorded by:"), self.cmb_recordist)

        self.cmb_ornithologist = QComboBox()
        self.cmb_ornithologist.addItems(self.config_data.get("ornithologistsENG", []))
        self.cmb_ornithologist.setEditable(True)
        verification_form.addRow(QLabel("Ornithologist:"), self.cmb_ornithologist)

        self.cmb_verification_status = QComboBox()
        self.cmb_verification_status.addItems(self.config_data.get("verification_statusesENG", []))
        verification_form.addRow(QLabel("Verification Status:"), self.cmb_verification_status)

        self.spn_confidence = QSpinBox()
        self.spn_confidence.setRange(0, 100)
        self.spn_confidence.setValue(100)
        self.spn_confidence.setSuffix("%")
        verification_form.addRow(QLabel("Confidence Level:"), self.spn_confidence)

        verification_layout.addLayout(verification_form)
        verification_layout.addStretch()

        # Equipment Tab
        equipment_tab = QWidget()
        equipment_layout = QVBoxLayout(equipment_tab)
        equipment_form = QFormLayout()
        equipment_form.setLabelAlignment(Qt.AlignLeft)
        equipment_form.setFormAlignment(Qt.AlignLeft)

        self.cmb_microphone = QComboBox()
        self.cmb_microphone.addItems(self.config_data.get("microphonesENG", []))
        self.cmb_microphone.setEditable(True)
        equipment_form.addRow(QLabel("Microphone:"), self.cmb_microphone)

        self.cmb_recorder = QComboBox()
        self.cmb_recorder.addItems(self.config_data.get("recordersENG", []))
        self.cmb_recorder.setEditable(True)
        equipment_form.addRow(QLabel("Recorder:"), self.cmb_recorder)

        equipment_layout.addLayout(equipment_form)

        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel("Project:"))
        self.txt_project = QLineEdit()
        self.txt_project.setText(self.config_data.get("projectENG"))
        self.btn_save_project = QPushButton("💾 Save Project Name")
        self.btn_save_project.clicked.connect(lambda: self.update_project_info(self.txt_project.text()))
        project_layout.addWidget(self.txt_project, 4)
        project_layout.addWidget(self.btn_save_project, 1)
        equipment_layout.addLayout(project_layout)

        equipment_layout.addWidget(QLabel("Notes:"))
        self.txt_notes = QTextEdit()
        self.txt_notes.setMaximumHeight(80)
        self.txt_notes.setPlaceholderText("Observation notes, behaviors, etc...")
        equipment_layout.addWidget(self.txt_notes)

        equipment_layout.addStretch()

        tabs.addTab(species_tab, "🐦 Species & Behavior")
        tabs.addTab(location_tab, "🌍 Location & Habitat")
        tabs.addTab(verification_tab, "👥 People & Verification")
        tabs.addTab(equipment_tab, "🎙️ Equipment")

        action_layout.addWidget(tabs)

        self.btn_save = fixed_button(QPushButton("💾 Add to Dataset"))
        self.btn_save.clicked.connect(self.save_segment)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            "QPushButton { background:#FF9800; color:white; font-weight:bold; min-height:32px; }")
        action_layout.addWidget(self.btn_save, alignment=Qt.AlignLeft)

        self.lbl_segment_message = QLabel("")
        self.lbl_segment_message.setWordWrap(True)
        self.lbl_segment_message.setStyleSheet(
            "QLabel { background:#e8f5e8; padding:6px 10px; border-radius:4px; font-size: 14px; color: #2e7d32; }")
        action_layout.addWidget(self.lbl_segment_message)

        csv_row = QHBoxLayout()
        self.btn_view_csv = fixed_button(QPushButton("📊 View CSV"))
        self.btn_view_csv.clicked.connect(self.view_csv)
        self.btn_export_csv = fixed_button(QPushButton("📤 Export CSV"))
        self.btn_export_csv.clicked.connect(self.export_csv)
        csv_row.addWidget(self.btn_view_csv)
        csv_row.addWidget(self.btn_export_csv)
        csv_row.addStretch()
        action_layout.addLayout(csv_row)

        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(file_group, "🎵 Audio File")
        self.main_tabs.addTab(view_group, "👁️ View")
        self.main_tabs.addTab(playproc_group, "🎵 Playback & Processing")
        self.main_tabs.addTab(zoom_group, "🔍 Zoom")
        self.main_tabs.addTab(selection_group, "⏰ Selection")
        self.main_tabs.addTab(detection_group, "🔍 Bird Call Detection")
        self.main_tabs.addTab(action_group, "🏷️ Dataset Labeling")

        left_layout.addWidget(self.main_tabs)

        # Matplotlib Panel
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        self.figure = plt.figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        top_layout.addWidget(self.canvas)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top_panel)
        splitter.addWidget(self.controls_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([int(h * 0.45), int(h * 0.55)])
        main_layout.addWidget(splitter)

        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_scroll = self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.player = QMediaPlayer()
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(self.update_duration)

        self.selection_player = QMediaPlayer()
        self.selection_player.stateChanged.connect(self.on_selection_player_state_changed)
        self.selection_player.positionChanged.connect(self.update_selection_position)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)

        self.btn_run_process.clicked.connect(self.run_processing_pipeline)
        self.btn_save_full.clicked.connect(self.save_full_audio_to_wav)

        self.setFocusPolicy(Qt.StrongFocus)
        self.apply_compact_mode(True)

        if nr is None:
            self.chk_denoise.setEnabled(False)
            self.chk_denoise.setChecked(False)
        if pyln is None:
            self.chk_lufs.setEnabled(False)
            self.chk_lufs.setChecked(False)
        if librosa is None:
            self.btn_detect_bird_calls.setEnabled(False)

        self.create_context_menu()
        self.show()
        self.update_background_widgets_state()

    def update_background_widgets_state(self):
        """Activates/deactivates background species widgets based on Species Count selection."""
        current_type = self.cmb_type_count.currentText()
        primary_selected = bool(self.cmb_turkish_name.currentText().strip())

        background_species_widgets = [
            self.cmb_background_turkish_name,
            self.txt_background_scientific_name,
            self.txt_background_family,
            self.txt_background_order,
            self.txt_other_species
        ]

        background_noise_widgets = [
            self.cmb_background_noise
        ]

        if current_type == "Single" or not primary_selected:
            for widget in background_species_widgets:
                widget.setEnabled(False)
                if widget == self.cmb_background_turkish_name:
                    widget.setCurrentText("")
                    self.txt_background_scientific_name.clear()
                    self.txt_background_family.clear()
                    self.txt_background_order.clear()
                elif widget == self.txt_other_species:
                    widget.clear()

            for widget in background_noise_widgets:
                widget.setEnabled(True)
                if widget == self.cmb_background_noise:
                    widget.setCurrentText("")
        else:
            for widget in background_species_widgets:
                widget.setEnabled(True)
            for widget in background_noise_widgets:
                widget.setEnabled(True)
