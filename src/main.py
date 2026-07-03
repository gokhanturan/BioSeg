"""Main entry point for BioSeg Labeling Studio."""

import sys
import os
import matplotlib
# Set backend before importing PyQt5
matplotlib.use('Qt5Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from src.ui.main_window import BirdSoundApp


def main():
    """Initialize and run the BioSeg application."""
    # Enable high DPI scaling for better display on modern monitors
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = BirdSoundApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()