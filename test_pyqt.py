import sys
from PyQt5.QtWidgets import QApplication, QLabel

print("PyQt5 test starting...")

app = QApplication(sys.argv)
print("QApplication created")

label = QLabel("Hello PyQt5 - BioSeg Test")
label.resize(400, 300)
label.show()
print("Label shown")

print("Entering event loop...")
sys.exit(app.exec_())