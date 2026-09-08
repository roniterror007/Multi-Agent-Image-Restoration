import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore
import qdarkstyle

def apply_shifts_numpy(img_bgr, shifts):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] += shifts[0]
    lab[:, :, 1] += shifts[1]
    lab[:, :, 2] += shifts[2]
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def bgr_to_qimage(bgr_img):
    h, w, c = bgr_img.shape
    bytes_per_line = 3 * w
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    qim = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    return qim

class ReviewApp(QtWidgets.QDialog):
    def __init__(self, original_pil, cleaned_pil, filename):
        super().__init__()
        self.filename = filename
        
        # Keep original image in BGR format for CV2 manipulation
        self.original_bgr = cv2.cvtColor(np.array(original_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.cleaned_bgr = cv2.cvtColor(np.array(cleaned_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        self.result = {
            "decision": "kill", # default if closed
            "shifts": (0, 0, 0),
            "issues": [],
            "comment": ""
        }
        
        self.is_custom_fix = False # tracks if user has touched sliders
        self.is_dark_mode = True # Tracks theme state
        
        self.initUI()
        self.update_preview()
        
    def initUI(self):
        self.setWindowTitle(f"Antigravity Interactive Review - {self.filename}")
        self.resize(1300, 750)
        
        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left Panel - Images
        images_layout = QtWidgets.QVBoxLayout()
        
        # Original Image
        self.lbl_original = QtWidgets.QLabel()
        self.lbl_original.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_original.setMinimumSize(500, 500)
        
        # Preview Image
        self.lbl_preview = QtWidgets.QLabel()
        self.lbl_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(500, 500)
        
        img_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        frame_orig = QtWidgets.QFrame()
        layout_orig = QtWidgets.QVBoxLayout(frame_orig)
        layout_orig.addWidget(QtWidgets.QLabel("<b>Original Image</b>"))
        layout_orig.addWidget(self.lbl_original)
        
        frame_prev = QtWidgets.QFrame()
        layout_prev = QtWidgets.QVBoxLayout(frame_prev)
        self.lbl_preview_title = QtWidgets.QLabel("<b>Pipeline Output (Cleaned)</b>")
        layout_prev.addWidget(self.lbl_preview_title)
        layout_prev.addWidget(self.lbl_preview)
        
        img_splitter.addWidget(frame_orig)
        img_splitter.addWidget(frame_prev)
        
        images_layout.addWidget(img_splitter)
        main_layout.addLayout(images_layout, stretch=4)
        
        # Right Panel - Controls
        controls_layout = QtWidgets.QVBoxLayout()
        
        # Sliders
        controls_layout.addWidget(QtWidgets.QLabel("<h3>Manual Fix (Lab Space)</h3>"))
        controls_layout.addWidget(QtWidgets.QLabel("<i>Moving these sliders will override the pipeline and generate a custom manual fix.</i>", wordWrap=True))
        controls_layout.addSpacing(10)
        
        self.lbl_val_l = QtWidgets.QLabel("0")
        self.slider_l = self.create_slider("Lightness (L)", -100, 100, 0, controls_layout, self.lbl_val_l)
        
        self.lbl_val_a = QtWidgets.QLabel("0")
        self.slider_a = self.create_slider("Green-Red (a*)", -100, 100, 0, controls_layout, self.lbl_val_a)
        
        self.lbl_val_b = QtWidgets.QLabel("0")
        self.slider_b = self.create_slider("Blue-Yellow (b*)", -100, 100, 0, controls_layout, self.lbl_val_b)
        
        btn_reset = QtWidgets.QPushButton("Revert to Pipeline Output")
        btn_reset.clicked.connect(self.reset_sliders)
        controls_layout.addWidget(btn_reset)
        
        controls_layout.addSpacing(30)
        controls_layout.addWidget(QtWidgets.QLabel("<h3>Actions</h3>"))
        
        btn_accept = QtWidgets.QPushButton("✅ Accept Current Preview")
        btn_accept.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 12px; font-size: 14px;")
        btn_accept.clicked.connect(self.on_accept)
        controls_layout.addWidget(btn_accept)
        
        btn_manual = QtWidgets.QPushButton("⚠️ Escalate for Manual Review")
        btn_manual.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold; padding: 12px; font-size: 14px;")
        btn_manual.clicked.connect(self.on_manual_review)
        controls_layout.addWidget(btn_manual)
        
        btn_kill = QtWidgets.QPushButton("❌ Kill Pipeline")
        btn_kill.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 12px; font-size: 14px;")
        btn_kill.clicked.connect(self.on_kill)
        controls_layout.addWidget(btn_kill)
        
        controls_layout.addSpacing(10)
        btn_theme = QtWidgets.QPushButton("🌗 Toggle Theme (T)")
        btn_theme.setStyleSheet("padding: 8px; font-size: 12px;")
        btn_theme.clicked.connect(self.toggle_theme)
        controls_layout.addWidget(btn_theme)
        
        QtWidgets.QShortcut(QtGui.QKeySequence("T"), self).activated.connect(self.toggle_theme)
        
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout, stretch=1)
        
        # Set images
        self.qimg_orig = bgr_to_qimage(self.original_bgr)
        self.qimg_cleaned = bgr_to_qimage(self.cleaned_bgr)
        
    def create_slider(self, name, min_val, max_val, default, parent_layout, lbl_val):
        layout = QtWidgets.QVBoxLayout()
        header = QtWidgets.QHBoxLayout()
        lbl_name = QtWidgets.QLabel(name)
        lbl_val.setStyleSheet("font-weight: bold;")
        header.addWidget(lbl_name)
        header.addStretch()
        header.addWidget(lbl_val)
        layout.addLayout(header)
        
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        
        def on_change(val):
            self.is_custom_fix = True
            self.lbl_preview_title.setText("<b>Custom Live Fix Preview</b>")
            self.lbl_preview_title.setStyleSheet("color: #4caf50;")
            lbl_val.setText(str(val))
            self.update_preview()
            
        slider.valueChanged.connect(on_change)
        layout.addWidget(slider)
        parent_layout.addLayout(layout)
        return slider

    def reset_sliders(self):
        self.is_custom_fix = False
        self.lbl_preview_title.setText("<b>Pipeline Output (Cleaned)</b>")
        self.lbl_preview_title.setStyleSheet("")
        
        # Disconnect momentarily to avoid triggering update_preview 3 times
        self.slider_l.blockSignals(True)
        self.slider_a.blockSignals(True)
        self.slider_b.blockSignals(True)
        
        self.slider_l.setValue(0)
        self.slider_a.setValue(0)
        self.slider_b.setValue(0)
        
        self.slider_l.blockSignals(False)
        self.slider_a.blockSignals(False)
        self.slider_b.blockSignals(False)
        
        # Manually update labels
        self.lbl_val_l.setText("0")
        self.lbl_val_a.setText("0")
        self.lbl_val_b.setText("0")
        
        self.update_preview()
        
    def update_preview(self):
        w = self.lbl_original.width()
        h = self.lbl_original.height()
        if w < 10 or h < 10:
            return
            
        pixmap_orig = QtGui.QPixmap.fromImage(self.qimg_orig)
        self.lbl_original.setPixmap(pixmap_orig.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        
        if not self.is_custom_fix:
            pixmap_prev = QtGui.QPixmap.fromImage(self.qimg_cleaned)
            self.lbl_preview.setPixmap(pixmap_prev.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            return
            
        # Apply custom shift
        l = self.slider_l.value()
        a = self.slider_a.value()
        b = self.slider_b.value()
        
        shifted_bgr = apply_shifts_numpy(self.original_bgr, (l, a, b))
        qimg_prev = bgr_to_qimage(shifted_bgr)
        pixmap_prev = QtGui.QPixmap.fromImage(qimg_prev)
        self.lbl_preview.setPixmap(pixmap_prev.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        
    def resizeEvent(self, event):
        self.update_preview()
        super().resizeEvent(event)
        
    def on_accept(self):
        self.result["decision"] = "accepted"
        if self.is_custom_fix:
            self.result["shifts"] = (self.slider_l.value(), self.slider_a.value(), self.slider_b.value())
            self.result["is_custom"] = True
        else:
            self.result["is_custom"] = False
        self.accept()
        
    def on_manual_review(self):
        self.result["decision"] = "manual_review"
        self.accept()
        
    def on_kill(self):
        self.result["decision"] = "kill"
        self.reject()

    def toggle_theme(self):
        app = QtWidgets.QApplication.instance()
        if self.is_dark_mode:
            app.setStyleSheet("")
            self.is_dark_mode = False
        else:
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
            self.is_dark_mode = True

def review_image(original, cleaned, filename, feedback_dir, route, attempt=1):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    
    dialog = ReviewApp(original, cleaned, filename)
    dialog.exec_()
    
    return dialog.result
