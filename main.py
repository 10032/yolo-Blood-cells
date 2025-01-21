import sys
import cv2
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog,
                            QProgressBar, QListWidget, QSizePolicy, QGroupBox)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
from ultralytics import YOLO
import time

class WorkerThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, model, input_source):
        super().__init__()
        self.model = model
        self.input_source = input_source
        self.mutex = QMutex()
        self._is_running = True

    def run(self):
        try:
            if isinstance(self.input_source, str):
                start_time = time.time()
                results = self.model(self.input_source)
                annotated_image = results[0].plot(line_width=2)
                elapsed = time.time() - start_time
                self.finished.emit((annotated_image, elapsed, results))
        except Exception as e:
            self.error.emit(str(e))

class ImageViewer(QLabel):
    def __init__(self, prompt_text, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet('''
            QLabel {
                border: 2px dashed #aaa;
                background: #f8f9fa;
                border-radius: 8px;
            }
            QLabel:hover {
                border-color: #4a90e2;
            }
        ''')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText(prompt_text)
        self._pixmap = None

    def setImage(self, image):
        if isinstance(image, str):
            self._pixmap = QPixmap(image)
        elif isinstance(image, QImage):
            self._pixmap = QPixmap.fromImage(image)
        self.updateDisplay()
        self.setText("")

    def updateDisplay(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(self.size(), 
                                       Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        self.updateDisplay()
        super().resizeEvent(event)

class ImageRecognizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_source = None
        self.worker = None
        self.initUI()
        self.setAcceptDrops(True)

    def initUI(self):
        self.setWindowTitle('测试版')
        self.setMinimumSize(1000, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 顶部控制区
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout(control_group)

        # 模型控制
        model_group = QGroupBox("模型管理")
        model_layout = QVBoxLayout(model_group)
        self.model_btn = QPushButton("加载YOLO模型")
        self.model_btn.clicked.connect(self.load_model)  # 正确的连接方法
        self.model_label = QLabel("当前模型：未加载")
        model_layout.addWidget(self.model_btn)
        model_layout.addWidget(self.model_label)

        # 输入源控制
        input_group = QGroupBox("输入源")
        input_layout = QVBoxLayout(input_group)
        self.file_btn = QPushButton("打开文件")
        self.file_btn.clicked.connect(self.open_file)
        input_layout.addWidget(self.file_btn)

        control_layout.addWidget(model_group)
        control_layout.addWidget(input_group)
        control_layout.addStretch()

        # 显示区域
        display_layout = QHBoxLayout()
        self.original_view = ImageViewer("拖放图片至此\n或点击选择文件")
        self.original_view.mousePressEvent = self.open_file
        self.annotated_view = ImageViewer("检测结果预览")
        display_layout.addWidget(self.original_view, 3)
        display_layout.addWidget(self.annotated_view, 3)

        # 状态栏
        self.status_bar = self.statusBar()
        self.progress = QProgressBar()
        self.progress.hide()
        self.status_bar.addPermanentWidget(self.progress)

        main_layout.addWidget(control_group)
        main_layout.addLayout(display_layout)

        self.update_controls(False)

    def load_model(self):  # 添加缺失的方法定义
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pt)"
        )
        if path:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = YOLO(path).to(device)
                self.model_label.setText(f"当前模型：{path.split('/')[-1]}")
                self.update_controls(True)
                self.status_bar.showMessage(f"模型加载成功 | 设备：{device.upper()}")
            except Exception as e:
                self.status_bar.showMessage(f"模型加载失败: {str(e)}")

    def update_controls(self, enabled=True):
        self.file_btn.setEnabled(enabled)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.process_image(file_path)
                break

    def process_image(self, image_path):
        if not self.model:
            self.status_bar.showMessage("请先加载模型", 5000)
            return

        self.original_view.setImage(image_path)
        self.worker = WorkerThread(self.model, image_path)
        self.worker.finished.connect(self.handle_image_result)
        self.worker.error.connect(self.handle_error)
        self.worker.start()
        self.progress.setRange(0, 0)
        self.progress.show()

    def handle_image_result(self, data):
        annotated_image, elapsed, results = data
        self.progress.hide()
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = 3 * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.annotated_view.setImage(q_img)
        self.status_bar.showMessage(f"检测完成 | 耗时: {elapsed:.2f}s | 检测到{len(results[0])}个对象")

    def open_file(self, event=None):
        options = QFileDialog.Option.DontUseNativeDialog
        path, _ = QFileDialog.getOpenFileName(
            self, 
        "选择文件", 
        "", 
        "图片文件 (*.png *.jpg *.jpeg)",
        options=options  # 直接传递枚举值
    )
        if path:
            self.process_image(path)

    def handle_error(self, error_msg):
        self.progress.hide()
        self.status_bar.showMessage(f"错误: {error_msg}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageRecognizerApp()
    window.show()
    sys.exit(app.exec())