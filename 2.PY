import sys
import cv2
import torch
import numpy as np
import pyautogui
import pygetwindow as gw
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QFileDialog,
                            QProgressBar, QSizePolicy, QGroupBox,
                            QTabWidget, QInputDialog)
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
        self.setWindowTitle('测试应用')
        self.setMinimumSize(1200, 800)

        # 创建主选项卡
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 图像处理标签页
        image_tab = self.create_image_tab()
        # 实时摄像头标签页
        camera_tab = self.create_camera_tab()
        # 视频处理标签页
        video_tab = self.create_video_tab()
        # 窗口捕获标签页
        window_tab = self.create_window_tab()

        self.tabs.addTab(image_tab, "图像处理")
        self.tabs.addTab(camera_tab, "实时摄像头")
        self.tabs.addTab(video_tab, "视频分析")
        self.tabs.addTab(window_tab, "窗口捕获")

        # 初始化状态栏
        self.status_bar = self.statusBar()
        self.progress = QProgressBar()
        self.progress.hide()
        self.status_bar.addPermanentWidget(self.progress)

    def create_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 控制面板
        control_group = QGroupBox("图像处理控制")
        control_layout = QHBoxLayout(control_group)

        self.model_btn = QPushButton("加载YOLO模型")
        self.model_btn.clicked.connect(self.load_model)
        self.model_label = QLabel("当前模型：未加载")
        
        control_layout.addWidget(self.model_btn)
        control_layout.addWidget(self.model_label)
        control_layout.addStretch()

        # 显示区域
        display_layout = QHBoxLayout()
        self.original_view = ImageViewer("拖放图片至此\n或点击选择文件")
        self.original_view.mousePressEvent = self.open_image_file
        self.result_view = ImageViewer("检测结果预览")
        display_layout.addWidget(self.original_view, 3)
        display_layout.addWidget(self.result_view, 3)

        layout.addWidget(control_group)
        layout.addLayout(display_layout)
        return tab

    def create_camera_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.camera_view = ImageViewer("摄像头预览")
        self.camera_btn = QPushButton("启动摄像头")
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        layout.addWidget(self.camera_view)
        layout.addWidget(self.camera_btn)
        
        # 摄像头相关初始化
        self.camera_active = False
        self.capture = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        return tab

    def create_video_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.video_view = ImageViewer("视频预览")
        
        # 视频控制栏
        control_layout = QHBoxLayout()
        self.video_play_btn = QPushButton("打开视频")
        self.video_play_btn.clicked.connect(self.toggle_video_play)
        self.video_stop_btn = QPushButton("停止")
        self.video_stop_btn.clicked.connect(self.stop_video)
        self.video_progress = QProgressBar()
        
        control_layout.addWidget(self.video_play_btn)
        control_layout.addWidget(self.video_stop_btn)
        control_layout.addWidget(self.video_progress)
        
        layout.addWidget(self.video_view)
        layout.addLayout(control_layout)
        
        # 视频相关初始化
        self.video_active = False
        self.video_capture = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        return tab

    def create_window_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.window_view = ImageViewer("窗口检测预览")
        self.window_btn = QPushButton("选择窗口")
        self.window_btn.clicked.connect(self.select_window)
        
        layout.addWidget(self.window_view)
        layout.addWidget(self.window_btn)
        
        # 窗口捕获相关初始化
        self.window_active = False
        self.target_window = None
        self.window_timer = QTimer()
        self.window_timer.timeout.connect(self.update_window_frame)
        return tab

    # ========== 核心功能方法 ==========
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pt)"
        )
        if path:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = YOLO(path).to(device)
                self.model_label.setText(f"当前模型：{path.split('/')[-1]}")
                self.status_bar.showMessage(f"模型加载成功 | 设备：{device.upper()}")
            except Exception as e:
                self.status_bar.showMessage(f"模型加载失败: {str(e)}")

    def toggle_camera(self):
        if not self.camera_active:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.status_bar.showMessage("无法打开摄像头")
                return
            self.camera_active = True
            self.camera_btn.setText("关闭摄像头")
            self.camera_timer.start(30)
        else:
            self.camera_active = False
            self.camera_timer.stop()
            self.capture.release()
            self.camera_btn.setText("启动摄像头")
            self.camera_view.setImage("")

    def update_camera_frame(self):
        if self.camera_active:
            ret, frame = self.capture.read()
            if ret:
                self.process_realtime_frame(frame, self.camera_view)

    def toggle_video_play(self):
        if not self.video_active:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)"
            )
            if path:
                self.video_capture = cv2.VideoCapture(path)
                if not self.video_capture.isOpened():
                    self.status_bar.showMessage("无法打开视频文件")
                    return
                self.video_active = True
                self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_progress.setMaximum(self.total_frames)
                self.video_timer.start(30)
                self.video_play_btn.setText("暂停")

    def stop_video(self):
        self.video_active = False
        if self.video_capture:
            self.video_capture.release()
        self.video_timer.stop()
        self.video_progress.reset()
        self.video_view.setImage("")
        self.video_play_btn.setText("打开视频")

    def update_video_frame(self):
        if self.video_active and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                self.video_progress.setValue(current_frame)
                self.process_realtime_frame(frame, self.video_view)
            else:
                self.stop_video()

    def select_window(self):
        # 模型检查
        if not self.model:
            self.status_bar.showMessage("请先加载模型再进行窗口捕获", 5000)
            return
        
        windows = gw.getAllTitles()
        valid_windows = [title for title in windows if title.strip() != '']
        
        if not valid_windows:
            self.status_bar.showMessage("没有找到可用窗口")
            return
        
        window, ok = QInputDialog.getItem(
            self, "选择窗口", "请选择要捕获的窗口:", valid_windows, 0, False
        )
        
        if ok and window:
            try:
                self.target_window = gw.getWindowsWithTitle(window)[0]
                self.window_active = True
                self.window_view.setText("正在捕获窗口...")
                self.window_timer.start(30)
                self.status_bar.showMessage(f"正在捕获窗口: {window}")
            except Exception as e:
                self.status_bar.showMessage(f"窗口选择失败: {str(e)}")

    def update_window_frame(self):
        if self.window_active and self.target_window:
            try:
                # 检查窗口是否仍然存在
                if not self.target_window.isActive:
                    raise Exception("目标窗口已关闭")
                
                # 获取窗口位置和尺寸
                left, top, width, height = (
                    self.target_window.left,
                    self.target_window.top,
                    self.target_window.width,
                    self.target_window.height
                )
                
                # 截屏并转换格式
                screen_shot = pyautogui.screenshot(region=(left, top, width, height))
                frame = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_RGB2BGR)
                
                # 进行目标检测
                results = self.model(frame)
                annotated_frame = results[0].plot()

                # 转换为QImage显示
                h, w, ch = annotated_frame.shape
                bytes_per_line = 3 * w
                q_img = QImage(annotated_frame.data, w, h, bytes_per_line, 
                             QImage.Format.Format_RGB888)
                self.window_view.setImage(q_img)
                
                # 更新状态信息
                self.status_bar.showMessage(f"窗口检测中 | 检测到{len(results[0])}个对象")
                
            except Exception as e:
                self.window_active = False
                self.window_timer.stop()
                self.status_bar.showMessage(f"窗口捕获错误: {str(e)}")
                self.window_view.setText("捕获异常，请重新选择窗口")

    def process_realtime_frame(self, frame, display_widget):
        """实时帧处理通用方法"""
        if not self.model:
            display_widget.setText("模型未加载")
            return
        
        try:
            results = self.model(frame)
            annotated_frame = results[0].plot()
            
            # 转换为QImage显示
            h, w, ch = annotated_frame.shape
            bytes_per_line = 3 * w
            q_img = QImage(annotated_frame.data, w, h, 
                         bytes_per_line, QImage.Format.Format_RGB888)
            display_widget.setImage(q_img)
            
            # 更新状态信息
            self.status_bar.showMessage(f"实时检测中 | 检测到{len(results[0])}个对象")
            
        except Exception as e:
            display_widget.setText("分析过程发生错误")
            self.status_bar.showMessage(f"分析错误: {str(e)}")

    def open_image_file(self, event=None):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", "图片文件 (*.png *.jpg *.jpeg)"
        )
        if path:
            self.process_image(path)

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
        self.result_view.setImage(q_img)
        self.status_bar.showMessage(f"检测完成 | 耗时: {elapsed:.2f}s | 检测到{len(results[0])}个对象")

    def handle_error(self, error_msg):
        self.progress.hide()
        self.status_bar.showMessage(f"错误: {error_msg}")

    # ========== 事件处理 ==========
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.process_image(file_path)
                break

    def closeEvent(self, event):
        # 释放所有资源
        if self.camera_active:
            self.capture.release()
        if self.video_active:
            self.video_capture.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageRecognizerApp()
    window.show()
    sys.exit(app.exec())
