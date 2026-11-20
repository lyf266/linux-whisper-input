#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音输入控制台与设置面板 (Voice Input Settings GUI)
基于 PyQt6 开发，提供输入设备选择、实时麦克风电平测试、提示音调节与试听、后台服务监控与一键重启。
"""

import os
import sys
import json
import time
import subprocess
import threading
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen, QLinearGradient
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSlider,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QFrame,
    QScrollArea,
    QSizePolicy
)

# 确保能加载 daemon.py 中的辅助模块
DIR = os.path.dirname(os.path.abspath(__file__))
if DIR not in sys.path:
    sys.path.insert(0, DIR)

import daemon as daemon_mod

CONFIG_PATH = os.path.join(DIR, "config.json")
SOUNDS_DIR = os.path.join(DIR, "sounds")


class AudioMeterThread(QThread):
    """麦克风实时拾音电平采集线程"""
    level_changed = pyqtSignal(float)  # 0.0 ~ 1.0

    def __init__(self, source_name: str = ""):
        super().__init__()
        self.source_name = source_name.strip()
        self.running = False
        self._stream = None

    def run(self):
        self.running = True
        try:
            import sounddevice as sd

            if self.source_name:
                os.environ["PULSE_SOURCE"] = self.source_name
                dev = "pulse"
            else:
                if "PULSE_SOURCE" in os.environ:
                    del os.environ["PULSE_SOURCE"]
                dev = None

            def callback(indata, frames, time_info, status):
                if not self.running:
                    return
                # 计算 RMS 电平
                rms = np.sqrt(np.mean(indata ** 2))
                # 归一化为 0.0 ~ 1.0 (根据经验动态提升低音灵敏度)
                normalized = min(1.0, float(rms) * 4.0)
                self.level_changed.emit(normalized)

            with sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype=np.float32,
                blocksize=800,  # 50ms 刷新率
                callback=callback,
                device=dev
            ) as stream:
                self._stream = stream
                while self.running:
                    time.sleep(0.04)

        except Exception as e:
            print(f"AudioMeterThread error: {e}")
            self.level_changed.emit(0.0)
        finally:
            self._stream = None

    def stop(self):
        self.running = False
        self.wait(1000)


class VUMeterWidget(QWidget):
    """自定义渐变拾音电平条"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(24)
        self.setFixedHeight(24)
        self._level = 0.0  # 0.0 ~ 1.0
        self._target_level = 0.0
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._step_decay)
        self._anim_timer.start(25)

    def set_level(self, val: float):
        self._target_level = max(0.0, min(1.0, val))
        if self._target_level > self._level:
            self._level = self._target_level  # 瞬态峰值立即响应
        self.update()

    def _step_decay(self):
        if self._level > self._target_level:
            self._level = max(self._target_level, self._level - 0.05)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        radius = 6.0

        # 背景槽
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(40, 44, 52, 160))
        painter.drawRoundedRect(0, 0, width, height, radius, radius)

        # 进度填充
        fill_width = int(width * self._level)
        if fill_width > 2:
            gradient = QLinearGradient(0, 0, width, 0)
            gradient.setColorAt(0.0, QColor(46, 204, 113))   # 绿色 (正常输入)
            gradient.setColorAt(0.65, QColor(241, 196, 15))  # 黄色 (饱满音量)
            gradient.setColorAt(0.90, QColor(231, 76, 60))   # 红色 (过载防爆音)

            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(0, 0, fill_width, height, radius, radius)

        # 刻度网格线与安全指示
        painter.setPen(QPen(QColor(255, 255, 255, 50), 1))
        for mark in (0.25, 0.50, 0.75):
            x = int(width * mark)
            painter.drawLine(x, 2, x, height - 3)


class SettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音输入控制台 (Voice Input Settings)")
        self.setMinimumSize(680, 740)
        self.resize(700, 800)

        self.meter_thread = None
        self.config_data = daemon_mod.DEFAULT_CONFIG.copy()
        self.load_current_config()

        # 统一字体（思源黑体 / Noto Sans CJK SC 优先）
        font = QFont("Source Han Sans SC", 10)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        self.setFont(font)

        self.init_ui()
        self.refresh_service_status()
        self.start_vu_meter()

        # 定时刷新服务状态
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.refresh_service_status)
        self.status_timer.start(5000)

    def load_current_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    self.config_data.update(json.load(f))
            except Exception as e:
                print(f"Error loading config: {e}")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(14)

        # 1. 顶部服务运维条
        status_box = self._create_service_status_bar()
        main_layout.addWidget(status_box)

        # 滚动区域支持不同分辨率自适应
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)

        # 2. 麦克风输入设备与实时测试卡片
        mic_group = self._create_mic_group()
        content_layout.addWidget(mic_group)

        # 3. 提示音效管理卡片
        sound_group = self._create_sound_group()
        content_layout.addWidget(sound_group)

        # 4. 识别与静音自动发送卡片
        vad_group = self._create_vad_group()
        content_layout.addWidget(vad_group)

        content_layout.addStretch()
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # 5. 底部操作与反馈栏
        bottom_layout = QHBoxLayout()
        self.feedback_label = QLabel("")
        self.feedback_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        bottom_layout.addWidget(self.feedback_label)
        bottom_layout.addStretch()

        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self.reset_to_defaults)
        bottom_layout.addWidget(reset_btn)

        save_btn = QPushButton("保存并应用")
        save_btn.setDefault(True)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                font-weight: bold;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #1f618d;
            }
        """)
        save_btn.clicked.connect(self.save_and_apply)
        bottom_layout.addWidget(save_btn)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        bottom_layout.addWidget(close_btn)

        main_layout.addLayout(bottom_layout)

    def _create_service_status_bar(self) -> QWidget:
        container = QFrame()
        container.setFrameShape(QFrame.Shape.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                background-color: palette(window);
                border: 1px solid palette(mid);
                border-radius: 6px;
                padding: 4px;
            }
        """)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(12, 8, 12, 8)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("font-size: 16px; color: #7f8c8d;")
        layout.addWidget(self.status_dot)

        self.status_text = QLabel("正在检测后台服务...")
        self.status_text.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_text)
        layout.addStretch()

        self.btn_refresh_status = QPushButton("刷新状态")
        self.btn_refresh_status.clicked.connect(self.refresh_service_status)
        layout.addWidget(self.btn_refresh_status)

        self.btn_reload_daemon = QPushButton("热重载配置")
        self.btn_reload_daemon.clicked.connect(self.reload_daemon_config)
        layout.addWidget(self.btn_reload_daemon)

        self.btn_restart_service = QPushButton("重启服务")
        self.btn_restart_service.clicked.connect(self.restart_service)
        layout.addWidget(self.btn_restart_service)

        return container

    def _create_mic_group(self) -> QGroupBox:
        group = QGroupBox("麦克风与音频输入设备 (Audio Input)")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # 设备下拉
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("输入设备:"))
        self.device_combo = QComboBox()
        self.device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.device_combo.currentIndexChanged.connect(self.on_device_selection_changed)
        dev_row.addWidget(self.device_combo)

        btn_refresh_devs = QPushButton("刷新设备")
        btn_refresh_devs.clicked.connect(self.populate_audio_devices)
        dev_row.addWidget(btn_refresh_devs)
        layout.addLayout(dev_row)

        # 同步系统全局默认源
        self.chk_sync_global = QCheckBox("切换时同步设为系统全局默认录音源 (PipeWire Default Source)")
        self.chk_sync_global.setToolTip("勾选后，保存配置时会同时执行 pactl set-default-source，令整机其他应用同步使用所选麦克风")
        layout.addWidget(self.chk_sync_global)

        # 实时拾音电平测试条
        meter_header = QHBoxLayout()
        meter_header.addWidget(QLabel("实时拾音测试 (对麦克风说话观察电平反应):"))
        self.meter_val_label = QLabel("静音 / 待机")
        self.meter_val_label.setStyleSheet("color: palette(text); font-family: monospace;")
        meter_header.addStretch()
        meter_header.addWidget(self.meter_val_label)
        layout.addLayout(meter_header)

        self.vu_widget = VUMeterWidget()
        layout.addWidget(self.vu_widget)

        self.populate_audio_devices()
        return group

    def _create_sound_group(self) -> QGroupBox:
        group = QGroupBox("提示音效管理 (Prompt Sound Effects)")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # 总开关
        self.chk_sound_enabled = QCheckBox("启用提示音效 (Sound Effects Enabled)")
        self.chk_sound_enabled.setChecked(self.config_data.get("sound_enabled", True))
        layout.addWidget(self.chk_sound_enabled)

        # 音量滑块
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("提示音量:"))
        self.sound_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.sound_vol_slider.setRange(0, 100)
        curr_vol = int(self.config_data.get("sound_volume", 0.35) * 100)
        self.sound_vol_slider.setValue(curr_vol)
        self.sound_vol_label = QLabel(f"{curr_vol}%")
        self.sound_vol_label.setFixedWidth(40)
        self.sound_vol_slider.valueChanged.connect(
            lambda v: self.sound_vol_label.setText(f"{v}%")
        )
        vol_row.addWidget(self.sound_vol_slider)
        vol_row.addWidget(self.sound_vol_label)
        layout.addLayout(vol_row)

        # 音效分项网格
        grid = QGridLayout()
        grid.setSpacing(8)

        # 1. 录音开始
        self.chk_start_sound = QCheckBox("录音开始音效")
        self.chk_start_sound.setChecked(self.config_data.get("sound_on_start", True))
        grid.addWidget(self.chk_start_sound, 0, 0)

        self.txt_start_sound = QLineEdit(self.config_data.get("sound_custom_start", ""))
        self.txt_start_sound.setPlaceholderText("系统预设柔和马林巴双音 (start.wav)")
        grid.addWidget(self.txt_start_sound, 0, 1)

        btn_browse_start = QPushButton("浏览...")
        btn_browse_start.clicked.connect(lambda: self.browse_sound_file(self.txt_start_sound))
        grid.addWidget(btn_browse_start, 0, 2)

        btn_test_start = QPushButton("试听")
        btn_test_start.clicked.connect(lambda: self.test_sound("start", self.txt_start_sound.text()))
        grid.addWidget(btn_test_start, 0, 3)

        # 2. 录音结束
        self.chk_stop_sound = QCheckBox("录音结束音效")
        self.chk_stop_sound.setChecked(self.config_data.get("sound_on_stop", True))
        grid.addWidget(self.chk_stop_sound, 1, 0)

        self.txt_stop_sound = QLineEdit(self.config_data.get("sound_custom_stop", ""))
        self.txt_stop_sound.setPlaceholderText("系统预设降序提示音 (stop.wav)")
        grid.addWidget(self.txt_stop_sound, 1, 1)

        btn_browse_stop = QPushButton("浏览...")
        btn_browse_stop.clicked.connect(lambda: self.browse_sound_file(self.txt_stop_sound))
        grid.addWidget(btn_browse_stop, 1, 2)

        btn_test_stop = QPushButton("试听")
        btn_test_stop.clicked.connect(lambda: self.test_sound("stop", self.txt_stop_sound.text()))
        grid.addWidget(btn_test_stop, 1, 3)

        # 3. 识别完成
        self.chk_success_sound = QCheckBox("识别完成音效")
        self.chk_success_sound.setChecked(self.config_data.get("sound_on_success", True))
        grid.addWidget(self.chk_success_sound, 2, 0)

        self.txt_success_sound = QLineEdit(self.config_data.get("sound_custom_success", ""))
        self.txt_success_sound.setPlaceholderText("系统预设清脆回响音 (success.wav)")
        grid.addWidget(self.txt_success_sound, 2, 1)

        btn_browse_success = QPushButton("浏览...")
        btn_browse_success.clicked.connect(lambda: self.browse_sound_file(self.txt_success_sound))
        grid.addWidget(btn_browse_success, 2, 2)

        btn_test_success = QPushButton("试听")
        btn_test_success.clicked.connect(lambda: self.test_sound("success", self.txt_success_sound.text()))
        grid.addWidget(btn_test_success, 2, 3)

        layout.addLayout(grid)
        return group

    def _create_vad_group(self) -> QGroupBox:
        group = QGroupBox("语音识别与智能停顿设置 (Whisper & VAD)")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # 语言
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("识别目标语言:"))
        self.lang_combo = QComboBox()
        langs = [
            ("auto", "自动检测语言 (推荐，中英混输自动判别)"),
            ("zh", "中文普通话 (zh)"),
            ("en", "英语 (en)"),
            ("yue", "粤语 (yue)"),
            ("ja", "日语 (ja)"),
            ("ko", "韩语 (ko)")
        ]
        curr_lang = self.config_data.get("language", "auto")
        for code, label in langs:
            self.lang_combo.addItem(label, code)
            if code == curr_lang:
                self.lang_combo.setCurrentIndex(self.lang_combo.count() - 1)
        lang_row.addWidget(self.lang_combo)
        layout.addLayout(lang_row)

        # 停顿自动发送
        self.chk_auto_stop = QCheckBox("说话停顿自动停止并转写 (Auto-stop on silence)")
        self.chk_auto_stop.setChecked(self.config_data.get("auto_stop_on_silence", True))
        layout.addWidget(self.chk_auto_stop)

        # 停顿时长
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("停顿判定时长:"))
        self.silence_dur_slider = QSlider(Qt.Orientation.Horizontal)
        self.silence_dur_slider.setRange(10, 50)  # 1.0s ~ 5.0s (步长 0.1s)
        curr_dur = int(self.config_data.get("silence_duration", 3.0) * 10)
        self.silence_dur_slider.setValue(curr_dur)
        self.silence_dur_label = QLabel(f"{curr_dur / 10.0:.1f} 秒")
        self.silence_dur_label.setFixedWidth(50)
        self.silence_dur_slider.valueChanged.connect(
            lambda v: self.silence_dur_label.setText(f"{v / 10.0:.1f} 秒")
        )
        dur_row.addWidget(self.silence_dur_slider)
        dur_row.addWidget(self.silence_dur_label)
        layout.addLayout(dur_row)

        return group

    def populate_audio_devices(self):
        """枚举 PipeWire/ALSA 输入设备并填入下拉框"""
        current_saved_dev = self.config_data.get("input_device", "").strip()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()

        # 默认项
        self.device_combo.addItem("系统默认输入设备 (PipeWire Default)", "")

        sources = []
        try:
            res = subprocess.run(
                ["pactl", "-f", "json", "list", "sources"],
                capture_output=True,
                text=True,
                check=True
            )
            data = json.loads(res.stdout)
            for s in data:
                name = s.get("name", "")
                if not name or ".monitor" in name:
                    continue
                desc = s.get("description", "")
                props = s.get("properties", {})
                bus = props.get("device.bus", "")
                bus_tag = f" [{bus.upper()}]" if bus else ""
                label = f"{desc}{bus_tag}"
                sources.append((name, label))
        except Exception as e:
            print(f"Error querying pactl sources: {e}")

        selected_idx = 0
        for idx, (name, label) in enumerate(sources, start=1):
            self.device_combo.addItem(f"{label} ({name})", name)
            if name == current_saved_dev:
                selected_idx = idx

        self.device_combo.setCurrentIndex(selected_idx)
        self.device_combo.blockSignals(False)

        # 切换电平监听设备
        self.start_vu_meter()

    def on_device_selection_changed(self, index: int):
        self.start_vu_meter()

    def start_vu_meter(self):
        """切换麦克风电平监听流"""
        if self.meter_thread:
            self.meter_thread.level_changed.disconnect(self.update_vu_meter)
            self.meter_thread.stop()
            self.meter_thread = None

        selected_source = self.device_combo.currentData() or ""
        self.meter_thread = AudioMeterThread(source_name=selected_source)
        self.meter_thread.level_changed.connect(self.update_vu_meter)
        self.meter_thread.start()

    def update_vu_meter(self, level: float):
        self.vu_widget.set_level(level)
        pct = int(level * 100)
        if pct < 2:
            self.meter_val_label.setText("静音 / 未检测到声音")
            self.meter_val_label.setStyleSheet("color: palette(mid); font-family: monospace;")
        elif pct < 50:
            self.meter_val_label.setText(f"拾音电平: {pct}% (说话声音正常)")
            self.meter_val_label.setStyleSheet("color: #27ae60; font-family: monospace; font-weight: bold;")
        elif pct < 85:
            self.meter_val_label.setText(f"拾音电平: {pct}% (音量饱满)")
            self.meter_val_label.setStyleSheet("color: #d35400; font-family: monospace; font-weight: bold;")
        else:
            self.meter_val_label.setText(f"拾音电平: {pct}% (接近峰值)")
            self.meter_val_label.setStyleSheet("color: #c0392b; font-family: monospace; font-weight: bold;")

    def browse_sound_file(self, target_line_edit: QLineEdit):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择提示音音频文件",
            os.path.expanduser("~"),
            "音频文件 (*.wav *.mp3 *.ogg *.flac);;所有文件 (*)"
        )
        if file_path:
            target_line_edit.setText(file_path)

    def test_sound(self, event_type: str, custom_path: str):
        vol = self.sound_vol_slider.value() / 100.0
        target = custom_path.strip() if custom_path.strip() else os.path.join(SOUNDS_DIR, f"{event_type}.wav")
        if not os.path.isfile(target):
            QMessageBox.warning(self, "音频不存在", f"找不到音效文件: {target}")
            return

        def _play():
            daemon_mod.play_audio_file(target, volume=vol, timeout_sec=3.0)

        threading.Thread(target=_play, daemon=True).start()

    def refresh_service_status(self):
        try:
            res = subprocess.run(
                ["systemctl", "--user", "is-active", "voice-input.service"],
                capture_output=True,
                text=True
            )
            status = res.stdout.strip()
            if status == "active":
                self.status_dot.setStyleSheet("font-size: 16px; color: #2ecc71;")
                self.status_text.setText("后台服务: 正在运行 (voice-input.service)")
            else:
                self.status_dot.setStyleSheet("font-size: 16px; color: #e74c3c;")
                self.status_text.setText(f"后台服务: {status} (请点击重启)")
        except Exception as e:
            self.status_dot.setStyleSheet("font-size: 16px; color: #f39c12;")
            self.status_text.setText(f"服务状态未知: {e}")

    def reload_daemon_config(self):
        """向后台 UNIX socket 发送 reload 信号"""
        try:
            import socket
            runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
            socket_path = os.path.join(runtime_dir, "voice-input.sock")
            if not os.path.exists(socket_path):
                self.show_feedback("服务 socket 未就绪，将在下次录音时自动生效", is_ok=True)
                return

            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(2.0)
            client.connect(socket_path)
            payload = json.dumps({"command": "reload"})
            client.sendall(payload.encode("utf-8"))
            try:
                client.recv(1024)
            except OSError:
                pass
            client.close()
            self.show_feedback("✓ 后台服务已热重载最新配置", is_ok=True)
        except Exception as e:
            self.show_feedback(f"热重载通知已触发 ({e})", is_ok=True)

    def restart_service(self):
        try:
            subprocess.run(
                ["systemctl", "--user", "restart", "voice-input.service"],
                check=True
            )
            time.sleep(0.5)
            self.refresh_service_status()
            self.show_feedback("✓ 后台服务已重启完毕", is_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "重启失败", f"重启服务失败: {e}")

    def save_and_apply(self):
        # 收集新配置
        selected_dev = self.device_combo.currentData() or ""
        self.config_data["input_device"] = selected_dev
        self.config_data["sound_enabled"] = self.chk_sound_enabled.isChecked()
        self.config_data["sound_volume"] = round(self.sound_vol_slider.value() / 100.0, 2)
        self.config_data["sound_on_start"] = self.chk_start_sound.isChecked()
        self.config_data["sound_on_stop"] = self.chk_stop_sound.isChecked()
        self.config_data["sound_on_success"] = self.chk_success_sound.isChecked()
        self.config_data["sound_custom_start"] = self.txt_start_sound.text().strip()
        self.config_data["sound_custom_stop"] = self.txt_stop_sound.text().strip()
        self.config_data["sound_custom_success"] = self.txt_success_sound.text().strip()

        self.config_data["language"] = self.lang_combo.currentData()
        self.config_data["auto_stop_on_silence"] = self.chk_auto_stop.isChecked()
        self.config_data["silence_duration"] = round(self.silence_dur_slider.value() / 10.0, 1)

        # 写入 config.json
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法写入 config.json: {e}")
            return

        # 同步系统全局默认源
        if self.chk_sync_global.isChecked() and selected_dev:
            try:
                subprocess.run(["pactl", "set-default-source", selected_dev], check=True)
            except Exception as e:
                print(f"Warning setting default source: {e}")

        # 通知守护进程
        self.reload_daemon_config()
        self.show_feedback("✓ 配置已保存并立即生效！", is_ok=True)

    def reset_to_defaults(self):
        reply = QMessageBox.question(
            self,
            "确认恢复默认",
            "是否恢复为系统推荐默认参数？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.config_data = daemon_mod.DEFAULT_CONFIG.copy()
            self.chk_sound_enabled.setChecked(True)
            self.sound_vol_slider.setValue(35)
            self.chk_start_sound.setChecked(True)
            self.chk_stop_sound.setChecked(True)
            self.chk_success_sound.setChecked(True)
            self.txt_start_sound.setText("")
            self.txt_stop_sound.setText("")
            self.txt_success_sound.setText("")
            self.chk_auto_stop.setChecked(True)
            self.silence_dur_slider.setValue(30)
            self.lang_combo.setCurrentIndex(0)
            self.device_combo.setCurrentIndex(0)
            self.save_and_apply()

    def show_feedback(self, msg: str, is_ok: bool = True):
        color = "#27ae60" if is_ok else "#e74c3c"
        self.feedback_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.feedback_label.setText(msg)
        QTimer.singleShot(4000, lambda: self.feedback_label.setText(""))

    def closeEvent(self, event):
        if self.meter_thread:
            self.meter_thread.stop()
            self.meter_thread = None
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VoiceInputSettings")
    window = SettingsWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
