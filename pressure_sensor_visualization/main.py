import sys
import serial
import serial.tools.list_ports
import struct
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QGroupBox,
    QSlider, QCheckBox, QMessageBox, QSplitter, QFileDialog
)
import csv
from datetime import datetime
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont, QLinearGradient, QRadialGradient

SENSOR_SIZE = 16
FRAME_HEADER = bytes([0x55, 0xAA])
FRAME_TAIL = 0x5A

COLORS = {
    "bg_primary": "#f5f7fa",
    "bg_secondary": "#ffffff",
    "bg_card": "#e8f4fc",
    "bg_input": "#f0f2f5",
    "accent": "#3a86ff",
    "accent_hover": "#5c9dff",
    "text_primary": "#2d3436",
    "text_secondary": "#636e72",
    "text_accent": "#3a86ff",
    "border": "#dfe6e9",
    "success": "#00b894",
    "warning": "#fdcb6e",
    "danger": "#d63031",
    "param_value": "#3a86ff",
    "help_btn": "#b2bec3",
    "heatmap_bg": "#dce6f0",
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS["bg_primary"]};
}}

QWidget {{
    background-color: {COLORS["bg_primary"]};
    color: {COLORS["text_primary"]};
    font-family: "PingFang SC", "Microsoft YaHei", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}

QGroupBox {{
    background-color: {COLORS["bg_secondary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    margin-top: 10px;
    padding-top: 18px;
    font-weight: bold;
    font-size: 13px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    top: 2px;
    padding: 0 6px;
    color: {COLORS["text_accent"]};
    font-size: 13px;
}}

QComboBox {{
    background-color: {COLORS["bg_input"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 6px 12px;
    color: {COLORS["text_primary"]};
    min-height: 30px;
}}

QComboBox:hover {{
    border-color: {COLORS["accent"]};
    background-color: {COLORS["bg_secondary"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLORS["text_secondary"]};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS["bg_secondary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    selection-background-color: {COLORS["accent"]};
    selection-color: white;
    outline: none;
    padding: 4px;
}}

QPushButton {{
    background-color: {COLORS["bg_input"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 8px 18px;
    color: {COLORS["text_primary"]};
    font-weight: bold;
    min-height: 34px;
}}

QPushButton:hover {{
    background-color: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
    color: white;
}}

QPushButton:pressed {{
    background-color: {COLORS["accent_hover"]};
}}

QPushButton:disabled {{
    background-color: {COLORS["bg_input"]};
    border-color: {COLORS["border"]};
    color: {COLORS["text_secondary"]};
}}

QSlider::groove:horizontal {{
    border: 1px solid {COLORS["border"]};
    height: 6px;
    background: {COLORS["bg_input"]};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLORS["accent"]};
    width: 18px;
    margin: -7px 0;
    border-radius: 9px;
}}

QSlider::handle:horizontal:hover {{
    background: {COLORS["accent_hover"]};
}}

QSlider::sub-page:horizontal {{
    background: {COLORS["text_accent"]};
    border-radius: 3px;
}}

QSlider::add-page:horizontal {{
    background: {COLORS["bg_input"]};
    border-radius: 3px;
}}

QCheckBox {{
    spacing: 8px;
    color: {COLORS["text_primary"]};
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {COLORS["border"]};
    border-radius: 4px;
    background-color: {COLORS["bg_secondary"]};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS["accent"]};
}}

QLabel {{
    color: {COLORS["text_primary"]};
}}

QSplitter::handle {{
    background-color: {COLORS["border"]};
    width: 2px;
    border-radius: 1px;
}}

QMessageBox {{
    background-color: {COLORS["bg_secondary"]};
    color: {COLORS["text_primary"]};
}}

QMessageBox QLabel {{
    color: {COLORS["text_primary"]};
}}

QMessageBox QPushButton {{
    min-width: 80px;
    border-radius: 6px;
    padding: 6px 16px;
}}

QFileDialog {{
    background-color: {COLORS["bg_secondary"]};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}
"""


class SerialThread(QThread):
    data_received = pyqtSignal(np.ndarray)

    def __init__(self, port, baudrate=460800):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.serial = None

    def run(self):
        self.running = True
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.5)
            buffer = b''
            while self.running:
                if self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    buffer += data
                    while len(buffer) >= 263:
                        idx = buffer.find(FRAME_HEADER)
                        if idx == -1:
                            buffer = b''
                            break
                        buffer = buffer[idx:]
                        if len(buffer) >= 263:
                            length = struct.unpack('<H', buffer[2:4])[0]
                            if length == 257 and buffer[4] == 0x01 and buffer[262] == FRAME_TAIL:
                                pressure_data = np.frombuffer(buffer[5:261], dtype=np.uint8)
                                pressure_data = pressure_data.reshape((SENSOR_SIZE, SENSOR_SIZE), order='F')
                                self.data_received.emit(pressure_data)
                                buffer = buffer[263:]
                            else:
                                buffer = buffer[1:]
        except serial.SerialException as e:
            print(f"Serial error: {e}")
        finally:
            if self.serial and self.serial.is_open:
                self.serial.close()

    def stop(self):
        self.running = False
        self.wait()


class HeatmapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = np.zeros((SENSOR_SIZE, SENSOR_SIZE), dtype=np.uint8)
        self.smooth = True
        self.smooth_factor = 1
        self.show_values = True
        self.font_size = 10
        self.setMinimumSize(500, 500)

    def set_data(self, data):
        if self.smooth:
            self.data = self.data * (1 - self.smooth_factor) + data * self.smooth_factor
        else:
            self.data = data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor(COLORS["bg_primary"]))

        width = self.width()
        height = self.height()
        margin = 24
        available = min(width, height) - margin * 2
        size = available
        offset_x = (width - size) // 2
        offset_y = (height - size) // 2
        cell_size = size / SENSOR_SIZE

        painter.setBrush(Qt.NoBrush)

        for i in range(SENSOR_SIZE):
            for j in range(SENSOR_SIZE):
                value = int(self.data[i, j])
                color = self.get_color(value)
                painter.fillRect(
                    int(offset_x + j * cell_size),
                    int(offset_y + i * cell_size),
                    int(cell_size - 2),
                    int(cell_size - 2),
                    color
                )

        painter.setPen(QColor(223, 230, 233, 80))
        for i in range(SENSOR_SIZE + 1):
            x = int(offset_x + i * cell_size)
            y = int(offset_y + i * cell_size)
            painter.drawLine(x, offset_y, x, offset_y + int(size))
            painter.drawLine(offset_x, y, offset_x + int(size), y)

        region_height = int(size * 10 / SENSOR_SIZE)
        painter.setPen(QColor(116, 185, 255, 80))
        painter.drawLine(offset_x, offset_y + region_height, offset_x + int(size), offset_y + region_height)
        painter.setPen(QColor(255, 118, 117, 80))
        neck_line = offset_y + int(size * 13 / SENSOR_SIZE)
        painter.drawLine(offset_x, neck_line, offset_x + int(size), neck_line)

        font = QFont("PingFang SC", 10, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(116, 185, 255, 180))
        painter.drawText(offset_x + 4, offset_y + 14, "头")
        painter.setPen(QColor(255, 118, 117, 180))
        painter.drawText(offset_x + 4, offset_y + region_height + 14, "颈")
        painter.drawText(offset_x + 4, neck_line + 14, "肩")

        if self.show_values and cell_size >= 18:
            font_size = max(7, min(int(cell_size * 0.35), 12))
            painter.setFont(QFont("PingFang SC", font_size, QFont.Bold))
            for i in range(SENSOR_SIZE):
                for j in range(SENSOR_SIZE):
                    value = int(self.data[i, j])
                    if value < 5:
                        continue
                    text_color = QColor(255, 255, 255, 230) if value > 128 else QColor(50, 50, 50, 200)
                    painter.setPen(text_color)
                    painter.drawText(
                        int(offset_x + j * cell_size + 2),
                        int(offset_y + i * cell_size + cell_size * 0.6),
                        str(value)
                    )

        self._draw_legend(painter, offset_x, offset_y + int(size) + 10, int(size), 14)

    def _draw_legend(self, painter, x, y, width, height):
        bar_width = width
        bar_height = height
        for i in range(bar_width):
            value = int(i * 255 / bar_width)
            color = self.get_color(value)
            painter.fillRect(x + i, y, 1, bar_height, color)

        painter.setPen(QColor(99, 110, 114))
        painter.setFont(QFont("PingFang SC", 9))
        painter.drawText(x, y + bar_height + 12, "0")
        painter.drawText(x + bar_width - 20, y + bar_height + 12, "255")

    def get_color(self, value):
        value = max(0, min(255, value))
        if value < 64:
            r = int(220 + (255 - 220) * value / 63)
            g = int(230 + (255 - 230) * value / 63)
            b = int(245 + (10 - 245) * value / 63)
        elif value < 128:
            t = (value - 64) / 63
            r = int(255 - 137 * t)
            g = int(255 - 73 * t)
            b = int(10 + 100 * t)
        elif value < 192:
            t = (value - 128) / 63
            r = int(118 + 137 * t)
            g = int(182 - 182 * t)
            b = int(110 - 110 * t)
        else:
            t = (value - 192) / 63
            r = int(255)
            g = int(0 + 40 * t)
            b = int(0 + 80 * t)
        return QColor(r, min(255, g), min(255, b))


class ParamRow(QWidget):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        label = QLabel(name)
        label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(label)

        self.value_label = QLabel("0")
        self.value_label.setStyleSheet(f"font-weight: bold; color: {COLORS['param_value']}; font-size: 12px;")
        self.value_label.setFixedWidth(80)
        layout.addWidget(self.value_label)

        help_btn = QPushButton("?")
        help_btn.setFixedSize(18, 18)
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 10px;
                border-radius: 9px;
                background: {COLORS['help_btn']};
                color: white;
                border: none;
                padding: 0;
            }}
            QPushButton:hover {{
                background: {COLORS['text_accent']};
            }}
        """)
        layout.addWidget(help_btn)

        layout.addStretch()
        self.help_btn = help_btn
        self.name = name


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" 枕头压力测试系统")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.resize(1500, 800)
        self.setMinimumSize(1100, 700)

        self.serial_thread = None
        self.current_data = np.zeros((SENSOR_SIZE, SENSOR_SIZE), dtype=np.uint8)
        self.recorded_data = []
        self.param_rows = []
        self._recording = False
        self._record_samples = []
        self._record_timer = QTimer()
        self._record_timer.timeout.connect(self._record_tick)
        self._record_remaining = 0
        self._record_countdown_label = None

        self.setup_ui()
        self.refresh_ports()

    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {COLORS['bg_primary']};")
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        left_panel = QWidget()
        left_panel.setFixedWidth(420)
        left_panel.setStyleSheet(f"background-color: {COLORS['bg_primary']};")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        left_scroll = QWidget()
        left_scroll_layout = QVBoxLayout(left_scroll)
        left_scroll_layout.setContentsMargins(0, 0, 0, 0)
        left_scroll_layout.setSpacing(8)

        conn_group = QGroupBox("  串口设置")
        conn_layout = QVBoxLayout()
        conn_layout.setSpacing(6)

        port_layout = QHBoxLayout()
        port_label = QLabel("端口:")
        port_label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 40px;")
        port_layout.addWidget(port_label)
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(100)
        port_layout.addWidget(self.port_combo)
        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedWidth(50)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                color: {COLORS['text_secondary']};
                font-size: 11px;
                padding: 4px 8px;
                min-height: 24px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['text_accent']};
                color: {COLORS['bg_primary']};
            }}
        """)
        refresh_btn.clicked.connect(self.refresh_ports)
        port_layout.addWidget(refresh_btn)
        conn_layout.addLayout(port_layout)

        baud_layout = QHBoxLayout()
        baud_label = QLabel("波特率:")
        baud_label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 40px;")
        baud_layout.addWidget(baud_label)
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["460800", "115200", "57600", "38400", "19200"])
        self.baud_combo.setCurrentText("460800")
        baud_layout.addWidget(self.baud_combo)
        conn_layout.addLayout(baud_layout)

        self.connect_btn = QPushButton(" 连接")
        self.connect_btn.setMinimumHeight(38)
        self.connect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #00d2a4;
            }}
            QPushButton:pressed {{
                background-color: #009e7a;
            }}
        """)
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(self.connect_btn)

        conn_group.setLayout(conn_layout)
        left_scroll_layout.addWidget(conn_group)

        display_group = QGroupBox("  显示设置")
        display_layout = QVBoxLayout()
        display_layout.setSpacing(6)

        self.smooth_check = QCheckBox("平滑处理")
        self.smooth_check.setChecked(True)
        self.smooth_check.stateChanged.connect(self.update_smooth)
        display_layout.addWidget(self.smooth_check)

        factor_layout = QHBoxLayout()
        factor_label = QLabel("平滑系数:")
        factor_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        factor_layout.addWidget(factor_label)
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setMinimum(1)
        self.smooth_slider.setMaximum(100)
        self.smooth_slider.setValue(30)
        self.smooth_slider.valueChanged.connect(self.update_smooth_factor)
        factor_layout.addWidget(self.smooth_slider)
        self.smooth_value_label = QLabel("30%")
        self.smooth_value_label.setFixedWidth(35)
        self.smooth_value_label.setStyleSheet(f"color: {COLORS['text_accent']}; font-size: 11px;")
        factor_layout.addWidget(self.smooth_value_label)
        display_layout.addLayout(factor_layout)

        self.values_check = QCheckBox("显示数值")
        self.values_check.setChecked(True)
        self.values_check.stateChanged.connect(self.update_show_values)
        display_layout.addWidget(self.values_check)

        display_group.setLayout(display_layout)
        left_scroll_layout.addWidget(display_group)

        pillow_group = QGroupBox("  枕头测试参数")
        pillow_layout = QVBoxLayout()
        pillow_layout.setSpacing(2)

        col1 = QVBoxLayout()
        col2 = QVBoxLayout()
        col1.setSpacing(1)
        col2.setSpacing(1)

        param_names_col1 = [
            "最大压力:", "平均压力:", "接触面积:", "均匀度:",
            "峰值位置:", "峰值压力:", "压力集中指数:"
        ]
        param_names_col2 = [
            "95百分位压力:", "压力梯度:", "头部承力:", "颈椎承力:",
            "肩膀承力:", "颈部连续性:", "颈部空隙:", "压力中心:"
        ]

        self.param_labels = {}
        for name in param_names_col1:
            row = ParamRow(name)
            col1.addWidget(row)
            self.param_labels[name] = row.value_label
            row.help_btn.clicked.connect(lambda checked, n=name: self._show_help(n))

        for name in param_names_col2:
            row = ParamRow(name)
            col2.addWidget(row)
            self.param_labels[name] = row.value_label
            row.help_btn.clicked.connect(lambda checked, n=name: self._show_help(n))

        cols_layout = QHBoxLayout()
        cols_layout.addLayout(col1)
        cols_layout.addLayout(col2)
        pillow_layout.addLayout(cols_layout)

        pillow_group.setLayout(pillow_layout)
        left_scroll_layout.addWidget(pillow_group)

        record_group = QGroupBox("  测试记录")
        record_layout = QVBoxLayout()
        record_layout.setSpacing(6)

        fields_input = [
            ("姓名:", "name_edit", ["测试者"]),
            ("身高(cm):", "height_edit", []),
            ("体重(kg):", "weight_edit", []),
            ("肩宽(cm):", "shoulder_width_edit", []),
            ("枕头品牌:", "pillow_brand_edit", ["Pillow A", "Pillow B", "Pillow C", "Custom"]),
            ("枕头高度(cm):", "pillow_height_edit", []),
            ("枕头硬度:", "pillow_hardness_edit", ["软", "适中", "偏硬"]),
            ("颈椎曲度:", "neck_curve_edit", ["正常", "变直", "反弓"]),
            ("睡姿:", "sleep_pos_combo", ["仰睡", "侧睡"]),
        ]
        for label_text, attr_name, items in fields_input:
            row_layout = QHBoxLayout()
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 70px;")
            row_layout.addWidget(label)
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(items)
            row_layout.addWidget(combo)
            setattr(self, attr_name, combo)
            record_layout.addLayout(row_layout)

        comfort_layout = QHBoxLayout()
        comfort_label = QLabel("舒适度(-5~5):")
        comfort_label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 40px;")
        comfort_layout.addWidget(comfort_label)
        self.comfort_slider = QSlider(Qt.Horizontal)
        self.comfort_slider.setMinimum(-5)
        self.comfort_slider.setMaximum(5)
        self.comfort_slider.setValue(0)
        self.comfort_slider.setTickPosition(QSlider.TicksBelow)
        self.comfort_slider.setTickInterval(1)
        comfort_layout.addWidget(self.comfort_slider)
        self.comfort_value_label = QLabel("0")
        self.comfort_value_label.setFixedWidth(20)
        self.comfort_value_label.setStyleSheet(f"color: {COLORS['text_accent']}; font-weight: bold; font-size: 13px;")
        comfort_layout.addWidget(self.comfort_value_label)
        self.comfort_slider.valueChanged.connect(lambda v: self.comfort_value_label.setText(str(v)))
        record_layout.addLayout(comfort_layout)

        note_layout = QHBoxLayout()
        note_label = QLabel("备注:")
        note_label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 40px;")
        note_layout.addWidget(note_label)
        self.note_edit = QComboBox()
        self.note_edit.setEditable(True)
        self.note_edit.addItems(["", "太硬", "太软", "合适", "颈椎疼", "肩膀疼", "很舒服"])
        note_layout.addWidget(self.note_edit)
        record_layout.addLayout(note_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.record_btn = QPushButton(" 记录数据")
        self.record_btn.setMinimumHeight(38)
        self.record_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.record_btn.clicked.connect(self.record_data)
        self.record_btn.setEnabled(False)
        btn_layout.addWidget(self.record_btn)

        self.save_btn = QPushButton(" 导出CSV")
        self.save_btn.setMinimumHeight(38)
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['text_accent']};
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #5ca0ff;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.save_btn.clicked.connect(self.export_csv)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)
        record_layout.addLayout(btn_layout)

        self.record_table = QLabel("")
        self.record_table.setWordWrap(True)
        self.record_table.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_input']};
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
            }}
        """)
        record_layout.addWidget(self.record_table)

        record_group.setLayout(record_layout)
        left_scroll_layout.addWidget(record_group)

        left_scroll_layout.addStretch()

        scroll_area_wrapper = QWidget()
        scroll_layout = QVBoxLayout(scroll_area_wrapper)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(left_scroll)

        from PyQt5.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidget(left_scroll)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        left_layout.addWidget(scroll)

        self.status_label = QLabel(" 未连接")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_input']};
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 11px;
                color: {COLORS['text_secondary']};
            }}
        """)
        left_layout.addWidget(self.status_label)

        self.heatmap_widget = HeatmapWidget()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.heatmap_widget)
        splitter.setSizes([420, 900])
        main_layout.addWidget(splitter)

    PARAM_HELP = {
        "最大压力:": "传感器检测到的最大压力值，单位0-255",
        "平均压力:": "所有接触点压力的平均值",
        "接触面积:": "有压力触发的传感器点占总点数的比例",
        "均匀度:": "压力分布的标准差，值越小越均匀",
        "峰值位置:": "压力最大点的X,Y坐标(左上角为0,0)",
        "峰值压力:": "局部压力感，越高越觉得硌",
        "压力集中指数:": "压力集中程度，越高越不舒适",
        "95百分位压力:": "所有压力值中前5%的平均值",
        "压力梯度:": "相邻点压力差的平均值，反映压力变化陡峭程度",
        "头部承力:": "头部区域(0-10行)占总压力的百分比",
        "颈椎承力:": "颈椎区域(10-13行)占总压力的百分比",
        "肩膀承力:": "肩膀区域(13-16行)占总压力的百分比",
        "颈部连续性:": "颈部区域压力值的标准差，越小越均匀",
        "颈部空隙:": "检测颈部是否有明显的低压力区域",
        "压力中心:": "压力中心的X,Y坐标位置"
    }

    def _show_help(self, name):
        QMessageBox.information(self, name, self.PARAM_HELP.get(name, "无说明"))

    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)

    def toggle_connection(self):
        if self.serial_thread and self.serial_thread.isRunning():
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self):
        port = self.port_combo.currentText()
        if not port:
            QMessageBox.warning(self, "Warning", "Please select a port")
            return

        baudrate = int(self.baud_combo.currentText())
        self.serial_thread = SerialThread(port, baudrate)
        self.serial_thread.data_received.connect(self.on_data_received)
        self.serial_thread.start()

        self.connect_btn.setText(" 断开")
        self.connect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #e74c3c;
            }}
        """)
        self.status_label.setText(f" 已连接 {port} @ {baudrate}")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_input']};
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 11px;
                color: {COLORS['success']};
            }}
        """)
        self.record_btn.setEnabled(True)

    def disconnect_serial(self):
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread = None
        self.connect_btn.setText(" 连接")
        self.connect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #00d2a4;
            }}
        """)
        self.status_label.setText(" 未连接")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_input']};
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 11px;
                color: {COLORS['text_secondary']};
            }}
        """)
        self.record_btn.setEnabled(False)

    def on_data_received(self, data):
        self.current_data = data
        self.heatmap_widget.set_data(data)
        self.update_params_display(data)

    def update_smooth(self, state):
        self.heatmap_widget.smooth = (state == Qt.Checked)
        self.heatmap_widget.smooth_factor = self.smooth_slider.value() / 100.0

    def update_smooth_factor(self, value):
        self.heatmap_widget.smooth_factor = value / 100.0
        self.smooth_value_label.setText(f"{value}%")

    def update_show_values(self, state):
        self.heatmap_widget.show_values = (state == Qt.Checked)
        self.heatmap_widget.update()

    def calculate_params(self, data):
        max_val = int(data.max())
        avg_val = int(data.mean())
        contact_points = np.sum(data > 10)
        contact_area = int(contact_points / (SENSOR_SIZE * SENSOR_SIZE) * 100)
        std_val = int(data.std())

        peak_pos = np.unravel_index(data.argmax(), data.shape)
        peak_pos_str = f"({peak_pos[1]},{peak_pos[0]})"

        peak_val = max_val

        flat_data = data.flatten()
        flat_data = flat_data[flat_data > 0]
        if len(flat_data) > 0:
            concentration = int((flat_data ** 2).sum() / max((flat_data.sum() ** 2), 1) * 100)
            sorted_data = np.sort(flat_data)[::-1]
            p95_count = max(1, int(len(sorted_data) * 0.05))
            p95_val = int(sorted_data[:p95_count].mean())
        else:
            concentration = 0
            p95_val = 0

        gradient_sum = 0
        gradient_count = 0
        for i in range(SENSOR_SIZE):
            for j in range(SENSOR_SIZE - 1):
                gradient_sum += abs(int(data[i, j]) - int(data[i, j + 1]))
                gradient_count += 1
            if i < SENSOR_SIZE - 1:
                for j in range(SENSOR_SIZE):
                    gradient_sum += abs(int(data[i, j]) - int(data[i + 1, j]))
                    gradient_count += 1
        gradient = int(gradient_sum / max(gradient_count, 1))

        head_region = data[:10, :].sum()
        neck_region = data[10:13, :].sum()
        shoulder_region = data[13:, :].sum()
        total = head_region + neck_region + shoulder_region

        if total > 0:
            head_ratio = int(head_region / total * 100)
            neck_ratio = int(neck_region / total * 100)
            shoulder_ratio = int(shoulder_region / total * 100)
        else:
            head_ratio = 0
            neck_ratio = 0
            shoulder_ratio = 0

        neck_continuity = int(data[10:13, :].std())

        y_coords, x_coords = np.where(data > 10)
        if len(x_coords) > 0:
            center_x = int(x_coords.mean() / SENSOR_SIZE * 100)
            center_y = int(y_coords.mean() / SENSOR_SIZE * 100)
            center = f"{center_x}%, {center_y}%"
        else:
            center = "0%, 0%"

        if len(x_coords) >= 4:
            mid_row = data[7, :]
            bottom_row = data[15, :]
            gap_score = 0
            if mid_row.mean() < bottom_row.mean() * 0.5:
                gap_score = 1
            neck_gap = "有" if gap_score else "无"
        else:
            neck_gap = "无"

        return max_val, avg_val, contact_area, std_val, peak_pos_str, peak_val, concentration, p95_val, gradient, head_ratio, neck_ratio, shoulder_ratio, neck_continuity, neck_gap, center

    def calculate_params_numeric(self, data):
        max_val = int(data.max())
        avg_val = int(data.mean())
        contact_points = np.sum(data > 10)
        contact_area = int(contact_points / (SENSOR_SIZE * SENSOR_SIZE) * 100)
        std_val = int(data.std())
        peak_val = max_val

        flat_data = data.flatten()
        flat_data = flat_data[flat_data > 0]
        if len(flat_data) > 0:
            concentration = int((flat_data ** 2).sum() / max((flat_data.sum() ** 2), 1) * 100)
            sorted_data = np.sort(flat_data)[::-1]
            p95_count = max(1, int(len(sorted_data) * 0.05))
            p95_val = int(sorted_data[:p95_count].mean())
        else:
            concentration = 0
            p95_val = 0

        gradient_sum = 0
        gradient_count = 0
        for i in range(SENSOR_SIZE):
            for j in range(SENSOR_SIZE - 1):
                gradient_sum += abs(int(data[i, j]) - int(data[i, j + 1]))
                gradient_count += 1
            if i < SENSOR_SIZE - 1:
                for j in range(SENSOR_SIZE):
                    gradient_sum += abs(int(data[i, j]) - int(data[i + 1, j]))
                    gradient_count += 1
        gradient = int(gradient_sum / max(gradient_count, 1))

        head_region = data[:10, :].sum()
        neck_region = data[10:13, :].sum()
        shoulder_region = data[13:, :].sum()
        total = head_region + neck_region + shoulder_region

        if total > 0:
            head_ratio = int(head_region / total * 100)
            neck_ratio = int(neck_region / total * 100)
            shoulder_ratio = int(shoulder_region / total * 100)
        else:
            head_ratio = 0
            neck_ratio = 0
            shoulder_ratio = 0

        neck_continuity = int(data[10:13, :].std())

        return [max_val, avg_val, contact_area, std_val, peak_val, concentration, p95_val, gradient, head_ratio, neck_ratio, shoulder_ratio, neck_continuity]

    def update_params_display(self, data):
        params = self.calculate_params(data)
        names = [
            "最大压力:", "平均压力:", "接触面积:", "均匀度:",
            "峰值位置:", "峰值压力:", "压力集中指数:",
            "95百分位压力:", "压力梯度:", "头部承力:", "颈椎承力:",
            "肩膀承力:", "颈部连续性:", "颈部空隙:", "压力中心:"
        ]
        values = [
            str(params[0]), str(params[1]), f"{params[2]}%", str(params[3]),
            params[4], str(params[5]), str(params[6]),
            str(params[7]), str(params[8]), f"{params[9]}%", f"{params[10]}%",
            f"{params[11]}%", str(params[12]), params[13], params[14]
        ]
        for name, value in zip(names, values):
            if name in self.param_labels:
                self.param_labels[name].setText(value)

    def record_data(self):
        if self._recording:
            return
        self._recording = True
        self._record_samples = []
        self._record_remaining = 5
        self.record_btn.setText(f" 记录中 {self._record_remaining}s")
        self.record_btn.setEnabled(False)
        self._record_timer.start(100)

    def _record_tick(self):
        self._record_remaining -= 0.1
        if self.current_data is not None:
            self._record_samples.append(self.current_data.copy())
        if self._record_remaining <= 0:
            self._record_timer.stop()
            self._finish_recording()
        else:
            self.record_btn.setText(f" 记录中 {self._record_remaining:.1f}s")

    def _finish_recording(self):
        self._recording = False
        self.record_btn.setText(" 记录数据")
        self.record_btn.setEnabled(True)

        if len(self._record_samples) < 2:
            QMessageBox.warning(self, "Warning", "采样数据不足，请确保传感器已连接")
            return

        all_params = []
        for sample in self._record_samples:
            params = self.calculate_params_numeric(sample)
            all_params.append(params)

        all_params = np.array(all_params)
        avg_params = all_params.mean(axis=0).astype(int)
        min_params = all_params.min(axis=0).astype(int)
        max_params = all_params.max(axis=0).astype(int)
        std_params = all_params.std(axis=0).astype(int)
        median_params = np.median(all_params, axis=0).astype(int)
        last_params = all_params[-1]

        name = self.name_edit.currentText() or "测试者"
        height = self.height_edit.currentText() or ""
        weight = self.weight_edit.currentText() or ""
        shoulder_width = self.shoulder_width_edit.currentText() or ""
        pillow_brand = self.pillow_brand_edit.currentText() or ""
        pillow_height = self.pillow_height_edit.currentText() or ""
        pillow_hardness = self.pillow_hardness_edit.currentText() or ""
        neck_curve = self.neck_curve_edit.currentText() or ""
        sleep_pos = self.sleep_pos_combo.currentText()
        comfort = self.comfort_slider.value()
        note = self.note_edit.currentText() or ""

        record = {
            "name": name,
            "height": height,
            "weight": weight,
            "shoulder_width": shoulder_width,
            "pillow_brand": pillow_brand,
            "pillow_height": pillow_height,
            "pillow_hardness": pillow_hardness,
            "neck_curve": neck_curve,
            "sleep_pos": sleep_pos,
            "time": datetime.now().strftime("%H:%M:%S"),
            "comfort": comfort,
            "note": note,
            "samples": len(self._record_samples),
            "duration_s": 5.0,
            "avg_max": int(avg_params[0]),
            "avg_avg": int(avg_params[1]),
            "avg_area": int(avg_params[2]),
            "avg_std": int(avg_params[3]),
            "avg_peak_val": int(avg_params[4]),
            "avg_concentration": int(avg_params[5]),
            "avg_p95": int(avg_params[6]),
            "avg_gradient": int(avg_params[7]),
            "avg_head_ratio": int(avg_params[8]),
            "avg_neck_ratio": int(avg_params[9]),
            "avg_shoulder_ratio": int(avg_params[10]),
            "avg_neck_continuity": int(avg_params[11]),
            "std_max": int(std_params[0]),
            "std_avg": int(std_params[1]),
            "std_area": int(std_params[2]),
            "min_max": int(min_params[0]),
            "max_max": int(max_params[0]),
            "min_area": int(min_params[2]),
            "max_area": int(max_params[2]),
            "median_avg": int(median_params[1]),
            "last_max": int(last_params[0]),
            "last_avg": int(last_params[1]),
            "last_area": int(last_params[2]),
        }
        self.recorded_data.append(record)
        self.update_record_table()
        self.save_btn.setEnabled(True)

    def update_record_table(self):
        if not self.recorded_data:
            self.record_table.setText("暂无记录")
            return
        text = ""
        for r in self.recorded_data[-5:]:
            text += f"{r['time']}  {r['name']}  {r['sleep_pos']}  评分:{r['comfort']}/5\n"
            text += f"   身高:{r['height']} 体重:{r['weight']} 肩宽:{r['shoulder_width']}\n"
            text += f"   枕头:{r['pillow_brand']} 高度:{r['pillow_height']} 硬度:{r['pillow_hardness']} 颈椎:{r['neck_curve']}\n"
            text += f"   5s采样:{r.get('samples',0)}帧 均值:最大{r.get('avg_max',0)} 平均{r.get('avg_avg',0)} 面积{r.get('avg_area',0)}%\n"
            text += f"   波动:std_max={r.get('std_max',0)} std_avg={r.get('std_avg',0)} std_area={r.get('std_area',0)}\n"
            text += f"   备注:{r.get('note','')}\n\n"
        self.record_table.setText(text)

    def export_csv(self):
        if not self.recorded_data:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", f"pillow_test_{datetime.now().strftime('%Y%m%d')}.csv", "CSV Files (*.csv)")
        if not filename:
            return
        try:
            fieldnames = ["name", "height", "weight", "shoulder_width", "pillow_brand", "pillow_height", "pillow_hardness", "neck_curve", "sleep_pos", "time", "comfort", "note", "samples", "duration_s", "avg_max", "avg_avg", "avg_area", "avg_std", "avg_peak_val", "avg_concentration", "avg_p95", "avg_gradient", "avg_head_ratio", "avg_neck_ratio", "avg_shoulder_ratio", "avg_neck_continuity", "std_max", "std_avg", "std_area", "min_max", "max_max", "min_area", "max_area", "median_avg", "last_max", "last_avg", "last_area"]
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for r in self.recorded_data:
                    writer.writerow({k: r.get(k, '') for k in fieldnames})
            QMessageBox.information(self, "成功", f"已保存 {len(self.recorded_data)} 条记录")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def closeEvent(self, event):
        self.disconnect_serial()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
