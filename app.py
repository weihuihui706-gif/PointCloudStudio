
from html import escape
from copy import deepcopy
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QScrollArea,
    QSplitter,
    QProgressBar,
    QFrame
)

from PySide6.QtCore import Qt, QSaveFile, QIODevice, QThread, Signal, Slot, QTimer

from open3d_viewport import Open3DViewport
from viewer import load_point_cloud, save_point_cloud


class BackgroundJob(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation

    def run(self):
        try:
            self.succeeded.emit(self.operation())
        except Exception as error:
            logging.getLogger("PointCloudStudio").exception("后台处理失败")
            self.failed.emit(str(error))


class PointCloudApp(QMainWindow):

    def __init__(self):
        super().__init__()

        # 当前加载的点云文件
        self.current_file = None
        self.current_cloud = None
        self.saved_view = None
        self.sampled_cloud = None
        self.filter_source = None
        self.removed_cloud = None

        # 设置窗口
        self.setWindowTitle("PointCloud Studio")
        self.resize(1280, 800)

        self._job = None
        self._closing_requested = False
        self._completion = None
        self._task_failed = False
        log_directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PointCloudStudio" / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        self.log_path = log_directory / "app.log"
        self.logger = logging.getLogger("PointCloudStudio")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = RotatingFileHandler(self.log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(handler)
        self.init_ui()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(180)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.refresh_operation_buttons()

    def init_ui(self):
        """初始化图形界面"""

        # ===== 主窗口 =====

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 8)
        main_layout.setSpacing(12)

        # ===== 左侧文件管理区域 =====

        left_layout = QVBoxLayout()

        title = QLabel("PointCloud Studio")
        title.setStyleSheet(
            "font-size: 19px; font-weight: bold; color: #0f172a;"
        )

        self.open_button = QPushButton("打开点云文件")
        self.open_button.clicked.connect(
            self.open_file
        )

        self.reset_view_button = QPushButton("重置三维视角")
        self.reset_view_button.clicked.connect(
            self.reset_view
        )
        self.reset_view_button.setEnabled(False)

        self.file_label = QLabel("尚未加载文件")
        self.file_label.setWordWrap(True)
        self.file_label.setObjectName("fileSummary")

        left_layout.addWidget(title)
        left_layout.addWidget(QLabel("点云浏览与处理"))
        left_layout.addSpacing(8)
        left_layout.addWidget(self.open_button)
        left_layout.addWidget(self.reset_view_button)
        left_layout.addSpacing(15)
        left_layout.addWidget(self.file_label)
        left_layout.addStretch()

        # ===== 左侧属性与右侧三维视口 =====

        right_layout = QVBoxLayout()

        info_group = QFrame()
        info_group.setObjectName("propertyCard")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(16, 16, 16, 16)
        info_layout.setSpacing(12)
        info_title = QLabel("原始点云属性")
        info_title.setObjectName("cardTitle")
        info_layout.addWidget(info_title)

        self.info_label = QLabel(
            "请先打开一个点云文件。"
        )
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(
            Qt.AlignmentFlag.AlignTop
        )

        info_layout.addWidget(self.info_label)

        left_layout.insertWidget(left_layout.count() - 1, info_group)
        self.viewport = Open3DViewport()
        self.viewport.close_ready.connect(self.close)
        display_controls = QHBoxLayout()
        display_controls.addWidget(QLabel("点大小"))
        self.point_size_input = QSpinBox()
        self.point_size_input.setRange(1, 25)
        self.point_size_input.setValue(5)
        self.point_size_input.setSuffix(" px")
        self.point_size_input.valueChanged.connect(self.viewport.set_point_size)
        display_controls.addWidget(self.point_size_input)
        display_controls.addWidget(QLabel("背景"))
        self.background_input = QComboBox()
        self.background_input.addItem("深色", (0.12, 0.14, 0.18))
        self.background_input.addItem("浅色", (0.95, 0.95, 0.95))
        self.background_input.addItem("黑色", (0.0, 0.0, 0.0))
        self.background_input.currentIndexChanged.connect(self.change_background)
        self.change_background()
        display_controls.addWidget(self.background_input)
        self.axis_checkbox = QCheckBox("显示坐标轴")
        self.axis_checkbox.setChecked(True)
        self.axis_checkbox.toggled.connect(self.change_axis_visibility)
        display_controls.addWidget(self.axis_checkbox)
        screenshot_button = QPushButton("保存视口截图")
        screenshot_button.clicked.connect(self.save_screenshot)
        display_controls.addWidget(screenshot_button)
        display_controls.addStretch()

        view_controls = QHBoxLayout()
        self.remember_view_button = QPushButton("记住视角")
        self.remember_view_button.setEnabled(False)
        self.remember_view_button.clicked.connect(self.remember_view)
        self.restore_view_button = QPushButton("恢复记住的视角")
        self.restore_view_button.setEnabled(False)
        self.restore_view_button.clicked.connect(self.restore_saved_view)
        view_controls.addWidget(self.remember_view_button)
        view_controls.addWidget(self.restore_view_button)
        self.save_view_button = QPushButton("保存视角文件")
        self.save_view_button.setEnabled(False)
        self.save_view_button.clicked.connect(self.save_view_file)
        self.load_view_button = QPushButton("读取视角文件")
        self.load_view_button.setEnabled(False)
        self.load_view_button.clicked.connect(self.load_view_file)
        view_controls.addWidget(self.save_view_button)
        view_controls.addWidget(self.load_view_button)
        view_controls.addStretch()

        workspace_header = QHBoxLayout()
        workspace_title = QLabel("三维工作区")
        workspace_title.setStyleSheet("font-size: 17px; font-weight: bold;")
        workspace_header.addWidget(workspace_title)
        workspace_header.addStretch()
        tools_toggle = QCheckBox("展开工具")
        tools_toggle.setChecked(True)
        workspace_header.addWidget(tools_toggle)
        right_layout.addLayout(workspace_header)
        self.tools_panel = self.create_voxel_controls()
        tools_toggle.toggled.connect(self.tools_panel.setVisible)
        right_layout.addWidget(self.tools_panel)
        for name, controls in (("显示", display_controls), ("视角", view_controls)):
            page = QWidget()
            page.setLayout(controls)
            self.tool_tabs.addTab(page, name)
        descriptions = [
            "1 · 调整体素边长，生成处理预览。",
            "2 · 可直接对原始点云去噪，也可先降采样；结果可检查、导出。",
            "3 · 可直接估计原始点云法向量，或处理当前结果。",
            "调整点大小、背景和坐标轴，或保存截图。",
            "记住观察角度，或保存、读取视角文件。",
        ]
        self.tool_help = QLabel(descriptions[0])
        self.tool_help.setWordWrap(True)
        self.tool_help.setObjectName("cardDescription")
        self.tools_panel.layout().insertWidget(1, self.tool_help)
        self.tool_tabs.currentChanged.connect(
            lambda index: self.tool_help.setText(descriptions[index])
        )
        right_layout.addWidget(self.viewport, 1)
        right_layout.addWidget(QLabel("左键拖动旋转 · 滚轮缩放 · Ctrl + 左键拖动平移"))

        # ===== 合并左右布局 =====

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMinimumWidth(240)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(260)
        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 980])
        main_layout.addWidget(splitter)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.statusBar().showMessage("就绪 · 打开点云开始处理")

        # ===== 设置简单样式 =====

        self.open_button.setObjectName("primary")
        self.downsample_button.setObjectName("primary")
        self.export_button.setObjectName("primary")
        self.setStyleSheet("""
            QMainWindow, QWidget { font-family: "Microsoft YaHei UI"; font-size: 12px; }
            QMainWindow { background: #f1f5f9; }
            QLabel, QCheckBox { color: #243247; }
            QPushButton {
                background: #ffffff; color: #243247; border: 1px solid #cbd5e1;
                border-radius: 5px; padding: 7px 10px; min-height: 18px;
            }
            QPushButton:hover { background: #eff6ff; border-color: #60a5fa; }
            QPushButton:pressed { background: #dbeafe; }
            QPushButton:focus, QLineEdit:focus, QSpinBox:focus,
            QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #2563eb; }
            QCheckBox { spacing: 7px; padding: 3px 0; }
            QToolTip { background: #ffffff; color: #243247; border: 1px solid #cbd5e1; }
            QPushButton:disabled { color: #94a3b8; background: #f1f5f9; border-color: #e2e8f0; }
            QPushButton#primary { background: #2563eb; color: white; border-color: #2563eb; }
            QPushButton#primary:hover { background: #1d4ed8; }
            QPushButton#primary:disabled { background: #cbd5e1; border-color: #cbd5e1; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: white; color: #243247; border: 1px solid #cbd5e1;
                border-radius: 4px; padding: 5px; min-height: 20px;
            }
            QFrame#propertyCard, QFrame#toolsCard {
                background: #ffffff; border: 1px solid #dfe6ef; border-radius: 10px;
            }
            QLabel#cardTitle {
                font-size: 14px; font-weight: 600; color: #172b46;
                background: transparent; border: none; padding: 0;
            }
            QLabel#cardDescription {
                color: #738096; background: transparent; border: none; padding: 0;
            }
            QLabel#fileSummary {
                color: #64748b; padding: 8px 0; background: transparent;
            }
            QTabWidget::pane { border: none; background: white; }
            QTabWidget { background: white; }
            QTabBar::tab { padding: 8px 16px; color: #64748b; background: #f8fafc; border: none; border-bottom: 2px solid transparent; }
            QTabBar::tab:selected { color: #1d4ed8; background: white; border-bottom: 2px solid #2563eb; }
            QScrollArea { border: none; background: transparent; }
            QSplitter::handle { background: #e2e8f0; width: 5px; }
            QStatusBar { color: #64748b; }
        """)


    def create_voxel_controls(self):
        group = QFrame()
        group.setObjectName("toolsCard")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        tools_title = QLabel("点云工具")
        tools_title.setObjectName("cardTitle")
        layout.addWidget(tools_title)
        self.tool_tabs = QTabWidget()
        sampling_page = QWidget()
        sampling_layout = QVBoxLayout(sampling_page)
        sampling_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        denoise_page = QWidget()
        denoise_layout = QVBoxLayout(denoise_page)
        normals_page = QWidget()
        normals_layout = QVBoxLayout(normals_page)
        self.tool_tabs.addTab(sampling_page, "降采样")
        self.tool_tabs.addTab(denoise_page, "去噪")
        self.tool_tabs.addTab(normals_page, "法向量")
        layout.addWidget(self.tool_tabs)
        controls = QHBoxLayout()

        size_label = QLabel("体素边长")
        size_label.setMinimumWidth(75)
        self.voxel_size_input = QLineEdit("0.05")
        self.voxel_size_input.setMaximumWidth(220)
        self.voxel_size_input.setToolTip("使用点云的坐标单位")
        size_label.setBuddy(self.voxel_size_input)
        controls.addWidget(size_label)
        controls.addWidget(self.voxel_size_input, 1)

        self.downsample_button = QPushButton("预览降采样")
        self.downsample_button.clicked.connect(self.preview_downsample)
        self.voxel_size_input.returnPressed.connect(self.downsample_button.click)
        self.downsample_button.setEnabled(False)
        controls.addWidget(self.downsample_button)

        self.restore_button = QPushButton("恢复原始点云")
        self.restore_button.clicked.connect(self.restore_original)
        self.restore_button.setEnabled(False)
        controls.addWidget(self.restore_button)

        self.export_button = QPushButton("导出当前点云")
        self.export_button.clicked.connect(self.export_downsample)
        self.export_button.setEnabled(False)
        controls.addWidget(self.export_button)

        self.voxel_status = QLabel("请先打开点云文件。")
        self.voxel_status.setWordWrap(True)
        controls.removeWidget(self.restore_button)
        controls.removeWidget(self.export_button)
        sampling_layout.addLayout(controls)
        actions = QHBoxLayout()
        actions.addWidget(self.restore_button)
        actions.addWidget(self.export_button)
        layout.addLayout(actions)
        layout.addWidget(self.voxel_status)
        filter_controls = QHBoxLayout()
        filter_controls.addWidget(QLabel("统计去噪 · 邻居数"))
        self.neighbor_input = QSpinBox()
        self.neighbor_input.setRange(2, 1000)
        self.neighbor_input.setValue(20)
        filter_controls.addWidget(self.neighbor_input)
        filter_controls.addWidget(QLabel("标准差倍数"))
        self.std_ratio_input = QDoubleSpinBox()
        self.std_ratio_input.setRange(0.1, 10.0)
        self.std_ratio_input.setSingleStep(0.1)
        self.std_ratio_input.setValue(2.0)
        filter_controls.addWidget(self.std_ratio_input)
        self.filter_button = QPushButton("预览去噪")
        self.filter_button.setEnabled(False)
        self.filter_button.clicked.connect(self.preview_outlier_removal)
        filter_controls.addWidget(self.filter_button)
        denoise_layout.addLayout(filter_controls)
        radius_controls = QHBoxLayout()
        radius_controls.addWidget(QLabel("半径去噪 · 搜索半径"))
        self.radius_input = QDoubleSpinBox()
        self.radius_input.setDecimals(6)
        self.radius_input.setRange(0.000001, 1000000)
        self.radius_input.setValue(0.1)
        self.radius_input.setSingleStep(0.01)
        self.radius_input.setToolTip("使用点云的坐标单位")
        radius_controls.addWidget(self.radius_input)
        radius_controls.addWidget(QLabel("最少点数"))
        self.radius_points_input = QSpinBox()
        self.radius_points_input.setRange(1, 10000)
        self.radius_points_input.setValue(5)
        radius_controls.addWidget(self.radius_points_input)
        self.radius_filter_button = QPushButton("预览半径去噪")
        self.radius_filter_button.setEnabled(False)
        self.radius_filter_button.clicked.connect(
            lambda: self.preview_outlier_removal(radius_filter=True)
        )
        radius_controls.addWidget(self.radius_filter_button)
        denoise_layout.addLayout(radius_controls)
        self.removed_checkbox = QCheckBox("检查移除点（保留点灰色，移除点红色）")
        self.removed_checkbox.setEnabled(False)
        self.removed_checkbox.toggled.connect(self.toggle_removed_points)
        removed_controls = QHBoxLayout()
        removed_controls.addWidget(self.removed_checkbox)
        self.export_removed_button = QPushButton("导出红色移除点")
        self.export_removed_button.setEnabled(False)
        self.export_removed_button.clicked.connect(self.export_removed)
        removed_controls.addWidget(self.export_removed_button)
        denoise_layout.addLayout(removed_controls)
        normal_controls = QHBoxLayout()
        normal_controls.addWidget(QLabel("法向量邻居数"))
        self.normal_neighbors = QSpinBox()
        self.normal_neighbors.setRange(3, 1000)
        self.normal_neighbors.setValue(30)
        normal_controls.addWidget(self.normal_neighbors)
        self.normal_button = QPushButton("估计当前结果法向量")
        self.normal_button.setEnabled(False)
        self.normal_button.clicked.connect(self.estimate_current_normals)
        normal_controls.addWidget(self.normal_button)
        self.normal_checkbox = QCheckBox("显示法向量")
        self.normal_checkbox.toggled.connect(self.viewport.set_normals_visible)
        normal_controls.addWidget(self.normal_checkbox)
        normals_layout.addLayout(normal_controls)
        self.orient_normals_button = QPushButton("法向量朝向当前相机")
        self.orient_normals_button.setEnabled(False)
        self.orient_normals_button.clicked.connect(self.orient_current_normals)
        orientation_controls = QHBoxLayout()
        orientation_controls.addWidget(self.orient_normals_button)
        self.consistent_normals_button = QPushButton("邻域一致法向量定向")
        self.consistent_normals_button.setEnabled(False)
        self.consistent_normals_button.clicked.connect(
            lambda: self.orient_current_normals(consistent=True)
        )
        self.consistent_normals_button.setToolTip("使用上方的法向量邻居数")
        orientation_controls.addWidget(self.consistent_normals_button)
        normals_layout.addLayout(orientation_controls)
        return group

    def run_background(self, title, operation, completion):
        if self._job is not None:
            return
        self._completion = completion
        self._task_failed = False
        self.centralWidget().setEnabled(False)
        self.progress.show()
        self.statusBar().showMessage(title + "…")
        self.logger.info("开始 %s", title)
        job = BackgroundJob(operation, self)
        self._job = job
        job.succeeded.connect(self.complete_background)
        job.failed.connect(self.fail_background)
        job.finished.connect(self.finish_background)
        job.start()

    @Slot(object)
    def complete_background(self, result):
        if self._closing_requested:
            return
        try:
            self._completion(result)
            self.refresh_operation_buttons()
        except Exception as error:
            self.logger.exception("更新处理结果失败")
            self.fail_background(str(error))

    @Slot(str)
    def fail_background(self, message):
        self._task_failed = True
        if not self._closing_requested:
            QMessageBox.critical(self, "处理失败", f"{message}\n\n错误日志：{self.log_path}")

    @Slot()
    def finish_background(self):
        job = self._job
        self._job = None
        self._completion = None
        self.progress.hide()
        self.centralWidget().setEnabled(True)
        self.statusBar().showMessage("处理失败，请查看错误日志" if self._task_failed else "处理完成", 5000)
        job.deleteLater()
        if self._closing_requested:
            QTimer.singleShot(0, self.close)

    def open_file(self):
        if self._job is not None:
            return
        """打开文件选择窗口"""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择三维点云文件",
            str(Path.cwd()),
            "点云文件 (*.ply *.pcd *.xyz)"
        )

        if not file_path:
            return

        def calculate():
            cloud = load_point_cloud(file_path)
            return cloud

        def apply_result(value):
            cloud = value

            self.viewport.show_cloud(cloud)

            self.saved_view = None
            self.remember_view_button.setEnabled(True)
            self.save_view_button.setEnabled(True)
            self.load_view_button.setEnabled(True)
            self.restore_view_button.setEnabled(False)
            self.current_file = file_path
            self.current_cloud = cloud
            self.file_label.setText(f"当前文件：\n{Path(file_path).name}")
            self.show_cloud_info(cloud)
            self.reset_view_button.setEnabled(True)
            self.downsample_button.setEnabled(True)
            self.show_original_status()

        self.run_background("读取点云", calculate, apply_result)

    def preview_downsample(self):
        if self._job is not None:
            return
        try:
            voxel_size = float(self.voxel_size_input.text())
            if not math.isfinite(voxel_size) or voxel_size <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "体素边长无效", "请输入大于 0 的有限数值，例如 0.05。")
            return

        source_cloud = self.current_cloud
        def calculate():
            sampled_cloud = source_cloud.voxel_down_sample(voxel_size)
            if sampled_cloud.is_empty():
                raise ValueError("降采样结果为空，请调整体素边长。")
            return sampled_cloud

        def apply_result(value):
            sampled_cloud = value
            self.viewport.show_cloud(sampled_cloud)

            self.clear_removed_preview()
            self.normal_button.setEnabled(True)
            self.filter_source = sampled_cloud
            self.filter_button.setEnabled(True)
            self.radius_filter_button.setEnabled(True)
            self.sampled_cloud = sampled_cloud
            self.orient_normals_button.setEnabled(sampled_cloud.has_normals())
            self.consistent_normals_button.setEnabled(sampled_cloud.has_normals())
            self.export_button.setEnabled(True)
            original_count = len(self.current_cloud.points)
            sampled_count = len(sampled_cloud.points)
            self.voxel_status.setText(
                f"体素边长 {voxel_size:g} · 原始 {original_count:,} 点 → "
                f"预览 {sampled_count:,} 点（保留 {sampled_count / original_count:.1%}）"
            )
            self.restore_button.setEnabled(True)

        self.run_background("体素降采样", calculate, apply_result)

    def preview_outlier_removal(self, radius_filter=False):
        if self._job is not None:
            return
        source = self.filter_source
        if source is None:
            return
        neighbors = self.neighbor_input.value()
        if not radius_filter and len(source.points) <= neighbors:
            QMessageBox.warning(
                self, "点数不足", "邻居数需要小于本次输入点数，请减小邻居数。"
            )
            return
        source_name = "原始点云" if source is self.current_cloud else "降采样结果"
        radius = self.radius_input.value()
        minimum_points = self.radius_points_input.value()
        std_ratio = self.std_ratio_input.value()
        def calculate():
            if radius_filter:
                cleaned, indices = source.remove_radius_outlier(
                    nb_points=minimum_points,
                    radius=radius,
                )
                method = (
                    f"半径去噪（半径 {radius:g}，"
                    f"最少点数 {minimum_points}）"
                )
            else:
                cleaned, indices = source.remove_statistical_outlier(
                    nb_neighbors=neighbors, std_ratio=std_ratio
                )
                method = "统计去噪"
            if cleaned.is_empty():
                raise ValueError("去噪结果为空。统计去噪可增大标准差倍数；半径去噪可增大半径或减小最少点数。")
            removed = source.select_by_index(indices, invert=True)
            removed.paint_uniform_color([1, 0, 0])
            return cleaned, removed, method

        def apply_result(value):
            cleaned, removed, method = value
            self.viewport.show_cloud(cleaned)
            self.clear_removed_preview()
            self.removed_cloud = removed
            self.removed_checkbox.setEnabled(not removed.is_empty())
            self.export_removed_button.setEnabled(not removed.is_empty())
            self.normal_button.setEnabled(True)
            self.restore_button.setEnabled(True)
            self.sampled_cloud = cleaned
            self.orient_normals_button.setEnabled(cleaned.has_normals())
            self.consistent_normals_button.setEnabled(cleaned.has_normals())
            self.export_button.setEnabled(True)
            self.voxel_status.setText(
                f"原始 {len(self.current_cloud.points):,} 点 · "
                f"输入：{source_name} {len(source.points):,} 点 → {method} {len(cleaned.points):,} 点 · "
                f"移除 {len(source.points) - len(cleaned.points):,} 点"
            )

        self.run_background("点云去噪", calculate, apply_result)

    def clear_removed_preview(self):
        self.removed_checkbox.blockSignals(True)
        self.removed_checkbox.setChecked(False)
        self.removed_checkbox.blockSignals(False)
        self.removed_checkbox.setEnabled(False)
        self.export_removed_button.setEnabled(False)
        self.removed_cloud = None

    def toggle_removed_points(self, visible):
        if self.removed_cloud is None or self.sampled_cloud is None:
            return
        try:
            view = self.viewport.get_view_status()
            display = self.sampled_cloud
            if visible:
                display = deepcopy(self.sampled_cloud)
                display.paint_uniform_color([0.7, 0.7, 0.7])
                display += self.removed_cloud
            self.viewport.show_cloud(display)
            self.viewport.restore_view_status(view)
        except Exception as error:
            self.removed_checkbox.blockSignals(True)
            self.removed_checkbox.setChecked(not visible)
            self.removed_checkbox.blockSignals(False)
            QMessageBox.critical(self, "移除点显示失败", str(error))

    def estimate_current_normals(self):
        if self._job is not None:
            return
        source = self.active_cloud()
        if source is None:
            return
        neighbors = self.normal_neighbors.value()
        if len(source.points) < neighbors:
            QMessageBox.warning(self, "点数不足", "法向量邻居数不能超过当前结果的点数。")
            return
        def calculate():
            result = deepcopy(source)
            result.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighbors)
            )
            normals = np.asarray(result.normals)
            if not result.has_normals() or not np.isfinite(normals).all():
                raise ValueError("无法获得有效法向量。")
            return result

        def apply_result(value):
            result = value
            view = self.viewport.get_view_status()
            self.viewport.show_cloud(result)
            self.viewport.restore_view_status(view)
            self.sampled_cloud = result
            self.orient_normals_button.setEnabled(result.has_normals())
            self.consistent_normals_button.setEnabled(result.has_normals())
            self.clear_removed_preview()
            self.normal_checkbox.setChecked(True)
            self.voxel_status.setText(
                f"当前结果 {len(result.points):,} 点 · 已估计法向量 · 邻居数 {neighbors}"
            )

        self.run_background("估计法向量", calculate, apply_result)

    def orient_current_normals(self, consistent=False):
        if self._job is not None:
            return
        source = self.active_cloud()
        if source is None or not source.has_normals():
            return
        neighbors = self.normal_neighbors.value()
        try:
            location = None if consistent else self.viewport.get_camera_position()
        except Exception as error:
            self.logger.exception("读取相机位置失败")
            QMessageBox.critical(self, "法向量定向失败", str(error))
            return
        def calculate():
            result = deepcopy(source)
            if consistent:
                if len(source.points) <= neighbors:
                    raise ValueError("一致定向的邻居数必须小于当前结果点数。")
                result.orient_normals_consistent_tangent_plane(k=neighbors)
                description = f"已进行邻域一致定向 · 邻居数 {neighbors}"
            else:
                result.orient_normals_towards_camera_location(location)
                description = "法向量已朝向点击时的相机位置"
            return result, description

        def apply_result(value):
            result, description = value
            view = self.viewport.get_view_status()
            self.viewport.show_cloud(result)
            self.viewport.restore_view_status(view)
            self.sampled_cloud = result
            self.clear_removed_preview()
            self.normal_checkbox.setChecked(True)
            self.voxel_status.setText(
                f"当前结果 {len(result.points):,} 点 · {description}"
            )

        self.run_background("法向量定向", calculate, apply_result)

    def active_cloud(self):
        return self.sampled_cloud if self.sampled_cloud is not None else self.current_cloud

    def refresh_operation_buttons(self):
        cloud = self.active_cloud()
        available = cloud is not None and not cloud.is_empty()
        has_normals = available and cloud.has_normals()
        self.normal_button.setEnabled(available)
        self.export_button.setEnabled(available)
        self.orient_normals_button.setEnabled(has_normals)
        self.consistent_normals_button.setEnabled(has_normals)
        self.normal_checkbox.setEnabled(has_normals)
        if not has_normals:
            self.normal_checkbox.setChecked(False)
        self.restore_button.setEnabled(self.sampled_cloud is not None)

    def restore_original(self):
        try:
            self.viewport.show_cloud(self.current_cloud)
        except Exception as error:
            QMessageBox.critical(self, "恢复原始点云失败", str(error))
            return

        self.show_original_status()

    def show_original_status(self):
        self.consistent_normals_button.setEnabled(False)
        self.orient_normals_button.setEnabled(False)
        self.normal_button.setEnabled(False)
        self.clear_removed_preview()
        self.filter_source = self.current_cloud
        self.filter_button.setEnabled(self.current_cloud is not None)
        self.radius_filter_button.setEnabled(self.current_cloud is not None)
        self.sampled_cloud = None
        self.export_button.setEnabled(False)
        self.voxel_status.setText(f"正在显示原始点云 · {len(self.current_cloud.points):,} 点")
        self.restore_button.setEnabled(False)
        self.refresh_operation_buttons()

    def export_downsample(self):
        self.export_cloud(self.active_cloud(), "导出当前点云", "_export")

    def export_removed(self):
        self.export_cloud(self.removed_cloud, "导出红色移除点", "_removed")

    def export_cloud(self, cloud, title, filename_suffix):
        if cloud is None or cloud.is_empty():
            return

        formats = {
            "PLY（可保存颜色和法向量） (*.ply)": ".ply",
            "PCD（可保存颜色和法向量） (*.pcd)": ".pcd",
            "XYZ（仅坐标） (*.xyz)": ".xyz",
        }
        source = Path(self.current_file)
        suggested = source.with_name(f"{source.stem}{filename_suffix}")
        name, selected_filter = QFileDialog.getSaveFileName(
            self, title, str(suggested), ";;".join(formats),
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not name:
            return

        destination = Path(name)
        if not destination.suffix:
            destination = destination.with_suffix(formats.get(selected_filter, ".ply"))
        suffix = destination.suffix.lower()
        if suffix not in formats.values():
            QMessageBox.warning(self, "不支持的导出格式", "请使用 .ply、.pcd 或 .xyz 扩展名。")
            return

        try:
            if destination.resolve() == source.resolve():
                QMessageBox.warning(self, "请选择其他文件名", "导出结果不能覆盖当前源文件。")
                return

            overwrite = destination.exists() or destination.is_symlink()
            warnings = []
            if suffix == ".xyz" and (cloud.has_colors() or cloud.has_normals()):
                warnings.append("XYZ 只保存坐标，导出时会丢弃颜色和法向量。")
            if overwrite:
                warnings.append(f"目标文件已存在。要替换它吗？\n{destination}")
            if warnings:
                answer = QMessageBox.question(
                    self, "确认导出", "\n\n".join(warnings),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return

            save_point_cloud(destination, cloud, overwrite=overwrite)
        except Exception as error:
            QMessageBox.critical(self, "导出失败", str(error))
            return

        QMessageBox.information(
            self, "导出成功", f"已保存 {len(cloud.points):,} 个点。\n{destination}"
        )

    def show_cloud_info(self, cloud):
        """显示点云信息"""

        points = np.asarray(cloud.points)

        min_bound = cloud.get_min_bound()
        max_bound = cloud.get_max_bound()

        size = max_bound - min_bound

        info = (
            f"文件名称：{Path(self.current_file).name}\n\n"
            f"点云数量：{len(points):,}\n\n"
            f"数据维度：{points.shape}\n\n"
            f"是否有颜色：{cloud.has_colors()}\n\n"
            f"是否有法向量：{cloud.has_normals()}\n\n"
            f"中心坐标：\n{cloud.get_center()}\n\n"
            f"最小坐标：\n{min_bound}\n\n"
            f"最大坐标：\n{max_bound}\n\n"
            f"包围盒尺寸：\n{size}"
        )

        def coordinate_text(values):
            return "<br>".join(
                f"{axis}&nbsp; {float(value):,.5f}" for axis, value in zip("XYZ", values)
            )

        rows = [
            ("点数", f"{len(points):,}"),
            ("颜色", "有" if cloud.has_colors() else "无"),
            ("法向量", "有" if cloud.has_normals() else "无"),
            ("中心坐标", coordinate_text(cloud.get_center())),
            ("最小坐标", coordinate_text(min_bound)),
            ("最大坐标", coordinate_text(max_bound)),
            ("包围盒尺寸", coordinate_text(size)),
        ]
        table = "".join(
            f'<tr><td style="color:#64748b; padding:6px 8px 6px 0;">{label}</td>'
            f'<td style="padding:6px 0;">{value}</td></tr>'
            for label, value in rows
        )
        self.info_label.setText(
            f'<b>{escape(Path(self.current_file).name)}</b><br><br>'
            f'<table cellspacing="0" width="100%">{table}</table>'
        )
        self.info_label.setToolTip(info)
        self.file_label.setToolTip(str(self.current_file))

    def change_axis_visibility(self, visible):
        try:
            self.viewport.set_axis_visible(visible)
        except Exception as error:
            self.axis_checkbox.blockSignals(True)
            self.axis_checkbox.setChecked(not visible)
            self.axis_checkbox.blockSignals(False)
            QMessageBox.critical(self, "坐标轴切换失败", str(error))

    def save_screenshot(self):
        if self.current_cloud is None:
            QMessageBox.information(self, "请先打开点云", "打开点云并调整视角后，即可保存截图。")
            return

        suggested = Path(self.current_file).with_name(
            f"{Path(self.current_file).stem}_view.png"
        )
        name, _ = QFileDialog.getSaveFileName(
            self, "保存视口截图", str(suggested), "PNG 图片 (*.png)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not name:
            return
        path = Path(name)
        if not path.suffix:
            path = path.with_suffix(".png")
        if path.suffix.lower() != ".png":
            QMessageBox.warning(self, "图片格式无效", "请使用 .png 扩展名。")
            return
        if path.exists():
            answer = QMessageBox.question(
                self, "替换截图", f"目标文件已存在，要替换它吗？\n{path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        output = QSaveFile(str(path))
        try:
            image = self.viewport.capture_image()
            if not output.open(QIODevice.OpenModeFlag.WriteOnly):
                raise OSError(output.errorString())
            if not image.save(output, "PNG"):
                raise OSError("PNG 图片编码失败。")
            if not output.commit():
                raise OSError(output.errorString())
        except Exception as error:
            output.cancelWriting()
            QMessageBox.critical(self, "截图保存失败", str(error))
            return

        QMessageBox.information(
            self, "截图已保存", f"{image.width()} × {image.height()} 像素\n{path}"
        )

    def save_view_file(self):
        name, _ = QFileDialog.getSaveFileName(
            self, "保存当前视角", str(Path(self.current_file).with_suffix(".view.json")),
            "视角文件 (*.json)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not name:
            return
        path = Path(name)
        if not path.suffix:
            path = path.with_suffix(".json")
        if path.suffix.lower() != ".json":
            QMessageBox.warning(self, "文件格式无效", "请使用 .json 扩展名。")
            return
        if path.exists():
            answer = QMessageBox.question(
                self, "替换视角文件", f"目标文件已存在，要替换它吗？\n{path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        output = QSaveFile(str(path))
        try:
            content = self.viewport.get_view_status().encode("utf-8")
            if not output.open(QIODevice.OpenModeFlag.WriteOnly):
                raise OSError(output.errorString())
            if output.write(content) != len(content) or not output.commit():
                raise OSError(output.errorString())
        except Exception as error:
            output.cancelWriting()
            QMessageBox.critical(self, "保存视角失败", str(error))
            return
        self.statusBar().showMessage(f"已保存当前视角到 {path}", 5000)

    def load_view_file(self):
        name, _ = QFileDialog.getOpenFileName(
            self, "读取视角（请先打开对应的点云）",
            str(Path(self.current_file).parent), "视角文件 (*.json)",
        )
        if not name:
            return
        try:
            with Path(name).open("rb") as stream:
                content = stream.read(65537)
            if len(content) > 65536:
                raise ValueError("视角文件过大。")
            data = json.loads(content.decode("utf-8-sig"))
            if (
                data["class_name"] != "ViewTrajectory"
                or data["version_major"] != 1
                or data["version_minor"] != 0
                or len(data["trajectory"]) != 1
            ):
                raise ValueError("需要包含一个视角的 Open3D 视角文件。")
            view = data["trajectory"][0]
            vectors = {}
            for key in ("boundingbox_min", "boundingbox_max", "front", "lookat", "up"):
                vector = np.asarray(view[key], dtype=float)
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError(f"视角参数 {key} 无效。")
                vectors[key] = vector
            if (
                np.any(vectors["boundingbox_max"] < vectors["boundingbox_min"])
                or np.linalg.norm(vectors["front"]) < 1e-12
                or np.linalg.norm(vectors["up"]) < 1e-12
                or np.linalg.norm(np.cross(vectors["front"], vectors["up"])) < 1e-12
                or not 0 < float(view["zoom"]) <= 2
                or not 5 <= float(view["field_of_view"]) <= 90
            ):
                raise ValueError("视角的方向、缩放或范围参数无效。")
            status = json.dumps(data, allow_nan=False)
            self.viewport.restore_view_status(status)
        except Exception as error:
            QMessageBox.critical(self, "读取视角失败", str(error))
            return
        self.saved_view = status
        self.restore_view_button.setEnabled(True)
        self.statusBar().showMessage("已恢复文件中的视角，并更新视角记忆。", 5000)

    def remember_view(self):
        try:
            self.saved_view = self.viewport.get_view_status()
        except Exception as error:
            QMessageBox.critical(self, "记住视角失败", str(error))
            return
        self.restore_view_button.setEnabled(True)
        self.statusBar().showMessage("已记住当前视角，再次点击可更新。", 4000)

    def restore_saved_view(self):
        if self.saved_view is None:
            return
        try:
            self.viewport.restore_view_status(self.saved_view)
        except Exception as error:
            QMessageBox.critical(self, "恢复视角失败", str(error))

    def change_background(self):
        self.viewport.set_background_color(self.background_input.currentData())

    def reset_view(self):
        self.viewport.reset_view()

    def closeEvent(self, event):
        if self._job is not None:
            self._closing_requested = True
            self.statusBar().showMessage("正在等待当前计算完成后关闭…")
            event.ignore()
            return
        if not self.viewport.shutdown():
            event.ignore()
            return
        super().closeEvent(event)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = PointCloudApp()
    window.show()

    sys.exit(app.exec())
