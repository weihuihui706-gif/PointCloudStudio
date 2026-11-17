"""将 Open3D 的 GLFW 窗口放入 Windows 上的 Qt 原生控件。"""

import ctypes
from ctypes import wintypes
import os
from uuid import uuid4

import numpy as np
import open3d as o3d
from PySide6.QtGui import QImage
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def _win32(name, result_type, *argument_types):
    function = getattr(ctypes.WinDLL("user32", use_last_error=True), name)
    function.restype = result_type
    function.argtypes = argument_types
    return function


HWND = wintypes.HWND
find_window = _win32("FindWindowW", HWND, wintypes.LPCWSTR, wintypes.LPCWSTR)
get_window_pid = _win32(
    "GetWindowThreadProcessId", wintypes.DWORD, HWND,
    ctypes.POINTER(wintypes.DWORD)
)
get_style = _win32("GetWindowLongW", wintypes.LONG, HWND, ctypes.c_int)
set_style = _win32("SetWindowLongW", wintypes.LONG, HWND, ctypes.c_int, wintypes.LONG)
set_parent = _win32("SetParent", HWND, HWND, HWND)
get_client_rect = _win32(
    "GetClientRect", wintypes.BOOL, HWND, ctypes.POINTER(wintypes.RECT)
)
set_window_pos = _win32(
    "SetWindowPos", wintypes.BOOL, HWND, HWND,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT
)

GWL_STYLE = -16
WS_CHILD = 0x40000000
WS_POPUP = 0x80000000
WS_OVERLAPPEDWINDOW = 0x00CF0000
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
GLFW_KEY_ESCAPE = 256
GLFW_KEY_Q = 81


def _checked_window_change(function, *args):
    # 旧样式或旧父窗口可能本来就是 0，需要结合 LastError 判断。
    ctypes.set_last_error(0)
    result = function(*args)
    error = ctypes.get_last_error()
    if not result and error:
        raise ctypes.WinError(error)
    return result


class Open3DViewport(QWidget):
    close_ready = Signal()

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setMinimumSize(400, 320)
        self._visualizer = None
        self._child = None
        self._cloud = None
        self._axis = None
        self._axis_visible = True
        self._point_size = 5.0
        self._normals_visible = False
        self._background_color = (0.12, 0.14, 0.18)
        self._polling = False
        self._closing = False

        self._placeholder = QLabel("打开点云文件后在此显示三维视图")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self._placeholder)

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._poll)

    def _start(self):
        host = int(self.winId())
        title = f"PointCloudStudio-viewport-{uuid4().hex}"
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        if not visualizer.create_window(title, width=640, height=480, visible=False):
            raise RuntimeError("Open3D 无法创建 OpenGL 窗口。")

        self._visualizer = visualizer
        try:
            child = find_window(None, title)
            pid = wintypes.DWORD()
            if not child or not get_window_pid(child, ctypes.byref(pid)) or pid.value != os.getpid():
                raise RuntimeError("找不到本进程创建的 Open3D 窗口。")

            style = _checked_window_change(get_style, child, GWL_STYLE)
            style = (style & ~(WS_POPUP | WS_OVERLAPPEDWINDOW)) | WS_CHILD
            _checked_window_change(set_style, child, GWL_STYLE, style)
            _checked_window_change(set_parent, child, host)
            self._child = child
            self._resize_child(SWP_FRAMECHANGED | SWP_SHOWWINDOW)

            # 关闭应用由 Qt 管理。保留 Open3D 其余鼠标与键盘操作。
            visualizer.register_key_callback(GLFW_KEY_ESCAPE, lambda _: False)
            visualizer.register_key_callback(GLFW_KEY_Q, lambda _: False)
            self._axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            if self._axis_visible and not visualizer.add_geometry(self._axis):
                raise RuntimeError("无法将坐标轴添加到三维视口。")

            self.set_background_color(self._background_color)
            self.set_point_size(self._point_size)
            self.set_normals_visible(self._normals_visible)
            self._placeholder.hide()
            self._timer.start()
        except Exception:
            self._destroy()
            raise

    def show_cloud(self, cloud):
        if self._visualizer is None:
            self._start()

        visualizer = self._visualizer
        if not visualizer.add_geometry(cloud, reset_bounding_box=False):
            raise RuntimeError("无法将点云添加到三维视口。")
        if self._cloud is not None:
            if not visualizer.remove_geometry(self._cloud, reset_bounding_box=False):
                visualizer.remove_geometry(cloud, reset_bounding_box=False)
                raise RuntimeError("无法替换当前点云。")

        self._cloud = cloud
        self.reset_view()

    def get_view_status(self):
        if self._visualizer is None or self._cloud is None:
            raise RuntimeError("三维视口尚未准备好，请重新打开点云。")
        return self._visualizer.get_view_status()

    def restore_view_status(self, status):
        if self._visualizer is None or self._cloud is None:
            raise RuntimeError("三维视口尚未准备好，请重新打开点云。")
        self._visualizer.set_view_status(status)
        self._visualizer.update_renderer()

    def get_camera_position(self):
        if self._visualizer is None or self._cloud is None:
            raise RuntimeError("三维视口尚未准备好。")
        control = self._visualizer.get_view_control()
        if control.get_field_of_view() <= 5:
            raise ValueError("当前为正交视图，请重置视角后再进行相机定向。")
        parameters = control.convert_to_pinhole_camera_parameters()
        camera_to_world = np.linalg.inv(parameters.extrinsic)
        location = camera_to_world[:3, 3]
        if not np.isfinite(location).all():
            raise ValueError("无法获得有效相机位置。")
        return location

    def capture_image(self):
        if self._visualizer is None or self._cloud is None:
            raise RuntimeError("三维视口尚未准备好，请重新打开点云。")
        pixels = np.asarray(
            self._visualizer.capture_screen_float_buffer(do_render=True)
        )
        if pixels.size == 0:
            raise RuntimeError("无法读取三维视口图像。")
        rgb = np.ascontiguousarray(np.rint(np.clip(pixels, 0, 1) * 255), dtype=np.uint8)
        height, width, _ = rgb.shape
        # copy 让 QImage 拥有像素，避免局部数组释放后失效。
        return QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()

    def set_normals_visible(self, visible):
        self._normals_visible = bool(visible)
        if self._visualizer is not None:
            self._visualizer.get_render_option().point_show_normal = self._normals_visible
            self._visualizer.update_renderer()

    def set_axis_visible(self, visible):
        visible = bool(visible)
        if visible == self._axis_visible:
            return
        if self._visualizer is not None:
            operation = (
                self._visualizer.add_geometry
                if visible else self._visualizer.remove_geometry
            )
            if not operation(self._axis, reset_bounding_box=False):
                raise RuntimeError("Open3D 无法更新坐标轴。")
            self._visualizer.update_renderer()
        self._axis_visible = visible

    def set_background_color(self, color):
        self._background_color = tuple(color)
        if self._visualizer is not None:
            self._visualizer.get_render_option().background_color = self._background_color
            self._visualizer.update_renderer()

    def set_point_size(self, size):
        self._point_size = float(size)
        if self._visualizer is not None:
            self._visualizer.get_render_option().point_size = self._point_size
            self._visualizer.update_renderer()

    def reset_view(self):
        if self._visualizer is not None:
            self._visualizer.reset_view_point(True)
            self._visualizer.update_renderer()

    def _resize_child(self, extra_flags=0):
        # Qt 布局使用逻辑尺寸，Win32 客户区给出同一 DPI 上下文中的原生尺寸。
        rect = wintypes.RECT()
        if not get_client_rect(int(self.winId()), ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not set_window_pos(
            self._child, None, 0, 0, rect.right, rect.bottom,
            SWP_NOZORDER | SWP_NOACTIVATE | extra_flags
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._child is not None:
            self._resize_child()

    def _poll(self):
        if self._polling:
            return
        self._polling = True
        try:
            running = self._visualizer.poll_events()
        finally:
            self._polling = False

        if self._closing:
            self.close_ready.emit()
        elif not running:
            self._destroy()
            self._placeholder.setText("三维视口已关闭，请重新打开点云文件")
            self._placeholder.show()

    def shutdown(self):
        if self._polling:
            # GLFW 派发事件时也可能收到 Qt 的关闭请求。
            self._closing = True
            return False
        self._destroy()
        return True

    def _destroy(self):
        self._timer.stop()
        if self._visualizer is not None:
            self._visualizer.destroy_window()
        self._visualizer = None
        self._child = None
        self._cloud = None
        self._axis = None
