from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QDockWidget,
    QTabWidget,
    QSplitter,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QSize, QEvent
import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkRenderer
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget

from ..utils import wrap_with_frame
from .transfer_editor import OpacityEditor
from .histogram_viewer import HistogramViewer
from .camera_control_panel import CameraControlPanel
from .slice_dock import SliceDock


def numpy_dtype_to_vtk(dtype):
    if np.issubdtype(dtype, np.uint8):
        return vtk.VTK_UNSIGNED_CHAR
    elif np.issubdtype(dtype, np.int16):
        return vtk.VTK_SHORT
    elif np.issubdtype(dtype, np.uint16):
        return vtk.VTK_UNSIGNED_SHORT
    elif np.issubdtype(dtype, np.int32):
        return vtk.VTK_INT
    elif np.issubdtype(dtype, np.uint32):
        return vtk.VTK_UNSIGNED_INT
    elif np.issubdtype(dtype, np.float32):
        return vtk.VTK_FLOAT
    elif np.issubdtype(dtype, np.float64):
        return vtk.VTK_DOUBLE
    else:
        return vtk.VTK_FLOAT  # fallback default


class VolumeDock(QDockWidget):
    def __init__(self, title="CT Volume Viewer"):
        super().__init__(title)
        self.container = QWidget()
        self.layout = QVBoxLayout()
        self.tab = QTabWidget()

        self.status_label = QLabel("尚未載入任何資料")

        self.layout.addWidget(wrap_with_frame(self.status_label))
        self.layout.addWidget(self.tab)

        # 設定VTK
        self.vtk_widget = QVTKRenderWindowInteractor(self.container)
        # 設定 Renderer
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)  # 全黑背景
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # 設定互動風格：TrackballCamera
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.Initialize()
        interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

        self.camera_panel = CameraControlPanel(interactor, self.renderer)

        # 設定Slice View
        self.slice_axial = SliceDock("axial", self.update_slice_plane)
        self.slice_coronal = SliceDock("coronal", self.update_slice_plane)
        self.slice_sagittal = SliceDock("sagittal", self.update_slice_plane)
        self.update_list = [self.slice_axial, self.slice_coronal, self.slice_sagittal]
        for slice in self.update_list:
            tweak_slice_dock(slice)

        self._init_control_panel()  # 增加control panel至tab中

        self._init_volume_panel()  # 增加volume panel至tab中

        self.container.setLayout(self.layout)
        self.setWidget(wrap_with_frame(self.container))

        self.volume = None
        self.volume_property = None
        self.slice_planes_cache = {}  # 統一管理全部平面
        self._slice_plane_actors = {}

    def _init_control_panel(self):
        self.histogram_viewer = HistogramViewer()
        self.opacity_editor = OpacityEditor(self.update_transfer_function)
        self.control_panel_widget = QWidget()  # 建立一個容器
        self.control_panel_layout = QVBoxLayout(self.control_panel_widget)
        self.control_panel_layout.addWidget(self.histogram_viewer)
        self.control_panel_layout.addWidget(self.opacity_editor)
        self.tab.addTab(self.control_panel_widget, "Control Panel")

    def _init_volume_panel(self):
        h_split = QSplitter(Qt.Orientation.Horizontal)  # 左/右
        left_v_split = QSplitter(Qt.Orientation.Vertical)  # 左側上/下
        right_v_split = QSplitter(Qt.Orientation.Vertical)  # 右側上/下
        # --------左右 split 合體---------
        h_split.addWidget(left_v_split)
        h_split.addWidget(right_v_split)
        # 加入內容至Split
        left_v_split.addWidget(self.slice_axial)
        left_v_split.addWidget(self.slice_coronal)
        right_v_split.addWidget(self.vtk_widget)
        right_v_split.addWidget(self.slice_sagittal)
        # # --------等比例初始大小 ----------
        # h_split.setSizes([1, 1])
        # left_v_split.setSizes([1, 1])
        # right_v_split.setSizes([1, 1])
        # 加進Tab中
        self.tab.addTab(h_split, "Volume Rendering")
        # self.tab.addTab(self.vtk_widget, "Volume Rendering")

    def update(self, img):
        for update_object in self.update_list:
            update_object.update(img)

        volume = img.get_fdata()

        self.opacity_editor.set_range(volume.min(), volume.max())

        self.histogram_viewer.set_histogram(volume)

        self.status_label.setText(
            f"✅ 成功載入\\nVolume shape: {volume.shape}\\n類型: {img.get_data_dtype()}\\n"
        )
        self.render_volume(volume, img)

    def render_volume(self, volume_data, img):
        volume_data = np.transpose(volume_data, (2, 1, 0))  # Z, Y, X → X, Y, Z
        volume_data = np.ascontiguousarray(volume_data)
        dtype = img.get_data_dtype()
        importer = vtk.vtkImageImport()
        data_string = volume_data.tobytes()
        importer.CopyImportVoidPointer(data_string, len(data_string))
        vtk_type = numpy_dtype_to_vtk(dtype)
        importer.SetDataScalarType(vtk_type)
        importer.SetNumberOfScalarComponents(1)
        importer.SetWholeExtent(
            0,
            volume_data.shape[2] - 1,
            0,
            volume_data.shape[1] - 1,
            0,
            volume_data.shape[0] - 1,
        )
        importer.SetDataExtentToWholeExtent()
        spacing = img.header.get_zooms()
        importer.SetDataSpacing(*spacing[:3])
        importer.Update()

        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputConnection(importer.GetOutputPort())

        self.volume_color = vtk.vtkColorTransferFunction()
        self.volume_scalar_opacity = vtk.vtkPiecewiseFunction()

        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetColor(self.volume_color)
        self.volume_property.SetScalarOpacity(self.volume_scalar_opacity)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(volume_mapper)
        self.volume.SetProperty(self.volume_property)

        render_window = self.vtk_widget.GetRenderWindow()
        self.renderer.AddVolume(self.volume)
        self.renderer.ResetCamera()
        self.update_transfer_function()

    def _add_orientation_marker(self, interactor):
        axes = vtkAxesActor()
        axes.SetTotalLength(50, 50, 50)  # 可依視窗大小調整 XYZ 軸長度

        self.marker = vtkOrientationMarkerWidget()
        self.marker.SetOrientationMarker(axes)
        self.marker.SetInteractor(interactor)
        self.marker.SetViewport(0.0, 0.0, 0.2, 0.2)  # 左下角（調整為合適比例）
        self.marker.SetEnabled(True)
        self.marker.InteractiveOn()

    def update_transfer_function(self):
        """
        # vtkPiecewiseFunction 說明：
        # 它是一種從 scalar value → scalar opacity 的插值函數，
        # 預設情況下對應範圍不限於 [0,1]，但 VTK volume renderer 會以這個函數
        # 返回的 opacity 值作為每個 voxel 的不透明度。
        # 因此：你可以加點 (0, 0.0), (500, 0.2)，代表在 0 為完全透明，500 為稍微不透明。
        # 若加上更多點，例如 (300, 1.0)，則中間會線性過渡到完全不透明。

        # 例如：
        # piecewise.AddPoint(0, 0.0)
        # piecewise.AddPoint(100, 0.1)
        # piecewise.AddPoint(300, 0.8)
        # piecewise.AddPoint(500, 0.0)
        # 表示在 100 到 300 之間漸變為最不透明，500 又變透明
        """

        if not hasattr(self, "volume_color") or not hasattr(
            self, "volume_scalar_opacity"
        ):
            return

        self.volume_scalar_opacity.RemoveAllPoints()
        self.volume_color.RemoveAllPoints()

        points = self.opacity_editor.get_points()
        for x, y, color in points:
            self.volume_scalar_opacity.AddPoint(x, y)
            r, g, b = color
            self.volume_color.AddRGBPoint(x, r, g, b)

        if self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()

    def show_slice_plane(self, slice_dict, opacity=1.0):
        """
        同時顯示多個方向的切片平面。
        slice_dict: { "axial": (slice_index, img2d), "coronal": (...), "sagittal": (...) }
        """

        for view_type in self._slice_plane_actors.keys():
            if self._slice_plane_actors[view_type] is not None:
                self.renderer.RemoveActor(self._slice_plane_actors[view_type])
                self._slice_plane_actors[view_type] = None

        axis_map = {
            "axial": 2,
            "coronal": 1,
            "sagittal": 0,
        }

        if self.volume is None:
            return
        if len(slice_dict) == 0:
            self.vtk_widget.GetRenderWindow().Render()
            return

        bounds = [0] * 6
        self.volume.GetBounds(bounds)
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        for view_type, (slice_index, img2d) in slice_dict.items():

            # 根據 axis 計算 plane 座標
            axis = axis_map.get(view_type, 2)
            # --- 計算平面 ---
            if axis == 0:
                # sagittal: x 固定
                origin = [slice_index, y_min, z_min]
                point1 = [slice_index, y_max, z_min]
                point2 = [slice_index, y_min, z_max]
            elif axis == 1:
                # coronal: y 固定
                origin = [x_min, slice_index, z_min]
                point1 = [x_max, slice_index, z_min]
                point2 = [x_min, slice_index, z_max]
            elif axis == 2:
                # axial: z 固定
                origin = [x_min, y_min, slice_index]
                point1 = [x_max, y_min, slice_index]
                point2 = [x_min, y_max, slice_index]
            else:
                return

            # 建立 VTK 平面
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(*origin)
            plane.SetPoint1(*point1)
            plane.SetPoint2(*point2)
            plane.Update()

            # 將 numpy array 轉 vtkImageData
            img2d = np.array(img2d)
            img_min, img_max = img2d.min(), img2d.max()
            img_range = img_max - img_min if img_max > img_min else 1.0
            img_uint8 = np.clip((img2d - img_min) / img_range * 255, 0, 255).astype(
                np.uint8
            )

            # 將資料拷貝到 VTK
            importer = vtk.vtkImageImport()
            importer.CopyImportVoidPointer(img_uint8.tobytes(), img_uint8.size)
            importer.SetDataScalarTypeToUnsignedChar()
            importer.SetNumberOfScalarComponents(1)
            importer.SetWholeExtent(
                0, img_uint8.shape[1] - 1, 0, img_uint8.shape[0] - 1, 0, 0
            )
            importer.SetDataExtentToWholeExtent()
            importer.Update()
            vtk_img = importer.GetOutput()

            # 轉換為 vtkTexture
            texture = vtk.vtkTexture()
            texture.SetInputData(vtk_img)
            texture.InterpolateOn()

            # 建立 Actor
            plane_mapper = vtk.vtkPolyDataMapper()
            plane_mapper.SetInputConnection(plane.GetOutputPort())
            plane_actor = vtk.vtkActor()
            plane_actor.SetMapper(plane_mapper)
            plane_actor.SetTexture(texture)
            plane_actor.GetProperty().SetOpacity(opacity)

            self._slice_plane_actors[view_type] = plane_actor
            self.renderer.AddActor(plane_actor)

        self.vtk_widget.GetRenderWindow().Render()

    def update_slice_plane(self, view_type, slice_index, img2d, remove=False):
        if not remove:
            self.slice_planes_cache[view_type] = (slice_index, img2d)
        elif view_type in self.slice_planes_cache:
            del self.slice_planes_cache[view_type]
        # 統一顯示所有現有的平面
        self.show_slice_plane(self.slice_planes_cache)

    def prepare_for_exit(self):
        """釋放 VTK OpenGL context、Matplotlib 與子 SliceDock。"""

        if getattr(self, "_already_finalized", False):
            return

        try:
            # 釋放 volume & actors
            if self.renderer is not None:
                self.renderer.RemoveAllViewProps()

            # 關掉 orientation marker
            if hasattr(self, "marker") and self.marker is not None:
                self.marker.SetEnabled(False)
                self.marker = None

            # 終止 VTK interactor & 關閉 RenderWindow
            if self.vtk_widget is not None:
                rw = self.vtk_widget.GetRenderWindow()
                if rw is not None:
                    rw.Finalize()  # 釋放 OpenGL context
                    rw.SetInteractor(None)  # ← 關鍵：防止後面還有人呼叫 Render
                self.vtk_widget.TerminateApp()
                self.vtk_widget.hide()  # 防止 Qt 再 paint
        except Exception as exc:
            # 若仍有 context 失效等例外，不讓它往外拋
            print("VolumeDock _finalize_vtk() error:", exc)

        self._already_finalized = True


def tweak_slice_dock(dock: QDockWidget):
    """
    把 SliceDock 改成：
      • 可伸縮（取消 setFixedSize 效果）
      • 不能關閉、不能浮動，只能拖曳
      • 隱藏標題列 + 外框
    """
    MAXSIZE = 16777215
    # 1) 解除固定大小 → 重設 min/max + size policy
    dock.setMinimumSize(QSize(0, 0))
    dock.setMaximumSize(QSize(MAXSIZE, MAXSIZE))
    dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # 2) 移除 Dock 按鈕（╳ 與 ⛶）
    dock_features = QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
    dock.setFeatures(dock_features)

    # 3) 隱藏標題列 & 外框
    dock.setTitleBarWidget(QWidget(dock))  # 空白 widget → 沒標題列
    dock.setStyleSheet("QDockWidget { border: none; }")  # 去外框
