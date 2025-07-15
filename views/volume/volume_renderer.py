from PySide6.QtWidgets import QWidget, QVBoxLayout
import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkRenderer
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget

from .camera_control_panel import CameraControlPanel


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


class VolumeRenderer(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # 設定VTK
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        # 設定 Renderer
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)  # 全黑背景
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # 設定互動風格：TrackballCamera
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.Initialize()
        interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

        self.camera_panel = CameraControlPanel(interactor, self.renderer)
        self._add_orientation_marker(self.vtk_widget)

        self._volumes: dict[str, dict[str, any]] = {}
        # slice planes cache & actors (shared across volumes)
        self._slice_planes_cache: dict[str, tuple[int, np.ndarray]] = {}
        self._slice_plane_actors: dict[str, vtk.vtkActor | None] = {}

    def update_volume(
        self,
        volume_data: np.ndarray,
        img,
        img_name: str,
        *,
        replace: bool = True,
    ) -> None:
        """Create / update a volume **without triggering a render**.

        The new actor is added to the scene but left *invisible* (visibility
        off).  Call :py:meth:`show_volume` or :py:meth:`refresh_display` when
        you're ready to draw.
        """
        if img_name in self._volumes and replace:
            self.remove_volume(img_name)

        # Prepare VTK image data (XYZ order for VTK)
        vol_xyz = np.ascontiguousarray(np.transpose(volume_data, (2, 1, 0)))
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(vol_xyz.tobytes(), len(vol_xyz.tobytes()))
        importer.SetDataScalarType(numpy_dtype_to_vtk(img.get_data_dtype()))
        importer.SetNumberOfScalarComponents(1)
        importer.SetWholeExtent(
            0,
            vol_xyz.shape[2] - 1,
            0,
            vol_xyz.shape[1] - 1,
            0,
            vol_xyz.shape[0] - 1,
        )
        importer.SetDataExtentToWholeExtent()
        importer.SetDataSpacing(*img.header.get_zooms()[:3])
        importer.Update()

        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputConnection(importer.GetOutputPort())

        color_tf = vtk.vtkColorTransferFunction()
        opacity_tf = vtk.vtkPiecewiseFunction()

        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color_tf)
        prop.SetScalarOpacity(opacity_tf)
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()

        actor = vtk.vtkVolume()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        actor.SetVisibility(False)
        self.renderer.AddVolume(actor)

        self._volumes[img_name] = {
            "actor": actor,
            "prop": prop,
            "color_tf": color_tf,
            "opacity_tf": opacity_tf,
            "visible": False,
        }

    def _add_orientation_marker(self, interactor):
        axes = vtkAxesActor()
        axes.SetTotalLength(50, 50, 50)  # 可依視窗大小調整 XYZ 軸長度

        self.marker = vtkOrientationMarkerWidget()
        self.marker.SetOrientationMarker(axes)
        self.marker.SetInteractor(interactor)
        self.marker.SetViewport(0.0, 0.0, 0.2, 0.2)  # 左下角（調整為合適比例）
        self.marker.SetEnabled(True)
        self.marker.InteractiveOn()

    def show_volume(self, volume_id: str, visible: bool = True) -> None:
        """Toggle visibility of a registered volume and **re‑render**."""
        vol = self._volumes.get(volume_id)
        if vol is None:
            print(f"[VolumeRenderer-show_volume] unknown volume_id: {volume_id}")
            return
        actor: vtk.vtkVolume = vol["actor"]
        actor.SetVisibility(visible)
        vol["visible"] = visible
        if visible:
            self.renderer.ResetCamera()
        self.refresh_display()

    def refresh_display(self) -> None:
        """Force a render of the render‑window."""
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def update_transfer_function(
        self,
        points,
        img_name: str,
    ):
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
        vol = self._volumes.get(img_name)
        if vol is None:
            print(
                f"[VolumeRenderer-update_transfer_function] unknown volume_id: {img_name}"
            )
            return
        color_tf: vtk.vtkColorTransferFunction = vol["color_tf"]
        opacity_tf: vtk.vtkPiecewiseFunction = vol["opacity_tf"]
        visible = vol["visible"]
        self._apply_tf_points(color_tf, opacity_tf, points)
        if visible:
            self.refresh_display()

    @staticmethod
    def _apply_tf_points(
        color_tf: vtk.vtkColorTransferFunction,
        opacity_tf: vtk.vtkPiecewiseFunction,
        points: list[tuple[float, float, tuple[float, float, float]]],
    ) -> None:
        color_tf.RemoveAllPoints()
        opacity_tf.RemoveAllPoints()
        for x, y, (r, g, b) in points:
            opacity_tf.AddPoint(x, y)
            color_tf.AddRGBPoint(x, r, g, b)

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

    def update(self, img, img_name):
        volume = img.get_fdata()
        self.update_volume(volume, img, img_name=img_name)

    def rename_voluem(self, old_img_name, new_img_name):
        self._volumes[new_img_name] = self._volumes[old_img_name]
        del self._volumes[old_img_name]

    def remove_volume(self, volume_id: str) -> None:
        """Remove a volume from the scene and internal dict."""
        vol = self._volumes.pop(volume_id, None)
        if vol and vol["actor"] is not None:
            self.renderer.RemoveVolume(vol["actor"])
            self.vtk_widget.GetRenderWindow().Render()

    def clear_volumes(self) -> None:
        """Remove *all* volumes."""
        for vid in list(self._volumes.keys()):
            self.remove_volume(vid)

    def update_slice_plane(self, view_type, slice_index, img2d, remove=False):
        if not remove:
            self._slice_planes_cache[view_type] = (slice_index, img2d.copy())
        elif view_type in self._slice_planes_cache:
            del self._slice_planes_cache[view_type]
        # 統一顯示所有現有的平面
        self._show_slice_plane(self._slice_planes_cache)

    def _show_slice_plane(
        self, slice_dict: dict[str, tuple[int, np.ndarray]], *, opacity: float = 1.0
    ) -> None:
        """Render multiple oriented 2‑D slices as textured planes in 3‑D

        Parameters
        ----------
        slice_dict
            Mapping from *view type* ("axial", "coronal", "sagittal") to a
            tuple ``(slice_index, img2d)``.  ``img2d`` **must** be a NumPy array.
        opacity
            Actor opacity in the renderer (0 = fully transparent, 1 = opaque).
        """
        if self._volumes is None:
            return  # 沒有 3‑D volume 就無法畫切片

        # 移除舊 plane actors，避免堆疊記憶體 & 佔用畫面。
        for key, actor in self._slice_plane_actors.items():
            if actor is not None:
                self.renderer.RemoveActor(actor)
                self._slice_plane_actors[key] = None  # reset slot

        if not slice_dict:  # 空字典
            self.vtk_widget.GetRenderWindow().Render()
            return

        # 基礎設定 – 軸對應 & volume 邊界 (world coords)。
        axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
        bounds = [0.0] * 6
        ref_actor = next(iter(self._volumes.values()))["actor"]
        bounds = ref_actor.GetBounds()
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        # helper – 建立 (origin, point1, point2) 給 vtkPlaneSource
        def _plane_corners(
            axis: int, idx: int
        ) -> tuple[list[float], list[float], list[float]]:
            """Return three corner points describing an oriented plane."""
            if axis == 0:  # sagittal → X 固定
                return (
                    [idx, y_min, z_min],  # origin
                    [idx, y_max, z_min],  # point1 → Y 方向
                    [idx, y_min, z_max],  # point2 → Z 方向
                )
            if axis == 1:  # coronal → Y 固定
                return (
                    [x_min, idx, z_min],
                    [x_max, idx, z_min],
                    [x_min, idx, z_max],
                )
            # axial or default → Z 固定
            return (
                [x_min, y_min, idx],
                [x_max, y_min, idx],
                [x_min, y_max, idx],
            )

        for view_type, (slice_index, img2d) in slice_dict.items():
            axis = axis_map.get(view_type, 2)  # default axial

            # ------ 5‑1. 建立平面 geometry ------
            origin, p1, p2 = _plane_corners(axis, slice_index)
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(*origin)
            plane.SetPoint1(*p1)
            plane.SetPoint2(*p2)
            plane.Update()

            # ------ 5‑2. 將 2‑D NumPy 影像轉 VTK image ------
            img2d = np.asarray(img2d)  # 保險轉 ndarray
            # 線性映射到 0‑255 → uint8，方便做灰階紋理
            img_norm = np.clip(img2d - img2d.min(), 0, None)
            rng = img_norm.ptp() or 1.0  # ptp = peak‑to‑peak (max‑min)
            img_uint8 = (img_norm / rng * 255).astype(np.uint8)

            importer = vtk.vtkImageImport()
            importer.CopyImportVoidPointer(img_uint8.tobytes(), img_uint8.size)
            importer.SetDataScalarTypeToUnsignedChar()
            importer.SetNumberOfScalarComponents(1)
            importer.SetWholeExtent(
                0, img_uint8.shape[1] - 1, 0, img_uint8.shape[0] - 1, 0, 0
            )
            importer.SetDataExtentToWholeExtent()
            importer.Update()

            # ------ 5‑3. 轉為紋理貼到平面上 ------
            texture = vtk.vtkTexture()
            texture.SetInputConnection(importer.GetOutputPort())
            texture.InterpolateOn()  # 启用雙線性插值，避免像素鋸齒

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetTexture(texture)
            actor.GetProperty().SetOpacity(opacity)

            # ------ 5‑4. 加入 renderer 並保存引用 ------
            self.renderer.AddActor(actor)
            self._slice_plane_actors[view_type] = actor

        self.vtk_widget.GetRenderWindow().Render()
