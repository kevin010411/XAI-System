import math

from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkAnnotatedCubeActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingCore import vtkCamera, vtkRenderer
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkRenderingCore import vtkCameraInterpolator
from PySide6.QtCore import QTimer


class CameraControlPanel:
    def __init__(self, interactor, renderer: vtkRenderer):
        self.renderer = renderer
        self.interactor = interactor
        self.camera = renderer.GetActiveCamera()
        self._init_cube_marker()
        self._init_axes_marker()
        self.anim_timer = QTimer()
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_progress = 0.0
        self.camera_interpolator = vtkCameraInterpolator()

        self.last_click_position = None
        self.interactor.AddObserver(
            vtkCommand.LeftButtonPressEvent, self._on_left_click
        )

    def _init_cube_marker(self):
        self.cube_actor = vtkAnnotatedCubeActor()
        self.cube_actor.SetXPlusFaceText("R")
        self.cube_actor.SetXMinusFaceText("L")
        self.cube_actor.SetYPlusFaceText("A")
        self.cube_actor.SetYMinusFaceText("P")
        self.cube_actor.SetZPlusFaceText("S")
        self.cube_actor.SetZMinusFaceText("I")

        self.cube_actor.GetTextEdgesProperty().SetColor(1, 1, 1)
        self.cube_actor.GetCubeProperty().SetColor(0.4, 0.4, 0.4)

        self.cube_marker = vtkOrientationMarkerWidget()
        self.cube_marker.SetOrientationMarker(self.cube_actor)
        self.cube_marker.SetInteractor(self.interactor)
        self.cube_marker.SetViewport(0.75, 0.75, 1.0, 1.0)
        self.cube_marker.SetEnabled(True)
        self.cube_marker.SetInteractive(False)

    def _init_axes_marker(self):
        axes = vtkAxesActor()
        axes.SetTotalLength(50, 50, 50)

        self.axes_marker = vtkOrientationMarkerWidget()
        self.axes_marker.SetOrientationMarker(axes)
        self.axes_marker.SetInteractor(self.interactor)
        self.axes_marker.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.axes_marker.SetEnabled(True)
        self.axes_marker.SetInteractive(False)

    def _on_left_click(self, caller, event):
        click_pos = self.interactor.GetEventPosition()
        x_norm = click_pos[0] / self.interactor.GetRenderWindow().GetSize()[0]
        y_norm = click_pos[1] / self.interactor.GetRenderWindow().GetSize()[1]
        if 0.75 <= x_norm <= 1.0 and 0.75 <= y_norm <= 1.0:
            self._switch_camera_based_on_cube_direction()

    def _switch_camera_based_on_cube_direction(self):
        camera = self.renderer.GetActiveCamera()
        pos = camera.GetPosition()
        fp = camera.GetFocalPoint()
        direction = [p - f for p, f in zip(pos, fp)]
        abs_dir = list(map(abs, direction))
        idx = abs_dir.index(max(abs_dir))

        new_pos = list(fp)
        dist = (
            (pos[0] - fp[0]) ** 2 + (pos[1] - fp[1]) ** 2 + (pos[2] - fp[2]) ** 2
        ) ** 0.5

        if idx == 0:  # X為主軸
            new_pos[0] += dist if direction[0] > 0 else -dist
            if direction[0] > 0:
                view_up = self.rotate_vector(
                    (0, 1, 0), (1, 0, 0), 90
                )  # R → rotate Y-axis 180
            else:
                view_up = self.rotate_vector(
                    (0, -1, 0), (1, 0, 0), -90
                )  # L → rotate X-axis 180
        elif idx == 1:  # Y為主軸
            new_pos[1] += dist if direction[1] > 0 else -dist
            if direction[1] > 0:
                view_up = self.rotate_vector(
                    (1, 0, 0), (0, 1, 0), 270
                )  # A → rotate Y-axis 180
            else:
                view_up = self.rotate_vector(
                    (-1, 0, 0), (0, 1, 0), 90
                )  # P → rotate Y-axis 180
        else:  # Z為主軸
            new_pos[2] += dist if direction[2] > 0 else -dist
            if direction[1] > 0:
                view_up = (0, 1, 0)  # I → rotate Y-axis 180
            else:
                view_up = self.rotate_vector(
                    (0, -1, 0), (0, 0, 1), 180
                )  # S → rotate Y-axis 180

        self.camera_interpolator.Initialize()
        self.camera_interpolator.AddCamera(0.0, camera)

        temp_camera = vtkCamera()
        temp_camera.DeepCopy(camera)
        temp_camera.SetPosition(*new_pos)
        temp_camera.SetFocalPoint(*fp)
        temp_camera.SetViewUp(*view_up)
        self.camera_interpolator.AddCamera(1.0, temp_camera)

        self.anim_progress = 0.0
        self.anim_timer.start()

    def _on_anim_tick(self):
        self.anim_progress += 0.05
        if self.anim_progress >= 1.0:
            self.anim_progress = 1.0
            self.anim_timer.stop()

        self.camera_interpolator.InterpolateCamera(self.anim_progress, self.camera)
        self.renderer.ResetCameraClippingRange()
        self.interactor.GetRenderWindow().Render()

    def rotate_vector(self, vector, axis, angle_deg):
        angle_rad = math.radians(angle_deg)
        ux, uy, uz = axis
        x, y, z = vector
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)

        return (
            (c + ux * ux * (1 - c)) * x
            + (ux * uy * (1 - c) - uz * s) * y
            + (ux * uz * (1 - c) + uy * s) * z,
            (uy * ux * (1 - c) + uz * s) * x
            + (c + uy * uy * (1 - c)) * y
            + (uy * uz * (1 - c) - ux * s) * z,
            (uz * ux * (1 - c) - uy * s) * x
            + (uz * uy * (1 - c) + ux * s) * y
            + (c + uz * uz * (1 - c)) * z,
        )
