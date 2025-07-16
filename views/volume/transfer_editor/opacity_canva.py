from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib import patches
import numpy as np

"""
以下class還有需要優化的部分
1. 超過200行需要優化
2. 曲線目前沒辦法移動與控制大小
"""


class OpacityCurveCanvas(FigureCanvas):
    def __init__(
        self, points=None, curve_list=None, on_change_callback=None, x_range=(0.0, 1.0)
    ):
        self.fig = Figure(figsize=(4, 1.5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.on_change_callback = on_change_callback

        self.merge_mode = False
        self.dragging_point_index = None
        self.selected_point_index = None

        self.x_min, self.x_max = x_range
        self.points: list[tuple[int, int, int]] = points or [
            (self.x_min, 0.0, (0, 0, 1)),
            (self.x_max, 1.0, (1, 0, 0)),
        ]
        self.curve_list = curve_list if curve_list is not None else []
        self.normal_lines = []
        self.box_list = []
        self.scatter = None
        self._init_plot()
        self._connect_events()

    def _style_ax(self):
        self.ax.margins(0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.axis("off")

    def _init_plot(self):
        self.fig.tight_layout(pad=0)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.line_collection = LineCollection([], linewidths=2, colors="black")
        self.ax.add_collection(self.line_collection)
        self._style_ax()
        self.set_range(self.x_min, self.x_max)

    def _connect_events(self):
        self.mpl_connect("pick_event", self._on_pick)
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_drag)
        self.mpl_connect("button_release_event", self._on_release)

    def _on_pick(self, evt):
        if self.merge_mode or evt.artist is not self.scatter:
            return
        idx = int(evt.ind[0])
        mouse = evt.mouseevent
        if mouse.button == 1:  # 左鍵拖曳起點
            self.dragging_point_index = idx
            self.selected_point_index = idx
            self._notify_change()
        elif mouse.button == 3:  # 右鍵刪除
            if idx not in (0, len(self.points) - 1):
                self.points.pop(idx)
                self.selected_point_index = None
                self._notify_change()

    def _on_click(self, evt):
        """
        只處理左鍵 + 在軸內 + 未 pick 到點 的情況：新增控制點
        """
        if not (not self.merge_mode and evt.inaxes == self.ax and evt.button == 1):
            return
        # 若不是點在既有點上（pick_event 不會設定 dragging_idx），就新增
        if (
            self.dragging_point_index is None
            and evt.xdata is not None
            and evt.ydata is not None
        ):
            x = round(np.clip(evt.xdata, self.x_min + 1e-6, self.x_max - 1e-6), 4)
            y = np.clip(evt.ydata, 0.0, 1.0)
            self.points.append((x, y, (0.3, 0.6, 0.9)))
            self.points.sort(key=lambda p: p[0])
            self.selected_point_index = self.points.index(
                min(self.points, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
            )
            self._notify_change()

    def _on_drag(self, event):
        # 若無拖曳點或滑鼠不在軸內直接返回
        if self.dragging_point_index is None or event.inaxes != self.ax:
            return

        x, y = event.xdata, np.clip(event.ydata, 0.0, 1.0)

        idx = self.dragging_point_index
        if idx == 0:
            x = self.x_min
        elif idx == len(self.points) - 1:
            x = self.x_max
        else:
            # clamp 拖曳點在 (x_min, x_max)
            x = min(max(x, self.x_min + 1e-6), self.x_max - 1e-6)
            left = self.points[idx - 1][0] + 1e-4 if idx > 0 else self.x_min
            right = (
                self.points[idx + 1][0] - 1e-4
                if idx < len(self.points) - 1
                else self.x_max
            )
            x = min(max(x, left), right)
        self.set_point(idx, x, y)
        self._notify_change()

    def _on_release(self, event):
        if self.dragging_point_index is not None and event.ydata is None:
            x_point = self.points[self.selected_point_index][0]
            if event.y > 0:
                self.set_point(self.selected_point_index, x_point, 1)
            elif event.y < 0:
                self.set_point(self.selected_point_index, x_point, 0)
            self._notify_change()

        self.dragging_point_index = None

    def set_point(self, select_index, new_x, new_y):
        _, _, color = self.points[select_index]
        self.points[select_index] = (new_x, new_y, color)

    def _notify_change(self):
        self.update_points(self.points, self.curve_list)
        if self.on_change_callback is not None:
            self.on_change_callback(self.points)

    def update_points(
        self,
        points,
        curve_list=[],
    ):
        self.points = sorted(points, key=lambda p: p[0])
        self.curve_list = curve_list
        # 1. 更新主曲線
        self._update_line_collection()
        # 2. 更新點標記
        self._update_point_artists()
        # 3. 畫曲線
        self._update_curve()
        # 4. 強制刷新
        self._rescale_and_redraw()

    def _update_line_collection(self):
        xs, ys, cs = map(np.array, zip(*self.points))
        if self.merge_mode:
            x_values = np.linspace(self.x_min, self.x_max, 200)
            ys_uniform = list(np.interp(x_values, xs, ys))
            for curve_data in self.curve_list:
                if curve_data["type"] == "normal_curve":
                    mean: float = curve_data["mean"]
                    std: float = curve_data["std"]
                    ys_values: list[float] = [
                        self.get_normal_y(x, mean, std) for x in x_values
                    ]
                    ys_uniform: list[np.float64] = [
                        line_y + curve_y
                        for line_y, curve_y in zip(ys_values, ys_uniform)
                    ]
            segments = [
                [
                    (x_values[i], np.clip(ys_uniform[i], 0, 1)),
                    (x_values[i + 1], np.clip(ys_uniform[i + 1], 0, 1)),
                ]
                for i in range(len(x_values) - 1)
            ]
        else:
            segments = [
                [(xs[i], ys[i]), (xs[i + 1], ys[i + 1])] for i in range(len(xs) - 1)
            ]
        self.line_collection.set_segments(segments)
        self.line_collection.set_colors(["black"] * (len(xs) - 1))

    def _update_point_artists(self):
        # 先移除舊的 artist
        if self.scatter in self.ax.collections:
            self.scatter.remove()

        if self.merge_mode:
            return

        xs, ys, cs = map(np.array, zip(*self.points))
        sizes = np.where(
            np.arange(len(xs)) == (self.selected_point_index or -1), 100, 50
        )
        edge = [
            "white" if i == (self.selected_point_index or -1) else cs[i]
            for i in range(len(cs))
        ]
        self.scatter = self.ax.scatter(
            xs, ys, s=sizes, c=cs, edgecolors=edge, linewidths=1.5, zorder=3, picker=5
        )

    def _update_curve(self):
        # 先清除所有舊 normal 線條
        for line in self.normal_lines:
            line.remove()  # 從 axes 移除這條線
        self.normal_lines = []

        if self.merge_mode:
            return

        for curve_data in self.curve_list:
            if curve_data["type"] == "normal_curve":
                mean = curve_data["mean"]
                std = curve_data["std"]
                x_values = np.linspace(self.x_min, self.x_max, 200)
                y_values = [self.get_normal_y(x, mean, std) for x in x_values]
                (line,) = self.ax.plot(
                    x_values, y_values, "--", color="gray", alpha=0.8
                )
                mid_idx = len(x_values) // 2
                square_size = 0.05
                rect = patches.Rectangle(
                    (x_values[mid_idx], 0),
                    square_size,
                    square_size * 2,
                    color="red",
                    picker=True,
                )
                self.ax.add_patch(rect)
                self.normal_lines.append(line)

        self.ax.relim()

    def _rescale_and_redraw(self):
        self.ax.autoscale_view()
        self.draw()

    def set_range(self, min_val, max_val):
        self.ax.set_xlim(min_val, max_val)
        self.ax.set_ylim(0, 1)

        old_xmin, old_xmax = self.x_min, self.x_max
        self.x_min, self.x_max = min_val, max_val
        ratio = (np.array([x for x, _, _ in self.points]) - old_xmin) / (
            old_xmax - old_xmin or 1
        )

        def reproject(i, x, y, c, r) -> tuple[float, float, float]:
            if i == 0:  # 左端
                return (self.x_min, y, c)
            if i == len(self.points) - 1:  # 右端
                return (self.x_max, y, c)
            return (self.x_min + r * (self.x_max - self.x_min), y, c)

        # 同步端點 x 座標（保持左右端點為 x_min, x_max，內部點依比例調整）
        self.points = [
            reproject(i, x, y, c, r)
            for i, ((x, y, c), r) in enumerate(zip(self.points, ratio))
        ]
        self.update_points(self.points)

    def get_normal_y(self, x, normal_mean, normal_std):
        """
        根據x,mean,std計算常態分佈的y值
        使用公式 y = exp(-0.5 * ((x - mean) / std) ** 2)
        """
        y = np.exp(-0.5 * ((x - normal_mean) / normal_std) ** 2)
        return float(y)
