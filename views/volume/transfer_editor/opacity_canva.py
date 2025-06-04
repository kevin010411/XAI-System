from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib import patches
import numpy as np

"""
以下class還有需要優化的部分
1. 超過200行需要優化
2. 目前click事件是對所有FigureCanvas都有效，但實際上使用picker來選擇點會更簡單
3. 曲線目前沒辦法移動與控制大小
"""


class OpacityCurveCanvas(FigureCanvas):
    def __init__(
        self, points=None, curve_list=None, on_change_callback=None, x_range=(0.0, 1.0)
    ):
        self.fig = Figure(figsize=(4, 1.5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.on_change_callback = on_change_callback

        self.display_points = True
        self.dragging_point_index = None
        self.selected_point_index = None
        self.point_artists = []

        self.x_min, self.x_max = x_range
        self.points = (
            points
            if points is not None
            else [(self.x_min, 0.0, (0, 0, 1)), (self.x_max, 1.0, (1, 0, 0))]
        )
        self.curve_list = curve_list if curve_list is not None else []
        self.normal_lines = []
        self.box_list = []
        self._init_plot()
        self._connect_events()

    def _init_plot(self):
        self.fig.tight_layout(pad=0)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.margins(0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.line_collection = LineCollection([], linewidths=2, colors="black")
        self.ax.add_collection(self.line_collection)
        self.set_range(self.x_min, self.x_max)
        self.update_points(self.points)

    def _connect_events(self):
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_drag)
        self.mpl_connect("button_release_event", self._on_release)

    def _find_nearest_point(self, x, y, threshold=15):
        points_disp = np.array(
            [self.ax.transData.transform((px, py)) for px, py, _ in self.points]
        )
        dists = np.sqrt((points_disp[:, 0] - x) ** 2 + (points_disp[:, 1] - y) ** 2)
        min_idx = np.argmin(dists)
        if dists[min_idx] < threshold * (self.x_max - self.x_min):
            return int(min_idx)
        return None

    def _find_nearest_box(self, x, y, threshold=15):
        """
        此function目前沒有被使用
        """
        for i, box in enumerate(self.box_list):
            # box 為 matplotlib Rectangle patch, center xy
            bbox = box.get_bbox()
            cx, cy = bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2
            bx, by = self.ax.transData.transform((cx, cy))
            dist = np.sqrt((bx - x) ** 2 + (by - y) ** 2)
            if dist < min_dist and dist < threshold:
                nearest = {"type": "box", "curve_idx": i, "box": box}
                min_dist = dist
            print(dist)
        return None

    def _on_click(self, event):
        if not self.display_points or event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        idx = self._find_nearest_point(event.x, event.y)
        if event.button == 3:
            if idx is not None and idx != 0 and idx != len(self.points) - 1:
                del self.points[idx]
                self.selected_point_index = None
        if event.button == 1:
            if idx is not None:
                self.dragging_point_index = idx
                self.selected_point_index = idx
            else:
                # clamp 新增點在 (x_min, x_max)
                x = min(max(x, self.x_min + 1e-6), self.x_max - 1e-6)
                y = min(max(y, 0.0), 1.0)
                default_color = (0.3, 0.6, 0.9)
                self.points.append((x, y, default_color))
                self.points.sort(key=lambda p: p[0])
                self.selected_point_index = self.points.index(
                    min(self.points, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
                )
        self._notify_change()

    def _on_drag(self, event):
        if self.dragging_point_index is not None and event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            y = min(max(y, 0.0), 1.0)
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
            self.points = self.set_point(idx, x, y)
            self.selected_point_index = self.points.index(
                min(self.points, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
            )
            self._notify_change()

    def _on_release(self, event):
        if self.dragging_point_index is not None and event.ydata is None:
            x_point = self.points[self.selected_point_index][0]
            if event.y > 0:
                self.points = self.set_point(self.selected_point_index, x_point, 1)
            elif event.y < 0:
                self.points = self.set_point(self.selected_point_index, x_point, 0)
            self._notify_change()

        self.dragging_point_index = None

    def set_point(self, select_index, new_x, new_y):
        new_points = list(self.points)
        _, _, color = new_points[select_index]
        new_points[select_index] = (new_x, new_y, color)
        return sorted(new_points, key=lambda p: p[0])

    def _notify_change(self):
        self.update_points(self.points, self.curve_list)
        if self.on_change_callback is not None:
            self.on_change_callback(self.points)

    def update_points(
        self,
        points,
        curve_list=None,
    ):
        self.points = sorted(points, key=lambda p: p[0])
        if not self.points:
            return
        # 2. 拆解 x/y/color
        xs, ys, cs = self._extract_xyc(self.points)
        # 3. 更新主曲線
        self._update_line_collection(xs, ys)
        # 4. 更新點標記
        self._update_point_artists(xs, ys, cs)
        # 5. 畫 normal_curve
        self._update_curve(curve_list)
        # 6. 強制刷新
        self._rescale_and_redraw()

    def adjust_points_proportionally(self, points, old_xmin, old_xmax):
        """
        回傳依比例調整 x 座標的新 points
        """
        new_points = []
        for i, (x, y, c) in enumerate(points):
            if i == 0:
                new_x = self.x_min
            elif i == len(points) - 1:
                new_x = self.x_max
            else:
                if old_xmax > old_xmin:
                    ratio = (x - old_xmin) / (old_xmax - old_xmin)
                else:
                    ratio = 0.0
                new_x = self.x_min + ratio * (self.x_max - self.x_min)
            new_points.append((new_x, y, c))
        return new_points

    def _extract_xyc(self, points):
        xs, ys, cs = [], [], []
        for x, y, c in points:
            xs.append(x)
            ys.append(y)
            cs.append(c)
        return xs, ys, cs

    def _update_line_collection(self, xs, ys):
        segments = [
            [(xs[i], ys[i]), (xs[i + 1], ys[i + 1])] for i in range(len(xs) - 1)
        ]
        self.line_collection.set_segments(segments)
        self.line_collection.set_colors(["black"] * (len(xs) - 1))

    def _update_point_artists(self, xs, ys, cs):
        # 先移除舊的 artist
        for artist in self.point_artists:
            artist.remove()
        self.point_artists = []
        for i, (x_, y_, c) in enumerate(zip(xs, ys, cs)):
            size = 100 if i == self.selected_point_index else 50
            edge_color = "white" if i == self.selected_point_index else c
            artist = self.ax.scatter(
                x_,
                y_,
                s=size,
                color=[c],
                edgecolors=[edge_color],
                linewidths=1.5,
                zorder=3,
            )
            self.point_artists.append(artist)

    def _update_curve(self, curve_list):
        # 先清除所有舊 normal 線條
        for line in self.normal_lines:
            line.remove()  # 從 axes 移除這條線
        self.normal_lines = []

        for curve_data in curve_list or []:
            if curve_data["type"] == "normal_curve":
                mean = curve_data["mean"]
                std = curve_data["std"]
                x_values = np.linspace(self.x_min, self.x_max, 200)
                y_values = [self.get_normal_y(x, mean, std) for x in x_values]
                (line,) = self.ax.plot(
                    x_values, y_values, "--", color="gray", alpha=0.8
                )
                size = 0.05
                rect = patches.Rectangle(
                    (mean - size / 2, size * 2 - size),
                    size,
                    size * 2,
                    color="red",
                    picker=True,
                )
                self.ax.add_patch(rect)
                self.normal_lines.append(line)

    def _rescale_and_redraw(self):
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()

    def set_range(self, min_val, max_val):
        old_xmin, old_xmax = self.x_min, self.x_max
        self.x_min, self.x_max = min_val, max_val
        self.ax.set_xlim(min_val, max_val)
        self.ax.set_ylim(0, 1)
        # 同步端點 x 座標（保持左右端點為 x_min, x_max，內部點依比例調整）
        self.points = self.adjust_points_proportionally(self.points, old_xmin, old_xmax)
        self.update_points(self.points)
        self.draw()

    def get_normal_y(self, x, normal_mean, normal_std):
        """
        根據x,mean,std計算常態分佈的y值
        使用公式 y = exp(-0.5 * ((x - mean) / std) ** 2)
        """
        y = np.exp(-0.5 * ((x - normal_mean) / normal_std) ** 2)
        return float(y)

    def disable_point_display(self):
        self.display_points = False

    def enable_point_display(self):
        self.display_points = True
