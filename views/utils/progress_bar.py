from PySide6.QtCore import (
    Qt,
    QRectF,
    QPropertyAnimation,
    Slot,
    QAbstractAnimation,
    QEventLoop,
)
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QGraphicsDropShadowEffect,
    QSizePolicy,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QLinearGradient,
    QBrush,
    QPaintEvent,
    QCloseEvent,
)
from typing import Optional, Tuple, Union

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

ThemeSpec = Union[str, Tuple[int, int, int]]

_THEMES: dict[str, Tuple[int, int, int]] = {
    "ocean": (5, 184, 204),  # teal‑blue
    "sunset": (255, 94, 58),  # orange‑red
    "mint": (30, 214, 149),  # green‑mint
}


def _qcolor(rgb: Tuple[int, int, int]) -> QColor:
    return QColor(*rgb)


# -----------------------------------------------------------------------------
# Custom progress bar that draws centred percentage text
# -----------------------------------------------------------------------------


class FancyProgressBar(QProgressBar):
    """QProgressBar with centred text overlay and rounded‑corner gradient chunk."""

    def __init__(self, colour: QColor, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._colour = colour
        self.setTextVisible(False)  # We'll paint text ourselves
        self.setRange(0, 100)
        self._font = QFont("Noto Sans Mono", 10, QFont.Weight.Bold)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Remove busy flicker by enabling AA for text
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    @Slot(int)
    def set_chunk_colour(self, rgb: Tuple[int, int, int]):
        self._colour = _qcolor(rgb)
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent):  # noqa: N802 – Qt naming
        painter = QPainter(self)
        rect = self.rect().adjusted(1, 1, -1, -1)  # leave 1‑px border

        # Draw background rounded rect
        background = QColor(240, 240, 240, 230)
        painter.setBrush(background)
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawRoundedRect(rect, 10, 10)

        # Draw chunk
        progress_ratio = (self.value() - self.minimum()) / (
            self.maximum() - self.minimum()
        )
        if progress_ratio > 0:
            chunk_rect = QRectF(rect)
            chunk_rect.setWidth(rect.width() * progress_ratio)

            gradient = QLinearGradient(chunk_rect.topLeft(), chunk_rect.topRight())
            colour = self._colour
            gradient.setColorAt(0.0, colour.lighter(120))
            gradient.setColorAt(1.0, colour.darker(110))

            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(chunk_rect, 10, 10)

        # Draw text (%), centred
        painter.setFont(self._font)
        painter.setPen(QColor(40, 40, 40))
        percent = int(progress_ratio * 100)
        text = f"{self.value()}/{self.maximum()} ({percent}\u00a0%)"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        painter.end()


# -----------------------------------------------------------------------------
# Tqdm‑like wrapper widget
# -----------------------------------------------------------------------------


class ProgressBar(QWidget):
    """A modern, themeable progress‑bar HUD."""

    def __init__(
        self,
        total: int = 1,
        desc: str | None = None,
        theme: ThemeSpec = "ocean",
        max_width: int | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        if parent is None:
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowStaysOnTopHint
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if max_width is not None:
            self.setMaximumWidth(max_width)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # ---- widgets ----
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.label = QLabel(desc if desc else "Processing")
        self.label.setFont(QFont("Noto Sans Mono", 11))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self._bar_colour = self._resolve_colour(theme)
        self._bar = FancyProgressBar(self._bar_colour)
        self._bar.setMaximum(total)
        layout.addWidget(self._bar)

        # ---- Drop shadow effect ----
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

        # ---- animation (fade‑in) ----
        self._anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._anim.setDuration(250)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

        # current step
        self._current = 0

    @Slot(int)
    def update(self, n: int = 1):
        self._current += n
        self._bar.setValue(self._current)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

    @Slot()
    def reset(self, total: int = 1, desc: str = "Processing"):
        self._bar.setMaximum(total)
        # keep current in range
        self.set_current(0)
        self.label.setText(desc)
        self.label.setVisible(True)

    @Slot(int)
    def set_current(self, value: int):
        """Jump progress to *value* (0‑based)."""
        self._current = max(self._bar.minimum(), min(value, self._bar.maximum()))
        self._bar.setValue(self._current)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

    def closeEvent(self, event: QCloseEvent):  # noqa: N802 – Qt naming
        # Fade‑out animation
        self._anim.setDirection(QAbstractAnimation.Direction.Backward)
        self._anim.finished.connect(super().close)
        self._anim.start()
        event.ignore()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_colour(self, theme: ThemeSpec) -> QColor:
        if isinstance(theme, tuple):
            r, g, b = theme
            return QColor(r, g, b)
        return QColor(*_THEMES.get(theme.lower(), _THEMES["ocean"]))
