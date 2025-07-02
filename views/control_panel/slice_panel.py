from .base_panel import BasePanel


class SlicePanel(BasePanel):
    def __init__(self, slice_viewers, **kwargs):
        super().__init__(**kwargs)
        self.setWindowTitle("Slice Control Panel")
        self.slice_viewers = slice_viewers

    def update(self, img):
        super().update(img)
        for slice_viewer in self.slice_viewers:
            slice_viewer.update(img)
