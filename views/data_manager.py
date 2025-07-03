from PySide6.QtWidgets import QDockWidget
import os
import nibabel as nib
import numpy as np

from .control_panel.base_panel import BasePanel


class DataManager:
    def __init__(self):
        self.imgs = {}  # key: name or path, value: ndarray
        self.current_key = None
        self.observers: list[QDockWidget] = []

    def load_nifti(self, file_path):
        name = os.path.basename(file_path)
        if name not in self.imgs:
            img = nib.load(file_path)
            ornt = nib.orientations.io_orientation(img.affine)
            img_data = img.get_fdata()
            transformed_data = nib.orientations.apply_orientation(img_data, ornt)
            img = nib.Nifti1Image(transformed_data, np.eye(4))  # canonical affine
            self.imgs[name] = img
        self.set_current(name)

    def set_current(self, key):
        if key in self.imgs:
            self.current_key = key
            for obs in self.observers:
                if isinstance(obs, BasePanel) or obs.isVisible():
                    obs.update(self.imgs[key])

    def get_current(self):
        return self.imgs.get(self.current_key, None)

    def remove_img(self, img_name):
        """刪除第 idx 張影像；成功回傳 True，失敗回傳 False"""

        if img_name in self.imgs:
            img = self.imgs.pop(img_name)
            if hasattr(img, "close"):
                img.close()
            if self.current_key == img_name:
                self.current_key = next(iter(self.imgs), None)
            self.set_current(self.current_key)
            return True

        return False

    def add_img(self, image_name, image_data):
        """新增影像到 imgs 字典中"""
        if image_name not in self.imgs:
            self.imgs[image_name] = image_data
        for observer in self.observers:
            if isinstance(observer, BasePanel):
                BasePanel.update(observer, image_data)

    def register(self, observer):
        self.observers.append(observer)
        if self.get_current() is not None:
            # 如果有當前影像，則立即更新觀察者
            observer.update(self.get_current())
