from PySide6.QtWidgets import QDockWidget
from PySide6.QtCore import QStringListModel, Qt
import os
import nibabel as nib
import numpy as np

from .control_panel.base_panel import BasePanel


class DataManager:
    def __init__(self):
        self.imgs = {}  # key: name or path, value: ndarray
        self.img_name_list_model = QStringListModel()
        self.current_key = None
        self.observers: list[QDockWidget] = []

    def add_img_name_to_list_model(self, img_name):
        row = self.img_name_list_model.rowCount()
        self.img_name_list_model.insertRow(row)
        idx = self.img_name_list_model.index(row)
        self.img_name_list_model.setData(idx, img_name)

    def load_nifti(self, file_path):
        name = os.path.basename(file_path)
        if name not in self.imgs:
            img = nib.load(file_path)
            ornt = nib.orientations.io_orientation(img.affine)
            img_data = img.get_fdata()
            transformed_data = nib.orientations.apply_orientation(img_data, ornt)
            img = nib.Nifti1Image(transformed_data, np.eye(4))  # canonical affine
            self.imgs[name] = img
            self.add_img_name_to_list_model(name)
        self.set_current(name)

    def save_nifti(self, file_path, img_name):
        """儲存目前選擇的影像到指定路徑"""
        if img_name in self.imgs:
            img = self.imgs[img_name]
            nib.save(img, file_path)
            print(f"影像 {img_name} 已儲存到 {file_path}")
        else:
            print(f"影像 {img_name} 不存在，無法儲存。")

    def set_current(self, key):
        if key in self.imgs:
            self.current_key = key
            for obs in self.observers:
                if isinstance(obs, BasePanel) or obs.isVisible():
                    obs.update(key, self.imgs[key])

    def get_current(self, key=None):
        """取得當前影像的資料；如果沒有當前影像，則回傳 None"""
        return self.imgs.get(self.current_key, None)

    def get_img(self, img_name):
        """取得指定影像的資料；如果不存在，則回傳 None"""
        return self.imgs.get(img_name, None)

    def remove_img(self, img_name):
        """刪除第 idx 張影像；成功回傳 True，失敗回傳 False"""

        if img_name in self.imgs:
            img = self.imgs.pop(img_name)
            if hasattr(img, "close"):
                img.close()
            self.img_name_list_model.removeRow(self.find_text(img_name))
            return True

        return False

    def find_text(self, text: str) -> int:
        """傳回第一個匹配的 row，找不到回 -1。"""
        for r in range(self.img_name_list_model.rowCount()):
            if (
                self.img_name_list_model.data(
                    self.img_name_list_model.index(r), Qt.DisplayRole
                )
                == text
            ):
                return r
        return -1

    def add_img(self, image_name, image_data):
        """新增影像到 imgs 字典中"""
        if image_name not in self.imgs:
            self.imgs[image_name] = image_data
            self.add_img_name_to_list_model(image_name)
        elif image_data is not None:
            # 如果影像已存在，則更新其資料
            print(f"影像 {image_name} 已存在，更新其資料。")
            self.imgs[image_name] = image_data
        for observer in self.observers:
            if isinstance(observer, BasePanel):
                BasePanel.update(observer, image_data)

    def register(self, observer):
        self.observers.append(observer)
        if self.get_current() is not None:
            # 如果有當前影像，則立即更新觀察者
            observer.update(self.get_current())
