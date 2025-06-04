from PySide6.QtWidgets import QDockWidget
import os
import nibabel as nib
import numpy as np


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
                if obs.isVisible():
                    obs.update(self.imgs[key])

    def get_current(self):
        return self.imgs.get(self.current_key, None)

    def register(self, observer):
        self.observers.append(observer)
