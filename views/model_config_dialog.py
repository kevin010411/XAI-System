# model_config_dialog.py
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
)
import torch
from mmengine import Config
from .segmentation_models.custom_module import build_model


class ModelConfigDialog(QDialog):
    def __init__(self, parent=None):
        """
        :param model_presets: List of (顯示名稱, dict: {class, weights_path, kwargs})
        """
        super().__init__(parent)
        self.setWindowTitle("模型設定")
        self.model_presets = self.get_model_config(
            "./views/modules/segmentation/configs"
        )
        self.config = None
        self.model_object = None

        main_layout = QVBoxLayout(self)
        self.combo = QComboBox()
        self.combo.addItem("不讀入模型")  # 第一個是 None
        for display_name, _ in self.model_presets:
            self.combo.addItem(display_name)
        self.info_label = QLabel("請選擇模型")
        main_layout.addWidget(self.combo)
        main_layout.addWidget(self.info_label)

        btn_box = QHBoxLayout()
        ok_btn = QPushButton("確定")
        cancel_btn = QPushButton("取消")
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        main_layout.addLayout(btn_box)

        self.combo.currentIndexChanged.connect(self.on_selection_changed)
        ok_btn.clicked.connect(self.on_accept)
        cancel_btn.clicked.connect(self.reject)

        self.on_selection_changed(0)  # 預設

    def on_selection_changed(self, idx):
        if idx == 0:
            self.info_label.setText("未選擇模型，執行時將不會載入模型")
            self.model_object = None
        else:
            display_name, _ = self.model_presets[idx - 1]
            self.info_label.setText(f"已選模型：{display_name}")
            self.model_object = None  # 只有在按確定時才會真的去 instantiate

    def on_accept(self):
        idx = self.combo.currentIndex()
        if idx == 0:
            self.model_object = None
            self.accept()
        else:
            # 準備 instantiate 並 load weights
            _, config_path = self.model_presets[idx - 1]  # model_name,config_path
            try:
                self.config = Config.fromfile(config_path["path"])
                model = build_model(self.config.model)
                state = torch.load(
                    self.config.pretrain_path,
                    map_location="cuda" if torch.cuda.is_available() else "cpu",
                )
                if "state_dict" in state:
                    state = state["state_dict"]
                    model.load_state_dict(state)
                else:
                    model.load(state)
                model.eval()
                self.model_object = model
                self.accept()
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.critical(self, "模型載入失敗", f"錯誤: {e}")
                self.model_object = None

    def get_model_and_config(self):
        """None 代表不載入模型，否則回傳 PyTorch model instance"""
        return self.model_object, self.config

    def get_model_config(self, root_path):
        from pathlib import Path

        if isinstance(root_path, str):
            root_path = Path(root_path)
        preset = []
        for path in root_path.glob("*.py"):
            preset.append((path.stem, {"path": path}))
        return preset
