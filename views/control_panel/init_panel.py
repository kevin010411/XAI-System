from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QLabel,
    QFileDialog,
    QPushButton,
    QComboBox,
    QMessageBox,
)
from PySide6.QtCore import Qt
from .base_panel import BasePanel


class InitPanel(BasePanel):

    def __init__(self, data_manager, parent=None):
        super().__init__(data_manager, parent)

        center_text = QLabel("初始化頁面，歡迎使用此程式")
        self.layout.addWidget(center_text)
        # 添加載入 NIfTI 檔案按鈕
        load_data_button = QPushButton("載入 NIfTI 檔案")
        load_data_button.clicked.connect(self.load_nifti)
        self.layout.addWidget(load_data_button)
        # 添加刪除目前影像按鈕
        btn_delete = QPushButton("刪除目前影像")
        btn_delete.clicked.connect(self.delete_current_img)
        self.layout.addWidget(btn_delete)

        # 添加影像選擇下拉選單
        self.img_selector = QComboBox()
        self.img_selector.currentIndexChanged.connect(self.on_img_selected)
        self.layout.addWidget(self.img_selector)

        self.refresh_img_selector()

    def load_nifti(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇 NIfTI 檔案", "", "NIfTI Files (*.nii.gz *.nii)"
        )
        if file_path:
            try:
                self.data_manager.load_nifti(file_path)
                self.refresh_img_selector(select_last=True)
            except Exception as e:
                print("讀取失敗:", e)

    def delete_current_img(self):
        """刪除目前選擇的影像。"""
        reply = QMessageBox.question(
            self,
            "確認刪除",
            f"確定要刪除「{self.img_selector.currentText()}」？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.data_manager.remove_img(self.data_manager.current_key)
        self.refresh_img_selector(select_last=True)

    # ---------- 下拉選單 ----------
    def refresh_img_selector(self, select_last: bool = False):
        """重新把 data_manager.imgs 填進下拉清單。"""
        self.img_selector.blockSignals(True)  # 暫停 signal，避免多次觸發
        self.img_selector.clear()

        for idx, (img_name, img_data) in enumerate(self.data_manager.imgs.items()):
            # 預設用檔名 (若 DataManager 有別名屬性可自行替換)
            self.img_selector.addItem(img_name, userData=idx)  # userData 存 index

        if select_last and self.data_manager.imgs:
            self.img_selector.setCurrentIndex(len(self.data_manager.imgs) - 1)

        self.img_selector.blockSignals(False)

    def on_img_selected(self, index: int):
        """使用者在下拉選了新影像 → 通知 DataManager。"""
        if index < 0:
            return
        try:
            self.data_manager.set_current(self.img_selector.itemText(index))
        except AttributeError:
            # 如果是用物件而非 index，請改成 self.data_manager.set_current(self.data_manager.imgs[index])
            pass

    def update(self, img):
        """更新 InitPanel 的內容，這裡可以顯示一些影像資訊。"""
        return
