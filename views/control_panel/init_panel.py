from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QFileDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)
from PySide6.QtCore import Qt
from .base_panel import BasePanel
from views.utils import CollapsibleBox


class InitPanel(BasePanel):

    def __init__(self, data_manager, **kwargs):
        super().__init__(data_manager, **kwargs)

        center_text = QLabel("初始化頁面，歡迎使用此程式")

        self.header_section = CollapsibleBox("Header (影像標頭)")
        self.extra_section = CollapsibleBox("Extensions / Extra")
        self.layout.addWidget(self.header_section)
        self.layout.addWidget(self.extra_section)

        # Initial populate (no image yet)
        self._update_meta_view(None)
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
        self.img_selector.currentIndexChanged.connect(self.on_img_selected)

    def load_nifti(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇 NIfTI 檔案", "", "NIfTI Files (*.nii.gz *.nii)"
        )
        if file_path:
            try:
                self.data_manager.load_nifti(file_path)
            except Exception as e:
                print("init_panel - 讀取失敗:", e)

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

    def on_img_selected(self, index: int):
        """使用者在下拉選了新影像 → 通知 DataManager。"""
        if index < 0:
            return
        self._update_meta_view(
            self.data_manager.get_img(self.img_selector.itemText(index))
        )

    def _build_header_table(self, header) -> QTableWidget:
        keys = header.keys()
        table = QTableWidget(len(keys), 2)
        table.setHorizontalHeaderLabels(["Field", "Value"])
        for row, key in enumerate(keys):
            table.setItem(row, 0, QTableWidgetItem(key))
            val = header[key]
            pretty = (
                val.tolist() if hasattr(val, "tolist") else val
            )  # numpy scalar → python
            table.setItem(row, 1, QTableWidgetItem(str(pretty)))
        table.resizeColumnsToContents()
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        return table

    def _build_extra_table(self, exts) -> QTableWidget:
        table = QTableWidget(len(exts), 3)
        table.setHorizontalHeaderLabels(["#", "Code", "Content (preview)"])
        for idx, ext in enumerate(exts):
            table.setItem(idx, 0, QTableWidgetItem(str(idx)))
            table.setItem(idx, 1, QTableWidgetItem(str(ext.get_code())))
            content = ext.get_content()
            if isinstance(content, bytes):
                preview = content.decode(errors="replace")[:80]
            else:
                preview = str(content)[:80]
            table.setItem(idx, 2, QTableWidgetItem(preview))
        table.resizeColumnsToContents()
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        return table

    def _update_meta_view(self, img):
        """Rebuild header / extra tables or show placeholder when *img* is None."""
        self.header_section.clear()
        self.extra_section.clear()
        if img is None:
            placeholder = QLabel("(尚未選擇影像)")
            placeholder.setAlignment(Qt.AlignCenter)
            self.header_section.add_widget(placeholder)
            self.extra_section.add_widget(QWidget())  # empty
            return

        self.header_section.add_widget(self._build_header_table(img.header))
        self.extra_section.add_widget(self._build_extra_table(img.header.extensions))
