from PySide6.QtWidgets import QWidget


_PANEL_REGISTRY: dict[str, type[QWidget]] = {}


def register_panel(name: str):
    """把面板類別註冊到全域字典"""

    def decorator(cls: type[QWidget]):
        _PANEL_REGISTRY[name] = cls
        return cls

    return decorator


def create_xai_panel(cfg) -> QWidget:
    """
    cfg 需至少有 `type` 屬性 (或 key)，
    其餘參數可再視需要傳入 Panel 建構子。
    """
    panel_cls = _PANEL_REGISTRY.get(cfg.type)
    if panel_cls is None:
        print(f"Unsupported XAI panel type: {cfg.type}")
        return
    return panel_cls(cfg)  # 若建構子不需要 cfg，改成 panel_cls()
