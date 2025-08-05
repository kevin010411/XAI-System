# ram_cache.py
from collections import defaultdict
from pathlib import Path
import torch
from .patch_cache import PatchCache, Item, Slice


class RAMCache(PatchCache):
    """所有資料放在 RAM 中。"""

    def __init__(self) -> None:
        # {layer: [ (b_idx, slice, tensor), ... ]}
        self._store: defaultdict[str, list[Item]] = defaultdict(list)

    # ---------- 寫入 ----------
    def add(
        self,
        layer: str,
        slc: Slice,
        tensor: torch.Tensor,
        shape: tuple[int, int],
    ) -> None:
        # 轉 CPU + half() 節省記憶體（可按需調整）
        self._store[layer].append((slc, tensor, shape))

    # ---------- 讀取 ----------
    def iter(self, layer: str):
        return iter(self._store.get(layer, []))

    def layers(self):
        return list(self._store.keys())

    # ---------- 管理 ----------
    def clear(self):
        self._store.clear()

    # ---------- 序列化 ----------
    def save(self, path):
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(dict(self._store), path / "ramcache.pt")

    @classmethod
    def load(cls, path):
        import torch

        path = Path(path)
        obj = cls()
        obj._store = torch.load(path / "ramcache.pt")
        return obj
