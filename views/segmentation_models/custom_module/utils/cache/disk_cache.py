# disk_cache.py
import json
from pathlib import Path
from collections import defaultdict

import torch
from .patch_cache import PatchCache, Slice


class DiskCache(PatchCache):
    """把每個 tensor 立即寫入硬碟，適合大資料集."""

    def __init__(self, root: Path, compress: bool = True):
        self.root = Path(root)
        self.compress = compress
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, list[dict]] = defaultdict(list)  # 只存索引

    # ---------- 寫入 ----------
    def add(
        self,
        layer: str,
        b_idx: int,
        slc: Slice,
        tensor: torch.Tensor,
        shape: tuple[int, int],
    ):
        idx = len(self._meta[layer])
        fname = f"{layer}_{idx:07d}.pt"
        fpath = self.root / fname
        torch.save(
            tensor.detach().cpu().half(),
            fpath,
            _use_new_zipfile_serialization=self.compress,
        )
        self._meta[layer].append(
            {
                "b_idx": b_idx,
                "slice": [(s.start, s.stop) for s in slc],
                "file": fname,
                "shape": shape,
            }
        )

    # ---------- 讀取 ----------
    def iter(self, layer: str):
        import torch

        if layer not in self._meta:
            return iter([])
        for rec in self._meta[layer]:
            b_idx = rec["b_idx"]
            slc = tuple(slice(*pair) for pair in rec["slice"])
            tensor = torch.load(self.root / rec["file"], map_location="cpu")
            shape = rec["shape"]
            yield (b_idx, slc, tensor, shape)

    def layers(self):
        return list(self._meta.keys())

    # ---------- 管理 ----------
    def clear(self):
        # 刪檔案
        for p in self.root.glob("*"):
            p.unlink()
        self._meta.clear()

    # ---------- 序列化 ----------
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # 1) 把所有 tensor 檔案原封不動 copy
        for src in self.root.glob("*.pt"):
            dst = path / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
        # 2) meta
        (path / "meta.json").write_text(json.dumps(self._meta))

    @classmethod
    def load(cls, path):
        path = Path(path)
        obj = cls(path)  # 會先 mkdir，但如果已存在沒關係
        obj._meta = json.loads((path / "meta.json").read_text())
        return obj
