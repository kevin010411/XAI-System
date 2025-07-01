import torch, gc
from contextlib import contextmanager


@contextmanager
def mem_watch(tag):
    torch.cuda.reset_peak_memory_stats()
    yield
    print(tag, "peak MB =", torch.cuda.max_memory_allocated() / 1e6)
