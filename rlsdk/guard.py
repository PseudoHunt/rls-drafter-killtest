"""Leakage guard (brief §4.5): target-side arrays may only be read below the current frontier."""
import torch


class LeakageError(RuntimeError):
    pass


class Frontier:
    def __init__(self, t):
        self.t = int(t)

    def advance(self, new_t):
        if new_t < self.t:
            raise ValueError("frontier can only move forward")
        self.t = int(new_t)


class Guarded:
    """Wraps a [T, ...] tensor; any read touching an index >= frontier.t raises LeakageError."""

    def __init__(self, tensor, frontier, name="array"):
        self._x = tensor
        self._f = frontier
        self.name = name

    def __len__(self):
        return self._x.shape[0]

    def _check_max(self, hi):
        if hi >= self._f.t:
            raise LeakageError(f"{self.name}: read index {hi} >= frontier {self._f.t}")

    def __getitem__(self, idx):
        rest = ()
        if isinstance(idx, tuple):
            idx, rest = idx[0], idx[1:]
        n = self._x.shape[0]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(n)
            if step != 1:
                raise TypeError("only unit-step slices are supported")
            if stop > start:
                self._check_max(stop - 1)
        elif isinstance(idx, int):
            self._check_max(idx % n)
        else:
            t = torch.as_tensor(idx)
            if t.numel():
                self._check_max(int((t % n).max()))
        return self._x[(idx, *rest)]
