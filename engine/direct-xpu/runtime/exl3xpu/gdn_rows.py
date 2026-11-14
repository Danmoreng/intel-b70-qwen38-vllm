"""Reuse host-computed row indices without indexing device tensors by CPU masks."""
import torch


class HostRows:
    def __init__(self, mask):
        if mask.device.type != 'cpu' or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError('GDN row selector requires a one-dimensional CPU bool mask')
        self.mask = mask
        self.indices = None
        self.device_indices = {}
        self.prefix_length = None

    def _prepare(self):
        if self.indices is not None:
            return
        self.indices = self.mask.nonzero(as_tuple=True)[0]
        n = self.indices.numel()
        if torch.equal(self.indices, torch.arange(n, dtype=torch.long)):
            self.prefix_length = n

    def select(self, tensor):
        if tensor.shape[0] != self.mask.numel():
            raise ValueError('GDN mask and tensor row counts differ')
        self._prepare()
        if self.prefix_length is not None:
            return tensor[:self.prefix_length]
        if tensor.device.type == 'cpu':
            return tensor.index_select(0, self.indices)
        key = str(tensor.device)
        if key not in self.device_indices:
            pinned = self.indices.pin_memory()
            self.device_indices[key] = pinned.to(tensor.device, non_blocking=True)
        return tensor.index_select(0, self.device_indices[key])
