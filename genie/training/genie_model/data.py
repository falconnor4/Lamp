import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    def __init__(self, bin_path, seq_len, pad_id=0):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.n = max(1, (len(self.tokens) - 1) // seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = (i * self.seq_len) % max(1, len(self.tokens) - self.seq_len - 1)
        chunk = torch.from_numpy(self.tokens[start : start + self.seq_len].astype(np.int64))
        return chunk


class SyntheticDataset(Dataset):
    def __init__(self, n_samples, seq_len, vocab_size, pad_id=0):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return torch.randint(10, self.vocab_size, (self.seq_len,), generator=g)


def collate(batch):
    return torch.stack(batch)
