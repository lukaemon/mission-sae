import os

import tiktoken
from datasets import load_dataset, load_from_disk
import numpy as np
import torch

def load_owt(path="data/owt_tokenized"):
    ds = load_from_disk(path)
    print(f"Loaded {len(ds):,} sample texts from {path}")
    return ds


def sample(dataset, batch_size, rng: np.random.Generator, seq_len=64):
    batch_seq = rng.integers(0, len(dataset), (batch_size,))
    bin = []
    for r in dataset.select(batch_seq):
        assert (
            r["len"] > seq_len
        ), f"input_ids({r['len']}) is shorter than seq_len={seq_len}"

        ptr_max = r["len"] - seq_len
        ptr_start = rng.integers(0, ptr_max)
        segment = r["input_ids"][ptr_start : ptr_start + seq_len]
        segment = torch.tensor(segment, dtype=torch.long)
        bin.append(segment)

    return torch.stack(bin)  # (batch_size, seq_len)


if __name__ == "__main__":
    raw_ds = load_dataset("Skylion007/openwebtext")["train"]
    tk = tiktoken.get_encoding("gpt2")

    def tokenize(r):
        r["input_ids"] = tk.encode(r["text"])
        r["len"] = len(r["input_ids"])
        return r

    ds = raw_ds.map(tokenize, num_proc=32)
    os.makedirs("data", exist_ok=True)

    ds.save_to_disk("data/owt_tokenized")
    print("Dataset saved to data/owt_tokenized")
    print(f"It has {len(ds):,} text samples")
    print(f"Minimum token seq_len per text is {min(ds['len'])}")
