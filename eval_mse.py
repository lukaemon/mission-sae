import time
import argparse
from functools import partial

import numpy as np
import torch

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
from sparse_autoencoder.loss import normalized_mean_squared_error
from tqdm import tqdm

from openwebtext import load_owt, sample
from pretrained_sae import load_sae, load_homecook_sae

torch.set_grad_enabled(False)

seed = 42
layer_index = 8
n_batch = 256
batch_size = 16
seq_len = 64


def compute_mse(act_btd, hook, sae, bin):
    latent, info = sae.encode(act_btd)
    recon = sae.decode(latent, info)
    mse = normalized_mean_squared_error(recon, act_btd)

    bin.append(mse.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size_k", type=int, default=32)
    parser.add_argument("--oai", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    if args.size_k not in [32, 128]:
        raise ValueError("size_k must be either 32 or 128")

    device = utils.get_device()
    rng = np.random.default_rng(seed)
    ds = load_owt()
    
    gpt2 = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
    
    if args.oai == 0:
        sae = load_homecook_sae(args.size_k, device)
    else:
        sae = load_sae(args.size_k, device)

    bin = []
    hook_name = utils.get_act_name("resid_post", layer_index)
    hook_fn = partial(compute_mse, sae=sae, bin=bin)

    print(f"start processing MSE for {n_batch * batch_size * seq_len:,} tokens")
    with tqdm(range(n_batch), unit="batch", postfix={"tps": 0}) as pbar:
        for _ in pbar:
            start = time.perf_counter()

            batch = sample(ds, batch_size=batch_size, rng=rng)
            gpt2.run_with_hooks(
                batch, return_type=None, fwd_hooks=[(hook_name, hook_fn)]
            )

            delta = time.perf_counter() - start
            tok_per_batch = batch_size * seq_len
            tps = tok_per_batch / delta

            pbar.set_postfix({"tps": f"{tps:,.2f}"})

    avg_mse = np.mean(bin)
    print(f"SAE {args.size_k}k MSE={avg_mse:.4f}")
