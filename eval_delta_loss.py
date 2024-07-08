"""dimension annotation
b: batch
t: token position
d: d_model
"""

import argparse
from functools import partial

import numpy as np
import torch

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
from tqdm import tqdm

from openwebtext import load_owt, sample
from pretrained_sae import load_sae, load_homecook_sae

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

seed = 42
layer_index = 8
n_batch = 256
batch_size = 16


def hook_fn_reconstruct_act(act_btd, hook, sae):
    latent, info = sae.encode(act_btd)
    recon_act_btd = sae.decode(latent, info)

    return recon_act_btd


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

    loss = []
    loss_sae = []

    for _ in tqdm(range(n_batch), unit="batch"):
        batch_bt = sample(ds, batch_size, rng=rng)

        l = gpt2(batch_bt, return_type="loss")
        ll = gpt2.run_with_hooks(
            batch_bt,
            return_type="loss",
            fwd_hooks=[
                (
                    utils.get_act_name("resid_post", layer_index),
                    partial(hook_fn_reconstruct_act, sae=sae),
                )
            ],
        )

        loss.append(l.item())
        loss_sae.append(ll.item())

    print(f"orignial loss = {np.mean(loss): .3f}")
    print(f"sae loss = {np.mean(loss_sae): .3f}")
    print(f"delta = {np.mean(loss_sae) - np.mean(loss):.5f}")
