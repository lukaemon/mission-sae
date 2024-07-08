"""dimension annotation
b: batch
t: token position
d: d_model
v: model token vocab size
l: SAE n latent
k: topk
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
ablate_token_idx = 0
V = 50257
T = 16

# IndexError: index 31 is out of bounds for dimension 1 with size 31
# was using 32, some sample doesn't have 32 features, so do ablation on top 16 strongest activated feature
K = 30
D = 768

n_sample = 8


def fn_ablate_feature(
    act_btd, hook, ablate_idx, sae, ablate_token_idx=ablate_token_idx
):
    lact_btk, _ = sae.encode(act_btd)
    lact_k = lact_btk[0, ablate_token_idx]

    # Sort lact_k by absolute value, descending
    sorted_indices = torch.argsort(lact_k.abs(), descending=True)
    lact_k = lact_k[sorted_indices]

    ldir_dk = sae.decoder.weight[:, sorted_indices]
    all_feature_dk = ldir_dk * lact_k

    active_feature = all_feature_dk[:, all_feature_dk.sum(dim=0) != 0]  # (d, 32)s
    ablate_feature = active_feature[:, ablate_idx]  # (d, )

    act = act_btd.clone()

    # subtract ablate_feature from target token AND all previous tokens's activation
    # act[0, : ablate_token_idx + 1] -= ablate_feature

    # subtract ablate_feature only from target token activation
    act[0, ablate_token_idx] -= ablate_feature

    return act


def fn_ablate_resid_stream_channel(
    act_btd, hook, ablate_idx, sae, ablate_token_idx=ablate_token_idx
):
    act = act_btd.clone()
    act[:, ablate_token_idx, ablate_idx] = 0
    return act


def proc_ablate(logit_btv, sample_1t, ablate_fn, r, sae, gpt2):
    bin = []
    for i in range(r):
        ablated_logit_btv = gpt2.run_with_hooks(
            sample_1t,
            return_type="logits",
            fwd_hooks=[
                (
                    utils.get_act_name("resid_post", layer_index),
                    partial(
                        ablate_fn,
                        sae=sae,
                        ablate_idx=i,
                    ),
                )
            ],
        )

        logit_diff_tv = (
            logit_btv[0, ablate_token_idx : ablate_token_idx + T]
            - ablated_logit_btv[0, ablate_token_idx : ablate_token_idx + T]
        )

        median_diff_t = torch.median(logit_diff_tv, dim=1)[0]

        logit_diff_tv -= median_diff_t[..., None]
        bin.append(logit_diff_tv)

    vt = torch.stack(bin).view(-1, V * T)
    l1 = torch.abs(vt).sum(-1)
    l2 = (vt**2).sum(-1) ** 0.5

    bench = (l1 / l2) ** 2
    normalized_bench = bench / (V * T)

    return normalized_bench.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size_k", type=int, default=32)
    parser.add_argument("--oai", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    rng = np.random.default_rng(seed)
    device = utils.get_device()

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

    sparsity_feature, sparsity_channel = [], []

    for _ in tqdm(range(n_sample), unit="sample"):
        sample_1t = sample(ds, 1, rng=rng)
        logit_btv = gpt2(sample_1t)

        sf = proc_ablate(logit_btv, sample_1t, fn_ablate_feature, K, sae, gpt2)
        sparsity_feature.append(sf)

        sc = proc_ablate(
            logit_btv, sample_1t, fn_ablate_resid_stream_channel, D, sae, gpt2
        )
        sparsity_channel.append(sc)

    print(f"downstream sparsity of SAE feature: {np.mean(sparsity_feature) * 100:.2f}%")
    print(f"downstream sparsity of resid stream channel: {np.mean(sparsity_channel) * 100:.2f}%")
