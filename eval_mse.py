import time
import argparse
from functools import partial
from pathlib import Path

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
seq_len = 64
d_model = 768

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)


def compute_mse(act_btd, hook, sae, mean_act, bin):
    latent, info = sae.encode(act_btd)
    recon = sae.decode(latent, info)
    
    baseline_mse = (act_btd - mean_act).pow(2).mean()
    actual_mse = (recon - act_btd).pow(2).mean()
    normalized_mse = actual_mse / baseline_mse

    bin.append(normalized_mse.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size_k", type=int, default=32)
    parser.add_argument("--target_layer", type=int, default=8)
    parser.add_argument("--n_step", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=2048)
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

    trn_data_path = data_dir / f"act_nbd_layer_{args.target_layer}_n_{args.n_step}_bs_{args.batch_size}.bin"
    act_nbd = np.memmap(
        str(trn_data_path),
        dtype=np.float32,
        mode="r+",
        shape=(args.n_step, args.batch_size, d_model),
    )

    sample_act = act_nbd[-100:].reshape(-1, d_model)  # (100*2048, 768) last 100 step as sample act
    mean_act = sample_act.mean(0)
    mean_act = torch.from_numpy(mean_act).to(torch.float32).to(device)

    bin = []
    hook_name = utils.get_act_name("resid_post", layer_index)
    hook_fn = partial(compute_mse, sae=sae, mean_act=mean_act, bin=bin)

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
