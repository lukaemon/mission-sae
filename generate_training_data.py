"""dimension annotation
n: step
b: batch
t: token position
d: gpt d_model
v: gpt vocab size
l: SAE n latent
k: topk
"""

import argparse
from pathlib import Path
from functools import partial

import torch
import numpy as np

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
from tqdm import tqdm

from openwebtext import load_owt, sample

torch.set_grad_enabled(False)
seq_len = 64  # default value of all experiments per paper
d_model = 768  # gpt2 small

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)


def hook_fn_save_act(act_btd, hook, step, mmap_act_nbd):
    act_bd = act_btd[:, -1].detach().cpu().numpy()
    mmap_act_nbd[step] = act_bd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_layer", type=int, default=8)
    parser.add_argument("--n_step", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    ds = load_owt()
    gpt2 = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)

    output_path = data_dir / f"act_nbd_layer_{args.target_layer}_n_{args.n_step}_bs_{args.batch_size}.bin"
    mmap_act_nbd = np.memmap(
        str(output_path),
        dtype=np.float32,
        mode='w+',
        shape=(args.n_step, args.batch_size, d_model)
    )
    
    tpb = args.batch_size * seq_len
    total = args.n_step * tpb

    print(f"start data gereration: {args.n_step=:,}, {tpb=:,}, {total=:,}")

    with tqdm(range(args.n_step), desc="generating data", unit="step") as pbar:
        for i in pbar:
            batch_bt = sample(ds, args.batch_size)
            hook_fn = partial(hook_fn_save_act, step=i, mmap_act_nbd=mmap_act_nbd)
            gpt2.run_with_hooks(
                batch_bt,
                return_type=None,
                fwd_hooks=[(utils.get_act_name("resid_post", layer=args.target_layer), hook_fn)],
            )
    
    mmap_act_nbd.flush()
    print(f"data saved to {output_path}")