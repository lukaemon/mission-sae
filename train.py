"""dimension annotation
b: batch
t: token position
d: gpt d_model
v: gpt vocab size
l: SAE n latent
k: topk

Known delta to paper training spec:
- skip the whole param init section
- total training token is 1.31b, paper=6.4b
- no weight EMA
- no AuxK MSE loss, aka ghost grads


Don't understand
- We project away gradient information parallel to the decoder vectors, to account for interaction between Adam and decoder normalization.
- By convention, we average the gradient across the batch dimension. Therefore skip eps=6.25e-10. 
- For the main MSE loss, we compute an MSE normalization constant once at the beginning of training, and do not do any loss normalization per batch.
"""

import argparse
from pathlib import Path

import torch
import numpy as np

import transformer_lens.utils as utils
from sparse_autoencoder.model import Autoencoder, TopK
from sparse_autoencoder.loss import autoencoder_loss
from tqdm import tqdm

K = 32  # top k
seq_len = 64  # default value of all experiments per paper
d_model = 768  # gpt2 small

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_layer", type=int, default=8)
    parser.add_argument("--n_step", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    
    device = utils.get_device()

    trn_data_path = data_dir / f"act_nbd_layer_{args.target_layer}_n_{args.n_step}_bs_{args.batch_size}.bin"
    act_nbd = np.memmap(str(trn_data_path), dtype=np.float32, mode='r+', shape=(args.n_step, args.batch_size, d_model))
    print(f"trn data loaded with shape {act_nbd.shape}")

    n_latents = 2**15  # 32k
    n_inputs = 768  # gpt2 small d_model
    sae = Autoencoder(n_latents, n_inputs, activation=TopK(K), tied=True, normalize=True).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)

    print("start trining ...")
    with tqdm(range(args.n_step), unit="step") as pbar:
        for step in pbar:
            act_bd = act_nbd[step]
            act_bd = torch.from_numpy(act_bd).to(device)

            _, latent_bl, recon_bd = sae(act_bd)
            loss = autoencoder_loss(recon_bd, act_bd, latent_bl, l1_weight=0)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
