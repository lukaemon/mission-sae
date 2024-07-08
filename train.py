"""dimension annotation
b: batch
t: token position
d: gpt d_model
v: gpt vocab size
l: SAE n latent
k: topk
n: training step

Difference to paper training spec:
- total training token is 8 epoch of 1.31b, paper is 8 epoch of 6.4b
- We project away gradient information parallel to the decoder vectors, to account for interaction between Adam and decoder normalization.
- weight EMA
- ghost grads
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from geom_median.numpy import compute_geometric_median

import transformer_lens.utils as utils
from sparse_autoencoder.model import Autoencoder, TopK
from sparse_autoencoder.loss import autoencoder_loss
from tqdm import tqdm
import wandb
wandb.require("core")

K = 32  # top k
seq_len = 64  # default value of all experiments per paper
d_model = 768  # gpt2 small
n_latents = 2**17
n_inputs = 768  # gpt2 small d_model

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Use Tensor Cores even for fp32 matmuls, 128k training, 30min/epoch -> 20min
    torch.set_float32_matmul_precision("high")
    wandb.init(project="topk_sae", name="sae 128k (train improv)")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_layer", type=int, default=8)
    parser.add_argument("--n_step", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_epoch", type=int, default=8)
    args = parser.parse_args()

    device = utils.get_device()
    trn_data_path = (
        data_dir
        / f"act_nbd_layer_{args.target_layer}_n_{args.n_step}_bs_{args.batch_size}.bin"
    )
    act_nbd = np.memmap(
        str(trn_data_path),
        dtype=np.float32,
        mode="r+",
        shape=(args.n_step, args.batch_size, d_model),
    )
    print(f"trn data loaded with shape {act_nbd.shape}")


    # initializaiton
    sae = Autoencoder(n_latents, n_inputs, activation=TopK(K), normalize=True)
    
    sample_act = np.array(act_nbd[-100:]).reshape(-1, n_inputs)  # (100*2048, 768) last 100 step as sample act
    mse_scale = 1 / ((sample_act - sample_act.mean(0))**2).mean()
    mse_scale = torch.tensor(mse_scale, dtype=torch.float32, device=device)

    sae.encoder.weight.data = sae.decoder.weight.data.T.clone()  # tied init encoder to the transpose of the decoder
    sae.decoder.weight.data /= sae.decoder.weight.data.norm(dim=0)  # init decoder column to be unit-norm

    geometric_median_d = compute_geometric_median(sample_act).median
    geometric_median_d = torch.tensor(geometric_median_d, dtype=torch.float32, device=device)
    sae.pre_bias.data = geometric_median_d  # initialize the bias bpre to be the geometric median of a sample set of data points
    
    sae = sae.to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=4e-4)

    for epoch in range(args.n_epoch):
        print(f"... on epoch {epoch+1}/{args.n_epoch}")

        with tqdm(range(args.n_step), unit="step") as pbar:
            for step in pbar:
                act_bd = act_nbd[step]
                act_bd = torch.from_numpy(act_bd).to(device)

                _, _, recon_bd = sae(act_bd)
                loss = ((recon_bd - act_bd) ** 2).mean() * mse_scale

                loss.backward()

                sae.decoder.weight.data /= sae.decoder.weight.data.norm(dim=0)  # renormalize decoder column to be unit-norm

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix({"loss": f"{loss.item():.3f}"})
                wandb.log(dict(loss=loss))
    
    model_filename = f"sae_128k.pt"
    model_path = data_dir / 'sae' / model_filename
    torch.save(sae.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    wandb.finish()
