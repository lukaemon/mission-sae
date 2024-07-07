from pathlib import Path

import blobfile as bf
from sparse_autoencoder.model import Autoencoder, TopK

import torch

save_dir = Path("data") / "sae"
save_dir.mkdir(parents=True, exist_ok=True)


def load_sae(size_k, device, location='resid_post_mlp', layer_index=8):
    assert location in [
        "resid_delta_attn",
        "resid_delta_mlp",
        "resid_post_attn",
        "resid_post_mlp",
    ]
    assert layer_index in range(12)

    model_name = f"v5_{size_k}k_location_{location}_layer_{layer_index}.pt"
    az_path = f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v5_{size_k}k/autoencoders/{layer_index}.pt"
    model_path = save_dir / model_name

    if not model_path.exists():
        print(f"downloading {az_path}")
        with bf.BlobFile(az_path, mode="rb") as f:
            content = f.read()
            if len(content) == 0:
                raise ValueError(f"{az_path}: link contains 0 bytes")

            with model_path.open("wb") as wf:
                wf.write(content)
            print(f"SAE saved: {model_path}")

    state_dict = torch.load(model_path)

    model = Autoencoder.from_state_dict(state_dict)
    model = model.to(device)
    print(f"Loaded pretrained SAE {model_path}")
    
    return model


def load_homecook_sae(size_k, device):
    assert size_k in (32, 128), "only support 32k or 128k SAE"
    model_path = save_dir / f"sae_{size_k}k.pt"
    if not model_path.exists():
        raise ValueError(
            f"{model_path} doesn't exist. Use train.py to train one first."
        )

    sd = torch.load(model_path)
    sd.pop('activation_state_dict')
    sd.pop('activation.k')

    n_latents = 2**15 if size_k == 32 else 2**17
    sae = Autoencoder(n_latents, n_inputs=768, activation=TopK(32), normalize=True)
    sae.load_state_dict(sd)
    sae = sae.to(device)

    return sae
