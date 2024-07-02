from pathlib import Path

import blobfile as bf
import sparse_autoencoder
import torch

save_dir = Path("data") / "sae"
save_dir.mkdir(parents=True, exist_ok=True)


def load_sae(size_k, location, layer_index, device):
    assert location in ["resid_delta_attn", "resid_delta_mlp", "resid_post_attn", "resid_post_mlp"]
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
    model = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    model = model.to(device)
    print(f"Loaded pretrained SAE {model_path}")
    return model
