import os
import torch
from pathlib import Path

def generate_dummy_data(output_dir="./dummy_dataset", num_samples=5):
    """Generates dummy preprocessed .pt files for training tests."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 4s audio at 50Hz (latent frames) = 200 frames
    # 1s Prefix + 3s Prediction = 50 + 150
    T = 200 
    L = 20  # Condition tokens
    D_latent = 64
    D_cond = 64
    D_ctx = 128 # 64 (src) + 64 (mask)
    
    for i in range(num_samples):
        sample = {
            "target_latents": torch.randn(T, D_latent),
            "attention_mask": torch.ones(T),
            "encoder_hidden_states": torch.randn(L, D_cond),
            "encoder_attention_mask": torch.ones(L),
            "context_latents": torch.randn(T, D_ctx),
            "metadata": {"id": f"dummy_{i}"}
        }
        torch.save(sample, os.path.join(output_dir, f"sample_{i}.pt"))
        
    print(f"Generated {num_samples} dummy samples in {output_dir}")

if __name__ == "__main__":
    generate_dummy_data()
