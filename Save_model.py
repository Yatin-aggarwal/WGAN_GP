import torch
from pathlib import Path

def save_model(state):
    print("Saving model...")
    root = Path(".")
    torch.save(state, root/"Checkpoints"/"model.pt")
