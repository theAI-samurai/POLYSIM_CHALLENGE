import torch
import glob
import os

files = sorted(glob.glob('checkpoints/exp5/*.pt'))
if not files:
    print("No .pt files found in checkpoints/exp5/")

for f in files:
    size = os.path.getsize(f)
    print(f"{f} ({size} bytes): ", end="", flush=True)
    try:
        torch.load(f, map_location='cpu', weights_only=False)
        print("OK")
    except Exception as e:
        print(f"FAIL ({type(e).__name__}: {e})")
