import os
import torch
from datasets import load_dataset


class EmiliaKoreanDataset:
    def __init__(self):
        self.ds = load_dataset(
            "amphion/Emilia-Dataset",
            split="train",
            streaming=True,
            token=os.environ.get("HF_TOKEN"),
        ).select_columns(["json"])

    def __iter__(self):
        scanned = 0
        used = 0
        for ex in self.ds:
            scanned += 1
            print(f"[streaming] scanned {scanned}", flush=True)

            info = ex["json"]
            if info.get("language") != "KR":
                continue
            if "semantic" not in info:
                continue

            text = info.get("text", "").strip()
            semantic = info["semantic"]
            if not text or not semantic:
                continue

            semantic = torch.tensor(semantic, dtype=torch.long)
            used += 1
            print(f"[streaming] use KR semantic {used}", flush=True)
            yield text, semantic
