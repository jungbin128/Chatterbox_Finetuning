import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import IterableDataset

from dataset_loader_emilia import EmiliaKoreanDataset
from t3 import T3
from t3_config import T3Config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class KRIterableDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.base = EmiliaKoreanDataset()

    def __iter__(self):
        for text, semantic in self.base:
            yield text, semantic


def main():
        cfg = T3Config()
        model = T3(cfg).to(DEVICE)
        model.train()

        optimizer = Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        dataset = KRIterableDataset()

        max_epochs = 1
        max_steps = 10000
        log_every = 10
        step = 0

        for epoch in range(max_epochs):
            print(f"epoch {epoch + 1}", flush=True)
            for text, semantic in dataset:
                step += 1

                txt = torch.tensor(
                    [[cfg["bos"], 10, 20, 30, cfg["eos"]]],
                    dtype=torch.long,
                    device=DEVICE
                )

                semantic = semantic[: cfg["max_speech"]]
                target = semantic.unsqueeze(0).to(DEVICE)

                logits = model.forward_logits(txt, target)
                loss = criterion(
                    logits.reshape(-1, cfg["speech_vocab"]),
                    target.reshape(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % log_every == 0:
                    print(f"step {step} loss {loss.item():.4f}", flush=True)

                if step >= max_steps:
                    break
            if step >= max_steps:
                break

        os.makedirs("ckpt", exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": dict(cfg),
            },
            "ckpt/final.pt"
        )
        print("saved ckpt/final.pt", flush=True)


if __name__ == "__main__":
    main()
