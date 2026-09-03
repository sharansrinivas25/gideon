"""Training loop.

Deliberately plain: AdamW, cosine schedule with linear warmup, gradient
clipping, periodic evaluation on a held-out split. No tricks that would obscure
what the model is learning.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .config import GPTConfig, TrainConfig
from .data import CharDataset, download_tinyshakespeare
from .model import GPT
from .tokenizer import CharTokenizer


def lr_at(it: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``learning_rate * min_lr_ratio``.

    Warmup exists because Adam's second-moment estimate is garbage for the
    first few steps; taking full-size steps on a garbage estimate is how you
    get a loss spike you never recover from.
    """
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.learning_rate * cfg.min_lr_ratio
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    return min_lr + coeff * (cfg.learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, dataset: CharDataset, cfg: TrainConfig, device: str) -> dict:
    """Average loss over several batches - one batch is far too noisy to act on."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size, device)
            _, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    model_cfg: GPTConfig,
    train_cfg: TrainConfig,
    dataset: CharDataset,
    device: str,
    out_path: Path,
) -> tuple[GPT, list]:
    torch.manual_seed(train_cfg.seed)

    model = GPT(model_cfg).to(device)
    print(f"model: {model.num_params():,} non-embedding parameters")
    print(f"data:  {dataset}")

    optimizer = model.configure_optimizer(
        train_cfg.weight_decay, train_cfg.learning_rate, (train_cfg.beta1, train_cfg.beta2)
    )

    history: list[dict] = []
    best_val = float("inf")
    t_start = time.time()
    model.train()

    for it in range(train_cfg.max_iters + 1):
        lr = lr_at(it, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters:
            losses = estimate_loss(model, dataset, train_cfg, device)
            elapsed = time.time() - t_start
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr:.2e} | {elapsed/60:.1f} min"
            )
            history.append({"iter": it, **losses, "lr": lr, "elapsed_s": elapsed})
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(
                    out_path, model, dataset.tokenizer, it, best_val, history
                )

        if it == train_cfg.max_iters:
            break

        # gradient accumulation lets an effective batch larger than memory
        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_cfg.grad_accum_steps):
            x, y = dataset.get_batch("train", train_cfg.batch_size, device)
            _, loss = model(x, targets=y)
            (loss / train_cfg.grad_accum_steps).backward()
        # Clip before stepping: one bad batch can otherwise produce a gradient
        # large enough to knock the model out of the basin it was converging to.
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

    print(f"done in {(time.time() - t_start)/60:.1f} min | best val loss {best_val:.4f}")
    return model, history


def main() -> None:
    ap = argparse.ArgumentParser(description="Train gideon on TinyShakespeare")
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-iters", type=int, default=3000)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--out", default="results/ckpt.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    text = download_tinyshakespeare(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        learning_rate=args.lr,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )

    dataset = CharDataset(text, tokenizer, args.block_size)
    out_path = Path(args.out)
    _, history = train(model_cfg, train_cfg, dataset, args.device, out_path)

    hist_path = out_path.parent / "training_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"history -> {hist_path}")


if __name__ == "__main__":
    main()
