"""Training runner (PROTOCOL_v3.md sections 10, 21).

AdamW (betas 0.9/0.95, eps 1e-8); weight decay 0.1 on constrained shadow
tensors ONLY; grad clip 1.0; warmup 2% then cosine to 10% of peak; effective
batch 0.5M tokens via gradient accumulation; bf16 autocast + fp32 master
weights, no loss scaling. Checkpoints and resumes cleanly (section 21).

Used for both the smoke LR stage (section 10a) and comparative runs. It never
opens a comparative validation curve — the caller decides which ledger/mode a
run belongs to, and the freeze gate governs comparative analysis.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..arms.base import ConstrainedLinear
from ..config import RunConfig
from ..data.pipeline import WindowSampler
from ..model.transformer import ModelArgs, Transformer
from ..paths import checkpoints_dir
from ..seeding import paired_generator, seed_everything


def _lr_at(step: int, total: int, peak: float, warmup_frac: float, min_frac: float) -> float:
    warm = max(1, int(warmup_frac * total))
    if step < warm:
        return peak * step / warm
    prog = (step - warm) / max(1, total - warm)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return peak * (min_frac + (1 - min_frac) * cos)


def _param_groups(model: Transformer, weight_decay: float):
    constrained_ids = set()
    for mod in model.constrained_modules().values():
        for p in mod.parameters():
            constrained_ids.add(id(p))
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if id(p) in constrained_ids else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},        # constrained shadow only
        {"params": no_decay, "weight_decay": 0.0},
    ]


@dataclass
class RunnerResult:
    run_id: str
    val_loss_final: float
    val_loss_selected: float
    tokens_completed: int
    tokens_per_second: float
    peak_gpu_memory_bytes: int
    checkpoint_path: str
    status: str


class Runner:
    def __init__(self, cfg: RunConfig, *, train_bin: Path, val_bin: Path,
                 device: str = "cuda", vocab_size: int = 50257,
                 ckpt_tag: str | None = None):
        assert cfg.d_model is not None, "cfg.d_model must be set from the frozen cell"
        self.cfg = cfg
        self.device = device
        self.train_bin, self.val_bin = Path(train_bin), Path(val_bin)
        seed_everything(cfg.seed)
        gen = paired_generator(cfg.seed)  # paired init draws for C/D
        args = ModelArgs(arm=cfg.arm, d_model=cfg.d_model, vocab_size=vocab_size,
                         depth=cfg.policy.depth_blocks, head_dim=cfg.policy.head_dim,
                         d_ff_mult=cfg.policy.d_ff_multiplier,
                         seq_len=cfg.train.sequence_length, twist=cfg.twist,
                         init_std=cfg.train.init_std)
        self.model = Transformer(args, gen).to(device)
        self.opt = torch.optim.AdamW(
            _param_groups(self.model, cfg.train.weight_decay),
            lr=cfg.peak_lr, betas=cfg.train.betas, eps=cfg.train.eps)
        self.seq = cfg.train.sequence_length
        self.tokens_per_step = self._token_step(cfg)
        self.total_steps = max(1, cfg.train_tokens // self.tokens_per_step)
        self.accum = max(1, self.tokens_per_step // (cfg.microbatch * self.seq))
        self.step = 0
        self.best_val = float("inf")
        self.ckpt = checkpoints_dir() / f"{ckpt_tag or cfg.run_id}.pt"

    def _token_step(self, cfg: RunConfig) -> int:
        accum = max(1, cfg.train.effective_batch_tokens // (cfg.microbatch * self.seq))
        return accum * cfg.microbatch * self.seq

    # -- checkpoint / resume ------------------------------------------------
    def save(self):
        tmp = self.ckpt.with_suffix(".tmp")
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "step": self.step, "best_val": self.best_val,
                    "sampler": self._sampler.state_dict(),
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, tmp)
        tmp.replace(self.ckpt)  # atomic

    def maybe_resume(self):
        if not self.ckpt.exists():
            return False
        sd = torch.load(self.ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        self.step = sd["step"]
        self.best_val = sd["best_val"]
        self._resume_sampler = sd["sampler"]
        torch.set_rng_state(sd["rng"].cpu())
        return True

    @torch.no_grad()
    def evaluate(self, max_tokens: int = 2_000_000) -> float:
        self.model.eval()
        val = WindowSampler(self.val_bin, self.seq, self.cfg.microbatch,
                            seed=99, device=self.device)
        n_batches = max(1, max_tokens // (self.cfg.microbatch * self.seq))
        tot, cnt = 0.0, 0
        for _ in range(n_batches):
            x, y = val.batch()
            with torch.autocast(self.device, dtype=torch.bfloat16):
                _, loss = self.model(x, y)
            tot += loss.item(); cnt += 1
        self.model.train()
        return tot / cnt

    def train(self, *, checkpoint_every_steps: int = 25,
              on_step=None) -> RunnerResult:
        self._sampler = WindowSampler(self.train_bin, self.seq, self.cfg.microbatch,
                                      seed=self.cfg.seed, device=self.device)
        if getattr(self, "_resume_sampler", None) is not None:
            self._sampler.load_state_dict(self._resume_sampler)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.model.train()
        t0 = time.time()
        while self.step < self.total_steps:
            lr = _lr_at(self.step, self.total_steps, self.cfg.peak_lr,
                        self.cfg.train.warmup_fraction, self.cfg.train.min_lr_fraction)
            for pg in self.opt.param_groups:
                pg["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            for _ in range(self.accum):
                x, y = self._sampler.batch()
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                (loss / self.accum).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.opt.step()
            self.step += 1
            if on_step:
                on_step(self.step, self.total_steps, loss.item(), lr)
            if self.step % checkpoint_every_steps == 0:
                self.save()
        val_final = self.evaluate()
        self.best_val = min(self.best_val, val_final)
        self.save()
        dt = time.time() - t0
        toks = self.step * self.tokens_per_step
        return RunnerResult(
            run_id=self.cfg.run_id, val_loss_final=val_final,
            val_loss_selected=self.best_val, tokens_completed=toks,
            tokens_per_second=toks / dt if dt > 0 else 0.0,
            peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
            checkpoint_path=str(self.ckpt), status="complete")
