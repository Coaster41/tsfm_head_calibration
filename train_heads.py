#!/usr/bin/env python3
"""
Train prediction heads on cached latent embeddings from time series foundation models,
with optional validation on tsmixup data.

- Supports head types: "mse", "gaussian", "poisson", "neg_binom", "studentst", "quantiles", "mixture"
- Trains multiple head types sequentially for the same backbone/cache
- Optional validation on tsmixup: runs backbone to get latents/stats for contexts and evaluates head

Example (train + validate):
    python train_heads.py \
        --model-name yinglong \
        --heads quantiles gaussian \
        --batch-size 512 \
        --tot-iters 2097152 \
        --epochs 3 \
        --save-dir models \
        --validate \
        --val-samples 256 \
        --tsmixup-cache-dir "/extra/datalab_scratch0/ctadler/time_series_models/mechanistic_interpretability/data/tsmixup_cache/" \
        --tsmixup-processed-cache "/extra/datalab_scratch0/ctadler/time_series_models/mechanistic_interpretability/data/tsmixup_cache/tsmixup_processed_300000_512_128.pkl"

Dependencies:
    - PyTorch, einops, tqdm, numpy
    - pytorch_forecasting (QuantileLoss)
    - uni2ts.distribution (MixtureOutput, components)
    - utils.components.ResidualBlock
    - utils.data_loader.create_cached_tsmixup_datasets
    - load_cached_features.FullLatentShardDataset

Backbone-specific validation requirements (only when --validate is used):
    - timesfm (google/timesfm-2.0-500m-pytorch) if --model-name timesfm
    - uni2ts moirai2 if --model-name moirai2
    - chronos-bolt if --model-name chronos_bolt
    - tirex if --model-name tirex
    - transformers YingLong_110m if --model-name yinglong
"""

import os
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tqdm import tqdm
import einops

# Uni2TS distributions
from uni2ts.distribution import (
    MixtureOutput,
    NormalOutput,
    StudentTOutput,
    LaplaceOutput,
    NormalFixedScaleOutput,
    NegativeBinomialOutput,
    LogNormalOutput,
)
from uni2ts.distribution.negative_binomial import NegativeBinomial

# Quantile loss
from pytorch_forecasting.metrics.quantile import QuantileLoss

# Project-specific components and cached latent dataset loader
from utils.components import ResidualBlock
from load_cached_features import FullLatentShardDataset
from utils.data_loader import create_cached_tsmixup_datasets

eps = 1e-6


# ---------------------------- Utilities and Heads ---------------------------- #

def get_mask(latents: Tensor, labels: Tensor, max_val: float, min_val: float) -> Tensor:
    """Create mask removing samples with any NaN latents or labels outside [min_val, max_val]."""
    nan_mask = ~torch.isfinite(latents)  # mark invalid latent rows
    clamp_mask = (labels > max_val) | (labels < min_val)  # mark out-of-range labels
    mask = ~(torch.any(nan_mask, dim=1) | torch.any(clamp_mask, dim=1))  # keep only good rows
    return mask

def get_preds(model: nn.Module, latents: Tensor, horizon_len: int = 128, output_dims: int = 1,
              reshape: bool = True) -> Tensor:
    """
    Forward a head model and shape to [B, H, D].
    - Accepts latents [B, D] or [B, P, D].
    - If model returns [B, P, P_len*D], we rearrange to [B, P*P_len, D].
    """
    out = model(latents)
    if reshape:
        if out.ndim == 3:
            out = einops.rearrange(
                out,
                "batch patches (patch_len output_dims) -> batch (patches patch_len) output_dims",
                output_dims=output_dims
            )
        else:
            out = einops.rearrange(
                out,
                "batch (horizon_len output_dims) -> batch horizon_len output_dims",
                horizon_len=horizon_len, output_dims=output_dims
            )
    return out

def mixture_quantiles_by_sampling(dist, qs: List[float], num_samples: int = 4096) -> Tensor:
    with torch.no_grad():
        samples = dist.sample((num_samples,))  # [S, B, H]
        q_tensor = torch.tensor(qs, device=samples.device, dtype=samples.dtype)
        q_vals = torch.quantile(samples, q_tensor, dim=0)  # [Q, B, H]
    return q_vals  # [Q, B, H]


class MixtureHead(nn.Module):
    """
    Wrap a backbone (e.g., ResidualBlock) and a uni2ts DistrParamProj derived from MixtureOutput.
    Forward returns the parameter PyTree ready to instantiate a Mixture distribution.
    """
    def __init__(self, in_features: int, hidden_dims: int, horizon_len: int, mixture_output: MixtureOutput):
        super().__init__()
        self.ln = torch.nn.LayerNorm(in_features)
        self.backbone = ResidualBlock(
            input_dims=in_features,
            output_dims=in_features,       # keep embedding size unchanged
            hidden_dims=hidden_dims,
        )
        self.param_proj = mixture_output.get_param_proj(
            in_features=in_features,
            out_features=tuple(horizon_len for _ in range(len(mixture_output.components))),
        )
        self.mixture_output = mixture_output

    def forward(self, latents: torch.Tensor, out_feat_size: torch.Tensor):
        """
        latents: [B, D] or [B, D'] depending on latent shape.
        returns: PyTree of mixture parameters.
        """
        latents = self.ln(latents)
        h = self.backbone(latents)
        params = self.param_proj(h, out_feat_size)
        # params['components'][2]['total_count'] += eps
        return params


# ---------------------------- Loss/Forecast Core ---------------------------- #

def latent_forecast(
    pred_head: nn.Module,
    latents: Tensor,
    horizon_len: int,
    head_type: str,
    output_dims: Optional[int] = None,
    labels: Optional[Tensor] = None,
    mu0: Optional[Tensor] = None,
    sigma0: Optional[Tensor] = None,
    quantiles: Optional[List[float]] = None,
    forecast: bool = True,
    mixture_output: Optional[MixtureOutput] = None,
) -> Dict[str, Tensor]:
    """
    Compute loss and optionally forecast statistics for a given head_type.
    Matches the notebook behavior.
    """
    if quantiles is None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}

    if output_dims is None:
        output_dims_dict = {
            "mse": 1, "gaussian": 2, "poisson": 1, "neg_binom": 2, "studentst": 3, "quantiles": 9, "mixture": 128
        }
        output_dims = output_dims_dict[head_type]

    if head_type != "mixture":
        out = get_preds(pred_head, latents, horizon_len, output_dims)  # [B, H, D]

    if head_type == "mse":
        pred = out[:, :, 0]
        if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
            pred = pred * sigma0[:, None] + mu0[:, None]
        if torch.is_tensor(labels):
            loss = F.mse_loss(pred, labels)
            results["loss"] = loss
        if forecast:
            results["mean"] = pred.detach().cpu()
            results["median"] = pred.detach().cpu()

    elif head_type == "quantiles":
        pred = out  # [B, H, Q]
        if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
            pred = pred * sigma0[:, None, None] + mu0[:, None, None]
        if torch.is_tensor(labels):
            loss = QuantileLoss(quantiles).loss(pred, labels).mean()
            results["loss"] = loss
        if forecast:
            results["median"] = pred[:, :, 4].detach().cpu()
            results["mean"] = pred[:, :, 4].detach().cpu()
            results["quantiles"] = einops.rearrange(pred.detach().cpu(), "B H Q -> Q B H")

    elif head_type == "mixture":
        assert mixture_output is not None, "mixture_output must be provided for head_type='mixture'"

        if latents.ndim == 3:
            out_feat_size = torch.full([latents.shape[0] * latents.shape[1]], horizon_len // latents.shape[1], dtype=int, device=latents.device)
            n_patches = latents.shape[1]
            latents = einops.rearrange(latents, "B patches d_model -> (B patches) d_model")
        else:
            n_patches = 0
            out_feat_size = torch.full([latents.shape[0]], horizon_len, dtype=int, device=latents.device)

        params = pred_head(latents, out_feat_size)

        if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
            if n_patches > 0:
                mu0 = mu0.repeat_interleave(n_patches, dim=0)
                sigma0 = sigma0.repeat_interleave(n_patches, dim=0)
            dist = mixture_output.distribution(
                params,
                loc=mu0[:, None],
                scale=sigma0[:, None],
            )
        else:
            dist = mixture_output.distribution(params)

        if torch.is_tensor(labels):
            if n_patches > 0:
                labels = einops.rearrange(labels, "B (patches patch_len) -> (B patches) patch_len",
                                          patches = n_patches)
            loss = -dist.log_prob(labels).mean()
            results["loss"] = loss

        if forecast:
            with torch.no_grad():
                mean = dist.mean.detach().cpu()  # [B, H]
                median_quantiles = [0.5] + quantiles
                q_vals = mixture_quantiles_by_sampling(dist, median_quantiles, num_samples=4096).cpu()  # [1+Q, B, H]
                if n_patches > 0:
                    mean = einops.rearrange(mean, "(B patches) patch_len -> B (patches patch_len)", patches=n_patches)
                    q_vals = einops.rearrange(
                        q_vals, "Q (B patches) patch_len -> Q B (patches patch_len)", patches=n_patches
                    )
                results["mean"] = mean
                results["median"] = q_vals[0]
                results["quantiles"] = q_vals[1:]

    else:
        # Distribution heads: gaussian, poisson, neg_binom, studentst
        if head_type == "gaussian":
            pred_mu = out[:, :, 0]
            pred_std = F.softplus(out[:, :, 1]) + eps
            if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
                pred_mu = pred_mu * sigma0[:, None] + mu0[:, None]
                pred_std = pred_std * sigma0[:, None]
            distribution = torch.distributions.Normal(pred_mu, pred_std)

        elif head_type == "poisson":
            if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
                pred_lambda = F.softplus(out[:, :, 0] + torch.log(mu0[:, None].clamp_min(eps))) + eps
            else:
                pred_lambda = F.softplus(out[:, :, 0]) + eps
            distribution = torch.distributions.Poisson(pred_lambda)

        elif head_type == "neg_binom":
            pred_mu = torch.exp(out[:, :, 0]) + eps
            pred_r = F.softplus(out[:, :, 1]) + eps
            if torch.is_tensor(mu0):
                pred_mu = pred_mu * mu0[:, None].clamp_min(eps)
            pred_p = pred_r / (pred_r + pred_mu)
            pred_logits = torch.log(pred_mu / pred_r)
            distribution = NegativeBinomial(total_count=pred_r, logits=pred_logits)

        elif head_type == "studentst":
            pred_df = F.softplus(out[:, :, 0]) + eps
            pred_mu = out[:, :, 1]
            pred_std = F.softplus(out[:, :, 2]) + eps
            if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
                pred_mu = pred_mu * sigma0[:, None] + mu0[:, None]
                pred_std = pred_std * sigma0[:, None]
            distribution = torch.distributions.StudentT(df=pred_df, loc=pred_mu, scale=pred_std)
        else:
            raise KeyError(f"{head_type} is not a valid distribution or loss")

        if torch.is_tensor(labels):
            if head_type == "poisson":
                # distribution.rate exists for Poisson; use NLL to match notebook
                loss = F.poisson_nll_loss(distribution.rate, labels, log_input=False)
            else:
                loss = -distribution.log_prob(labels).mean()
            results["loss"] = loss

        if forecast:
            results["mean"] = distribution.mean.detach().cpu()
            # Attempt generic quantiles via icdf; not all distributions implement icdf
            try:
                results["median"] = distribution.icdf(0.5 * torch.ones_like(out[:, :, 0])).detach().cpu()
                quantiles_torch = torch.tensor(quantiles, device=out.device).view(1, 1, -1).expand(out.shape[0], out.shape[1], -1)
                results["quantiles"] = distribution.icdf(quantiles_torch).detach().cpu()
                # Rearrange to [Q, B, H] for consistency with other heads
                results["quantiles"] = einops.rearrange(results["quantiles"], "B H Q -> Q B H")
            except Exception:
                # Some distributions (e.g., Poisson/NB) may not implement icdf; skip quantiles
                results["median"] = results["mean"]

    return results


# ---------------------------- Configuration ---------------------------- #

@dataclass
class TrainConfig:
    # Backbone / cache selection
    model_name: str = "yinglong"
    cache_glob: Optional[str] = None  # if None, inferred from model_name

    # Head types to train sequentially
    head_types: List[str] = None  # e.g., ["quantiles", "gaussian", "studentst"]

    # Data/model shapes and sizes (defaults inferred from model_name)
    context_len: int = 512
    horizon_len: Optional[int] = None   # inferred if None
    hidden_size: Optional[int] = None   # inferred if None
    step_size: Optional[int] = None     # inferred if None
    intermediate_size: int = 1280       # MLP hidden dims in head

    # Optimization
    batch_size: int = 512
    tot_iters: int = 2097152
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: Optional[float] = 1.0
    use_amp: bool = True  # autocast + GradScaler

    # Data filtering
    max_label: float = 1000.0
    min_label_default: float = -1000.0

    # IO
    save_dir: str = "models"
    save_prefix: Optional[str] = None  # e.g., run ID

    # Device
    device: Optional[str] = None  # "cuda", "cpu", or None to auto-detect

    # Quantiles
    quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    # Randomness
    seed: int = 42

    # Validation flags and settings
    validate: bool = False
    val_samples: int = 256

    # tsmixup dataset options for validation
    tsmixup_cache_dir: Optional[str] = None
    tsmixup_processed_cache: Optional[str] = None
    tsmixup_max_samples: int = 300000
    tsmixup_num_workers: int = 4
    tsmixup_loader_batch_size: int = 4000


# ---------------------------- Trainer ---------------------------- #

class HeadTrainer:
    """
    Orchestrates loading cached latents, building heads, and training losses per head type.
    Optionally validates on tsmixup by running the backbone to get latents/stats.
    """

    # Defaults per model_name (matches your notebook)
    FILE_NAME_DICT = {
        "timesfm": "./data/timesfm_cache_last_fp16/last_shard_*.pt",
        "chronos_bolt": "./data/chronos_bolt_cache_fp16/squeeze_shard_*.pt",
        "moirai2": "./data/moirai2_cache_fp16/last_ctx_shard_*.pt",
        "tirex": "./data/tirex_cache_fp16/full_shard_*.pt",
        "yinglong": "./data/yinglong_cache_fp16/full_shard_*.pt",
    }
    HIDDEN_SIZE = {
        "timesfm": 1280,
        "chronos_bolt": 768,
        "moirai2": 384,
        "tirex": 512,
        "yinglong": 768,
    }
    HORIZON_LEN = {
        "timesfm": 128,
        "chronos_bolt": 64,
        "moirai2": 64,
        "tirex": 128,
        "yinglong": 128,
    }
    STEP_SIZE = {
        "timesfm": 128,
        "chronos_bolt": 64,
        "moirai2": 64,
        "tirex": 32,
        "yinglong": 32,
    }
    PATCH_LEN = {
        "timesfm": 32,
        "chronos_bolt": 32,
        "moirai2": 16,
        "tirex": 32,
        "yinglong": 32,
    }
    OUTPUT_DIMS = {"mse": 1, "gaussian": 2, "poisson": 1, "neg_binom": 2, "studentst": 3, "quantiles": 9, "mixture": None}
    COUNT_HEADS = ("poisson", "neg_binom")

    def __init__(self, cfg: TrainConfig):
        self.cfg = self._populate_config_defaults(cfg)
        self.device = self._get_device(self.cfg.device)
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        os.makedirs(self.cfg.save_dir, exist_ok=True)

        # Dataset
        cache_glob = self.cfg.cache_glob or self.FILE_NAME_DICT[self.cfg.model_name]
        self.train_ds = FullLatentShardDataset(cache_glob)  # Provided by your project
        self.scaler = torch.amp.GradScaler(enabled=self.cfg.use_amp and self.device.type == "cuda")

        # Validation members
        self._val_datasets_ready = False
        self._val_dataset = None
        self._backbone_model = None

    def _populate_config_defaults(self, cfg: TrainConfig) -> TrainConfig:
        if cfg.head_types is None or len(cfg.head_types) == 0:
            cfg.head_types = ["quantiles"]
        if cfg.horizon_len is None:
            cfg.horizon_len = self.HORIZON_LEN[cfg.model_name]
        if cfg.hidden_size is None:
            cfg.hidden_size = self.HIDDEN_SIZE[cfg.model_name]
        if cfg.step_size is None:
            cfg.step_size = self.STEP_SIZE[cfg.model_name]
        if cfg.save_prefix is None:
            cfg.save_prefix = f"{cfg.model_name}"
        return cfg

    @staticmethod
    def _get_device(device_str: Optional[str]) -> torch.device:
        if device_str is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _build_head(self, head_type: str) -> Tuple[nn.Module, Optional[MixtureOutput]]:
        """
        Build the prediction head and (optionally) a MixtureOutput spec for mixture loss.
        """
        if head_type == "mixture":
            # MOIRAI V1 Mixture Components
            # components = [ 
            #     StudentTOutput(),
            #     NormalFixedScaleOutput(),
            #     NegativeBinomialOutput(), 
            #     LogNormalOutput(),
            # ]
            components = [ 
                StudentTOutput(),
                NormalFixedScaleOutput(),
                LaplaceOutput(),
                LogNormalOutput(),
            ]
            mixture_output = MixtureOutput(components)
            head = MixtureHead(
                in_features=self.cfg.hidden_size,
                hidden_dims=self.cfg.intermediate_size,
                horizon_len=self.cfg.step_size,  # per-patch output size
                mixture_output=mixture_output
            ).to(self.device).train()
            return head, mixture_output
        else:
            mixture_output = None
            output_dims = self.OUTPUT_DIMS[head_type]
            head = ResidualBlock(
                input_dims=self.cfg.hidden_size,
                output_dims=self.cfg.step_size * output_dims,
                hidden_dims=self.cfg.intermediate_size,
            ).to(self.device).train()
            return head, mixture_output

    def _get_batch(self, start_idx: int, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Pull a consecutive slice from the cached shard dataset.
        Returns: latents [B, ...], stats [B, 2], labels [B, H]
        """
        latents, stats, labels = [], [], []
        for i in range(batch_size):
            latent, stat, label = self.train_ds[start_idx + i]
            latents.append(latent)
            stats.append(stat)
            labels.append(label)
        latents = torch.stack(latents).to(self.device).float()
        stats = torch.stack(stats).to(self.device)
        labels = torch.stack(labels).to(self.device)
        return latents, stats, labels

    def _head_save_path(self, head_type: str) -> str:
        return os.path.join(self.cfg.save_dir, f"{self.cfg.save_prefix}_{head_type}_head.pt")

    # ---------------------------- Training ---------------------------- #

    def train_head(self, head_type: str) -> Dict[str, float]:
        """
        Train a single head type and save checkpoint. Returns a dict of summary stats (e.g., median loss hist tail).
        """
        output_dims = self.OUTPUT_DIMS[head_type]
        pred_head, mixture_output = self._build_head(head_type)
        optimizer = torch.optim.AdamW(pred_head.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        training_loss_median_curve: List[float] = []
        loss_batch: List[float] = []

        max_label = self.cfg.max_label
        min_label = 0 if head_type in self.COUNT_HEADS else self.cfg.min_label_default

        total_steps = self.cfg.tot_iters // self.cfg.batch_size
        log_batch = self.cfg.batch_size * (1024 // 4)  # logging cadence similar to notebook
        pbar_outer = tqdm(range(self.cfg.epochs), desc="Epochs", leave=True)
        for epoch in pbar_outer:
            pbar = tqdm(range(0, self.cfg.tot_iters, self.cfg.batch_size), desc=f"Epoch: {epoch}", leave=False)
            for batch_num in pbar:
                latents, stats, labels = self._get_batch(batch_num, self.cfg.batch_size)
                if self.cfg.model_name in ("tirex", "yinglong"):
                    latents = einops.rearrange(latents, "B patches d_model -> B (patches d_model)")

                mu0 = stats[:, 0].float()
                sigma0 = stats[:, 1].float()
                labels_norm = (labels.float() - mu0[:, None]) / (sigma0[:, None] + eps)

                mask = get_mask(latents, labels if head_type in self.COUNT_HEADS else labels_norm, 
                                max_label, min_label)
                if torch.any(mask == 0):
                    latents = latents[mask]
                    stats = stats[mask]
                    labels = labels[mask]
                    labels_norm = labels_norm[mask]
                    mu0 = mu0[mask]
                    sigma0 = sigma0[mask]

                if self.cfg.model_name in ("tirex", "yinglong"):
                    latents = einops.rearrange(latents, "B (patches d_model) -> B patches d_model", d_model=self.cfg.hidden_size)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self.device.type, dtype=torch.float32, enabled=self.cfg.use_amp and self.device.type == "cuda"):
                    results = latent_forecast(
                        pred_head,
                        latents,
                        horizon_len=self.cfg.horizon_len,
                        head_type=head_type,
                        output_dims=output_dims,
                        forecast=False,
                        labels=labels if head_type in self.COUNT_HEADS else labels_norm,
                        mu0=mu0 if head_type in self.COUNT_HEADS else None,
                        sigma0=sigma0 if head_type in self.COUNT_HEADS else None,
                        mixture_output=mixture_output,
                        quantiles=list(self.cfg.quantiles),
                    )
                    loss = results["loss"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                if self.cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(pred_head.parameters(), self.cfg.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()

                loss_batch.append(float(loss.detach().cpu()))

                # Logging median loss
                if (batch_num % log_batch) == (log_batch - self.cfg.batch_size):
                    train_med_loss = float(np.median(np.array(loss_batch)))
                    loss_batch = []
                    pbar.set_description(
                        f"epoch: {epoch} step: {batch_num // self.cfg.batch_size + 1}/{total_steps} "
                        f"loss: {train_med_loss:.4f}"
                    )
                    training_loss_median_curve.append(train_med_loss)

        # Save head
        ckpt_path = self._head_save_path(head_type)
        torch.save(pred_head.state_dict(), ckpt_path)

        # Save training curve JSON
        curve_path = os.path.splitext(ckpt_path)[0] + "_train_curve.json"
        with open(curve_path, "w") as f:
            json.dump({"median_loss_curve": training_loss_median_curve}, f)

        return {
            "head_type": head_type,
            "last_median_loss": training_loss_median_curve[-1] if len(training_loss_median_curve) > 0 else float("nan"),
            "num_log_points": len(training_loss_median_curve),
            "checkpoint": ckpt_path,
        }

    # ---------------------------- Validation helpers ---------------------------- #

    def _prepare_tsmixup_val(self):
        """
        Prepare tsmixup validation dataset (val split) using create_cached_tsmixup_datasets.
        """
        if self._val_datasets_ready:
            return
        if self.cfg.tsmixup_cache_dir is None or self.cfg.tsmixup_processed_cache is None:
            raise ValueError("For --validate, please provide --tsmixup-cache-dir and --tsmixup-processed-cache.")
        _, val_dataset = create_cached_tsmixup_datasets(
            max_samples=self.cfg.tsmixup_max_samples,
            context_length=self.cfg.context_len,
            prediction_length=self.cfg.horizon_len,
            num_workers=self.cfg.tsmixup_num_workers,
            cache_dir=self.cfg.tsmixup_cache_dir,
            processed_cache_path=self.cfg.tsmixup_processed_cache,
            batch_size=self.cfg.tsmixup_loader_batch_size,
        )
        self._val_dataset = val_dataset
        self._val_datasets_ready = True

    def _load_model(self):
        """
        Load the backbone model for validation latent extraction.
        """
        if self._backbone_model is not None:
            return self._backbone_model

        model_name = self.cfg.model_name
        device = self.device
        context_len = self.cfg.context_len

        if model_name == "timesfm":
            try:
                import timesfm
            except ImportError as e:
                raise RuntimeError("timesfm is required for --validate with model_name=timesfm") from e
            pred_len = self.cfg.horizon_len
            model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend='gpu' if device.type == "cuda" else 'cpu',
                    context_len=context_len,
                    horizon_len=pred_len,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=50,
                    model_dims=1280,
                    use_positional_embedding=False,
                    point_forecast_mode='mean',
                    device=str(device)
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
            )
            model._model.eval()
            model._model.to(device)

        elif model_name == "moirai2":
            try:
                from uni2ts.model.moirai2 import (Moirai2Forecast, Moirai2Module)
            except ImportError as e:
                raise RuntimeError("uni2ts Moirai2 is required for --validate with model_name=moirai2") from e
            pred_len = self.cfg.horizon_len
            model = Moirai2Forecast(
                module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
                prediction_length=pred_len,
                context_length=context_len,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            moirai = model.module
            moirai.eval()
            moirai.to(device)

        elif model_name == "chronos_bolt":
            try:
                from chronos import ChronosBoltPipeline
            except ImportError as e:
                raise RuntimeError("chronos-bolt is required for --validate with model_name=chronos_bolt") from e
            pred_len = self.cfg.horizon_len
            model = ChronosBoltPipeline.from_pretrained(
                "amazon/chronos-bolt-base",
                device_map=str(device),
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            )
            model.model.eval()

        elif model_name == "tirex":
            try:
                from tirex import load_model as load_tirex_model
            except ImportError as e:
                raise RuntimeError("tirex is required for --validate with model_name=tirex") from e
            model = load_tirex_model("NX-AI/TiRex")
            # keep on whatever device tirex prefers; we only read outputs (hooked/called)

        elif model_name == "yinglong":
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as e:
                raise RuntimeError("transformers is required for --validate with model_name=yinglong") from e
            model = AutoModelForCausalLM.from_pretrained(
                'qcw2333/YingLong_110m',
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
            ).to(device)
            model.eval()
        else:
            raise KeyError(f"Unsupported model_name for validation: {model_name}")

        self._backbone_model = model
        return self._backbone_model

    def _get_preds_latents(self, model, context: torch.Tensor, pred_len: int):
        """
        Extract latents and stats from the backbone for provided contexts (tsmixup batch).
        Returns: quantile_forecasts, transformer_output(latents), stats
        """
        model_name = self.cfg.model_name
        device = self.device
        batch_size = context.shape[0]
        patch_len = self.PATCH_LEN[model_name]
        context_len = self.cfg.context_len

        if model_name == "timesfm":
            freq = [0] * len(context)
            _, quantile_forecasts, (transformer_output, stats) = model.forecast(
                context, freq=freq, get_stacked_transformer=True
            )  # transformer_output: [B, N, 1280], stats: [B, 2]
            transformer_output = transformer_output[:, -1, :]  # [B, 1280]

        elif model_name == "moirai2":
            # Convert inputs and call forecast to get latents and stats
            context_ = context.to(device)[:, :, None]
            from uni2ts.model.moirai2 import (Moirai2Forecast)
            # Internals require conversion
            target, observed_mask, sample_id, time_id, variate_id, prediction_mask = model._convert(
                patch_size=patch_len,
                past_target=context_,                                 # B x past_time x D
                past_observed_target=torch.isfinite(context_),        # B x past_time x D (bool)
                past_is_pad=torch.full_like(context_[:, :, 0], False, dtype=bool),  # B x past_time (bool)
            )
            model.module.get_reprs = True
            transformer_output, stats = model.module(
                target, observed_mask, sample_id, time_id, variate_id, prediction_mask, False
            )
            transformer_output = transformer_output[prediction_mask].reshape([batch_size, -1, transformer_output.shape[-1]])
            stats = stats[prediction_mask].reshape([batch_size, -1, stats.shape[-1]])
            model.module.get_reprs = False
            forecasts = model.module(target, observed_mask, sample_id, time_id, variate_id, prediction_mask, False)
            quantile_forecasts = einops.rearrange(
                forecasts[:, context_len // patch_len, :],
                "B (pred_len quantiles) -> B pred_len quantiles", quantiles=9, pred_len=pred_len
            )
            stats = stats[:,0,:]

        elif model_name == "chronos_bolt":
            # Hook decoder to grab last hidden state, and instance_norm for stats
            transformer_output = torch.zeros(context.shape[0], model.model.model_dim)
            stats = torch.zeros(context.shape[0], 2)
            def save_decoder_hook(module, input, output):
                transformer_output[:] = output.last_hidden_state.squeeze().detach().cpu()
            def save_encoder_hook(module, input, output):
                stats[:] = torch.stack(output[1], dim=-1).squeeze().detach().cpu()

            model.model.decoder.register_forward_hook(save_decoder_hook)
            model.model.instance_norm.register_forward_hook(save_encoder_hook)
            context_ = context.to(device)
            forecasts = model.model(context_)
            quantile_forecasts = forecasts.quantile_forecasts  # [B, H, Q]

        elif model_name == "tirex":
            d_model = model.model_config.block_kwargs.embedding_dim
            out_patches = pred_len // model.model_config.output_patch_size
            transformer_output = torch.zeros(context.shape[0], out_patches, d_model)
            stats = torch.zeros((context.shape[0], 2))
            forecasts = model.forecast(context=context, prediction_length=pred_len, batch_size=batch_size,
                                       max_accelerated_rollout_steps=4,
                                       get_loc_scale=stats,
                                       get_hidden_states=transformer_output)
            quantile_forecasts = forecasts[0]

        elif model_name == "yinglong":
            context_ = context.to(device)
            d_model = model.config.n_embd
            patch_size = model.config.patch_size
            out_patches = pred_len // patch_size
            transformer_output = torch.zeros((context.shape[0], out_patches, d_model))
            stats = torch.zeros((context.shape[0], 2))
            start_ind = context.shape[0] // patch_size
            end_ind = start_ind + out_patches

            def save_decoder_hook(module, input, output):
                transformer_output[:] = input[0][:, start_ind:end_ind].detach().cpu()
            def save_tokenizer_hook(module, input, output):
                x, x_raw, masks, mean, std, _ = output
                stats[:, 0] = mean.flatten().detach().cpu()
                stats[:, 1] = std.flatten().detach().cpu()

            if hasattr(model.lm_head, "_forward_hooks"):
                model.lm_head._forward_hooks.clear()
            if hasattr(model.tokenizer, "_forward_hooks"):
                model.tokenizer._forward_hooks.clear()
            model.lm_head.register_forward_hook(save_decoder_hook)
            model.tokenizer.register_forward_hook(save_tokenizer_hook)

            forecast_len = 2048  # large number for DCoT
            forecasts = model.generate(context_, future_token=forecast_len)
            quantile_forecasts = forecasts[:, :pred_len, [9, 19, 29, 39, 49, 59, 69, 79, 89]].float().detach().cpu()

        else:
            raise KeyError(f"Unsupported model_name for validation: {model_name}")

        if model_name != "moirai2" and model_name != "chronos_bolt" and model_name != "tirex" and model_name != "yinglong":
            # timesfm path already defined quantile_forecasts above
            pass

        # Common returns:
        # timesfm: transformer_output [B, D], stats [B, 2]
        # moirai2/tirex/yinglong: transformer_output [B, P, D], stats [B, 2]
        return quantile_forecasts, transformer_output, stats

    def _gather_tsmixup_batch(self, val_dataset, ts: int, pred_length: int, ctx_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mimics notebook's load_dataset('tsmixup', ts,...): collect contexts and labels.
        """
        # Choose first ts or random indices
        n = len(val_dataset)
        if ts > n:
            ts = n
        # You can also randomize indices if desired:
        # idxs = np.random.choice(n, size=ts, replace=False)
        idxs = range(ts)
        x_list, y_list = [], []
        for i in idxs:
            val_dict = val_dataset[i]
            x_list.append(val_dict['past_values'])
            y_list.append(val_dict['future_values'])
        x = torch.stack(x_list)[:, -ctx_len:]
        y = torch.stack(y_list)[:, :pred_length]
        return x, y

    def _compute_val_metrics(self, y_true: torch.Tensor, results: Dict[str, torch.Tensor], head_type: str) -> Dict[str, float]:
        """
        Compute basic validation metrics:
        - MSE on mean
        - MAE on median (or mean if median not available)
        - Optional quantile pinball loss and 80% interval coverage if quantiles exist
        """
        y = y_true.detach().cpu()
        mean_pred = results.get("mean", None)
        median_pred = results.get("median", mean_pred)

        metrics = {}
        if mean_pred is not None:
            metrics["mse"] = float(((mean_pred - y) ** 2).mean().item())
        if median_pred is not None:
            metrics["mae"] = float((median_pred - y).abs().mean().item())

        qpred = results.get("quantiles", None)  # [Q, B, H]
        qs = list(self.cfg.quantiles)
        if qpred is not None and qpred.ndim == 3 and qpred.shape[0] == len(qs):
            # Pinball loss
            try:
                q_bhq = einops.rearrange(qpred, "Q B H -> B H Q")
                qloss = QuantileLoss(qs).loss(q_bhq, y).mean().item()
                metrics["pinball_loss"] = float(qloss)
            except Exception:
                pass
            # 80% coverage if 0.1 and 0.9 available
            if 0.1 in qs and 0.9 in qs:
                i_lo = qs.index(0.1)
                i_hi = qs.index(0.9)
                lo = qpred[i_lo]
                hi = qpred[i_hi]
                cov = ((y >= lo) & (y <= hi)).float().mean().item()
                metrics["p10_p90_coverage"] = float(cov)
        return metrics

    def validate_head(self, head_type: str) -> Dict[str, float]:
        """
        Validate a trained head on tsmixup using the real backbone to get latents/stats.
        Saves a JSON with metrics next to the checkpoint.
        """
        self._prepare_tsmixup_val()
        model = self._load_model()

        # Build head and load checkpoint
        pred_head, mixture_output = self._build_head(head_type)
        pred_head.eval()
        ckpt_path = self._head_save_path(head_type)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for validation: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        pred_head.load_state_dict(state)

        # Sample validation contexts/labels
        x, y = self._gather_tsmixup_batch(
            self._val_dataset,
            ts=self.cfg.val_samples,
            pred_length=self.cfg.horizon_len,
            ctx_len=self.cfg.context_len
        )
        # Get latents and stats from backbone
        quantiles_backbone, latents, stats = self._get_preds_latents(model, x, self.cfg.horizon_len)
        latents = latents.to(self.device)
        stats = stats.to(self.device)

        # (OLD - FIXED) Stats layout: for most backbones [B, 2], for moirai2 during get_reprs we shaped to [B, P, 2]
        if stats.ndim == 3 and stats.shape[-1] == 2:
            # Reduce per-patch stats by taking the first or mean across patches; mean is safer
            mu0 = stats[..., 0].mean(dim=1).float()
            sigma0 = stats[..., 1].mean(dim=1).float()
        else:
            mu0 = stats[:, 0].float()
            sigma0 = stats[:, 1].float()

        # Forward head to get predictions in original units
        output_dims = self.OUTPUT_DIMS[head_type]
        with torch.no_grad():
            results = latent_forecast(
                pred_head,
                latents,
                horizon_len=self.cfg.horizon_len,
                head_type=head_type,
                output_dims=output_dims,
                forecast=True,
                labels=None,
                mu0=mu0 if head_type in self.COUNT_HEADS else mu0,  # pass for all to denorm when applicable
                sigma0=sigma0 if head_type in self.COUNT_HEADS else sigma0,
                mixture_output=mixture_output,
                quantiles=list(self.cfg.quantiles),
            )

        # Compute metrics vs ground truth y
        metrics = self._compute_val_metrics(y_true=y, results=results, head_type=head_type)

        # Save metrics
        val_path = os.path.splitext(ckpt_path)[0] + "_val.json"
        with open(val_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Validation metrics for {head_type}: {json.dumps(metrics, indent=2)}")
        return metrics

    # ---------------------------- Run ---------------------------- #

    def run(self) -> List[Dict[str, float]]:
        """
        Train all head types in order and return summaries. Optionally validate each head on tsmixup.
        """
        summaries = []
        print("Training configuration:")
        print(json.dumps(asdict(self.cfg), indent=2))
        for head_type in self.cfg.head_types:
            print(f"\n=== Training head: {head_type} ===")
            summary = self.train_head(head_type)
            print(f"Saved: {summary['checkpoint']}")
            print(f"Last median loss: {summary['last_median_loss']:.6f} (points: {summary['num_log_points']})")
            summaries.append(summary)
            if self.cfg.validate:
                print(f"--- Validating head: {head_type} on tsmixup ---")
                try:
                    self.validate_head(head_type)
                except Exception as e:
                    print(f"Validation failed for head {head_type}: {e}")
            # Optional: free CUDA memory between heads
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return summaries


# ---------------------------- CLI ---------------------------- #

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train prediction heads on cached latents with optional tsmixup validation.")
    parser.add_argument("--model-name", type=str, default="yinglong",
                        choices=["timesfm", "chronos_bolt", "moirai2", "tirex", "yinglong"])
    parser.add_argument("--cache-glob", type=str, default=None,
                        help="Glob path to cached latent shards (overrides default mapping for model_name).")
    parser.add_argument("--heads", type=str, nargs="+", default=["quantiles"],
                        help="Head types to train sequentially. Choices: mse gaussian poisson neg_binom studentst quantiles mixture")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tot-iters", type=int, default=2097152)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true", default=False, help="Enable autocast+GradScaler on CUDA.")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--save-prefix", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--horizon-len", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=1280)
    parser.add_argument("--max-label", type=float, default=1000.0)
    parser.add_argument("--min-label-default", type=float, default=-1000.0)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    parser.add_argument("--seed", type=int, default=42)

    # Validation flags
    parser.add_argument("--validate", action="store_true", default=False, help="Run tsmixup validation after training each head.")
    parser.add_argument("--val-samples", type=int, default=256, help="Number of validation samples to draw from tsmixup.")

    # tsmixup dataset args
    parser.add_argument("--tsmixup-cache-dir", type=str, default=None, help="Cache directory for tsmixup dataset.")
    parser.add_argument("--tsmixup-processed-cache", type=str, default=None, help="Processed cache path for tsmixup dataset.")
    parser.add_argument("--tsmixup-max-samples", type=int, default=300000)
    parser.add_argument("--tsmixup-num-workers", type=int, default=8)
    parser.add_argument("--tsmixup-loader-batch-size", type=int, default=4000)

    args = parser.parse_args()
    cfg = TrainConfig(
        model_name=args.model_name,
        cache_glob=args.cache_glob,
        head_types=args.heads,
        batch_size=args.batch_size,
        tot_iters=args.tot_iters,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_amp=args.use_amp,
        save_dir=args.save_dir,
        save_prefix=args.save_prefix,
        device=args.device,
        context_len=args.context_len,
        hidden_size=args.hidden_size,
        horizon_len=args.horizon_len,
        step_size=args.step_size,
        intermediate_size=args.intermediate_size,
        max_label=args.max_label,
        min_label_default=args.min_label_default,
        quantiles=tuple(args.quantiles),
        seed=args.seed,
        validate=args.validate,
        val_samples=args.val_samples,
        tsmixup_cache_dir=args.tsmixup_cache_dir,
        tsmixup_processed_cache=args.tsmixup_processed_cache,
        tsmixup_max_samples=args.tsmixup_max_samples,
        tsmixup_num_workers=args.tsmixup_num_workers,
        tsmixup_loader_batch_size=args.tsmixup_loader_batch_size,
    )
    return cfg


def main():
    cfg = parse_args()
    trainer = HeadTrainer(cfg)
    summaries = trainer.run()
    print("\nAll heads trained. Summaries:")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()