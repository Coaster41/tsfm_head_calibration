import os
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.distributions import (Normal, StudentT, Poisson)
import torch.nn.functional as F
import einops

# Project/Training-side imports
from utils.components import ResidualBlock
from pytorch_forecasting.metrics.quantile import QuantileLoss
from uni2ts.distribution import (
    MixtureOutput,
    StudentTOutput,
    NormalFixedScaleOutput,
    LogNormalOutput,
    LaplaceOutput,
)
from uni2ts.distribution.negative_binomial import NegativeBinomial
from scipy.special import stdtrit
from scipy.stats import (poisson, nbinom)
import math

# Reuse eps and latent_forecast + MixtureHead from the trainer script or define here
eps = 1e-6

def mixture_quantiles_by_sampling(dist, qs, num_samples=4096):
    with torch.no_grad():
        samples = dist.sample((num_samples,))
        q_tensor = torch.tensor(qs, device=samples.device, dtype=samples.dtype)
        q_vals = torch.quantile(samples, q_tensor, dim=0)
    return q_vals

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

def get_preds(model: nn.Module, latents: Tensor, horizon_len: int = 128, output_dims: int = 1,
              reshape: bool = True):
    out = model(latents)
    if reshape:
        if out.ndim == 3:
            out = einops.rearrange(
                out, "batch patches (patch_len output_dims) -> batch (patches patch_len) output_dims",
                output_dims=output_dims
            )
        else:
            out = einops.rearrange(
                out, "batch (horizon_len output_dims) -> batch horizon_len output_dims",
                horizon_len=horizon_len, output_dims=output_dims
            )
    return out

def latent_forecast(pred_head: nn.Module, latents: Tensor, horizon_len: int, head_type: str, 
                    output_dims: int | None = None, labels: Tensor | None = None, 
                    mu0: Tensor | None = None, sigma0: Tensor | None = None,
                    quantiles: list[float] | None = None, forecast: bool = True, 
                    mixture_output: MixtureOutput | None = None):
    if quantiles == None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    if output_dims == None:
        output_dims_dict = {"mse": 1, "gaussian": 2, "poisson": 1, "neg_binom": 2, "studentst": 3, 
                            "quantiles": 9, "mixture": 128}
        output_dims = output_dims_dict[head_type]
    
    if head_type != "mixture":
        out = get_preds(pred_head, latents, horizon_len, output_dims) # [B, H, D]
    if head_type == "mse":
        pred = out[:, :, 0] # [B, H]
        if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
            pred = pred * sigma0[:, None] + mu0[:, None]
        if torch.is_tensor(labels):
            loss = F.mse_loss(pred, labels)
            results["loss"] = loss
        if forecast:
            results["mean"] = pred.detach().cpu()
            results["median"] = pred.detach().cpu()

    elif head_type == "quantiles":
        pred = out # [B, H, Q]
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

        # 1) Predict mixture parameters
        # pred_head is MixtureHead and returns a PyTree with keys:
        #   "weights_logits" -> [B, H, K]
        #   "components" -> list of dicts with the per-component parameters ([B, H])
        if latents.ndim == 3:
            out_feat_size = torch.full([latents.shape[0] * latents.shape[1]], horizon_len // latents.shape[1], dtype=int, device=latents.device)
            n_patches = latents.shape[1]
            latents = einops.rearrange(latents, "B patches d_model -> (B patches) d_model")
        else:
            n_patches = 0
            out_feat_size = torch.full([latents.shape[0]], horizon_len, dtype=int, device=latents.device)
        params = pred_head(latents, out_feat_size)
        # print(f"isfinite: params['components'][2]['total_count'] {torch.isfinite(params['components'][2]['total_count']).all()}, shape {params['components'][2]['total_count'].shape} min {params['components'][2]['total_count'].min():.2e} max {params['components'][2]['total_count'].max():.2e}")
        # print(f"isfinite: params['components'][2]['logits'] {torch.isfinite(params['components'][2]['logits']).all()}, shape {params['components'][2]['logits'].shape} min {params['components'][2]['logits'].min():.2e} max {params['components'][2]['logits'].max():.2e}")


        # 2a) TRAINING: If you trained on normalized labels (continuous):
        #     Use dist on normalized space (no affine transform).
        # 2b) TRAINING (counts): keep labels as integers and use as-is.
        if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
            if n_patches > 0: # repeat for added dimension
                mu0 = mu0.repeat_interleave(n_patches, dim=0)
                sigma0 = sigma0.repeat_interleave(n_patches, dim=0)
            dist = mixture_output.distribution(
                params,
                loc=mu0[:, None],     # [B, 1] -> broadcast to [B, H]
                scale=sigma0[:, None]
            )
        else:
            dist = mixture_output.distribution(params)

        if torch.is_tensor(labels):
            if n_patches > 0:
                labels = einops.rearrange(labels, "B (patches patch_len) -> (B patches) patch_len",
                                          patches = n_patches)
            # print(f"labels: min {labels.min()} max {labels.max()} shape {labels.shape} isreal {torch.isreal(labels).all()}")
            loss = -dist.log_prob(labels).mean()
            results["loss"] = loss

        if forecast:
            # 3) FORECAST: return distribution in original units by affine transform (if you used normalized training)
            with torch.no_grad():
                mean = dist.mean.detach().cpu()  # [B, H]
                # quantiles via sampling
                median_quantiles = [0.5] + quantiles
                q_vals = mixture_quantiles_by_sampling(dist, median_quantiles, num_samples=4096).cpu()  # [1+Q, B, H]
                if n_patches > 0:
                    mean = einops.rearrange(mean, "(B patches) patch_len -> B (patches patch_len)", patches=n_patches)
                    q_vals = einops.rearrange(q_vals, "Q (B patches) patch_len -> Q B (patches patch_len)", patches=n_patches)
                results["mean"] = mean
                results["median"] = q_vals[0] 
                results["quantiles"] = q_vals[1:]

    else: # distribution loss
        if head_type == "gaussian":
            pred_mu = out[:, :, 0]
            pred_std = F.softplus(out[:, :, 1]) + eps
            if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
                pred_mu = pred_mu * sigma0[:, None] + mu0[:, None]
                pred_std = pred_std * sigma0[:, None]
            distribution = Normal(pred_mu, pred_std)

        elif head_type == "poisson":
            if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
                pred_lambda = F.softplus(out[:, :, 0] + torch.log(mu0[:, None].clamp_min(eps))) + eps
            else:
                pred_lambda = F.softplus(out[:, :, 0]) + eps
            distribution = Poisson(pred_lambda)

        elif head_type == "neg_binom":
            pred_mu =  torch.exp(out[:, :, 0]) + eps # any real
            pred_r = F.softplus(out[:, :, 1])  + eps # dispersion > 0
            if torch.is_tensor(mu0):
                pred_mu = pred_mu * mu0[:, None].clamp_min(eps)
            pred_p = pred_r / (pred_r + pred_mu) # Pytorch/Scipy
            # distribution = NegativeBinomial(total_count=pred_r, probs=pred_p) # Pytorch
            pred_logits = torch.log(pred_mu / pred_r)
            distribution = NegativeBinomial(total_count=pred_r, logits=pred_logits)
            
        elif head_type == "studentst":
            pred_df = F.softplus(out[:, :, 0]) + eps
            pred_mu = out[:, :, 1]
            pred_std = F.softplus(out[:, :, 2]) + eps
            if torch.is_tensor(mu0) and torch.is_tensor(sigma0):
                pred_mu = pred_mu * sigma0[:, None] + mu0[:, None]
                pred_std = pred_std * sigma0[:, None]
            distribution = StudentT(pred_df, loc=pred_mu, scale=pred_std)
        else:
            raise KeyError(f"{head_type} is not a valid distribution or loss")
        if torch.is_tensor(labels):
            if head_type == "poisson":
                loss = F.poisson_nll_loss(pred_lambda, labels, log_input=False)
            else:
                loss = -distribution.log_prob(labels).mean()
            results["loss"] = loss
        if forecast:
            results["mean"] = distribution.mean.detach().cpu()
            if head_type in  ("studentst", "poisson", "neg_binom"):
                median_quantiles = [0.5] + quantiles
                p = torch.tensor(median_quantiles)[:, None, None] * \
                    torch.ones((len(median_quantiles), out.shape[0], out.shape[1]))
                if head_type == "studentst":
                    preds = stdtrit(pred_df.detach().cpu().repeat(len(median_quantiles), 1, 1), p)
                    preds = preds * pred_std.detach().cpu() + pred_mu.detach().cpu()
                if head_type == "poisson":
                    preds = poisson.ppf(p, pred_lambda.detach().cpu().repeat(len(median_quantiles), 1, 1))
                if head_type == "neg_binom":
                    preds = nbinom.ppf(p, pred_r.detach().cpu().repeat(len(median_quantiles), 1, 1), 
                                       pred_p.detach().cpu().repeat(len(median_quantiles), 1, 1))
                results["median"] = preds[0]
                results["quantiles"] = preds[1:]

            else:
                results["median"] = distribution.icdf(0.5*torch.ones_like(out[:,:,0])).detach().cpu()
                quantile_preds = []
                for quantile in quantiles:
                    quantile_preds.append(distribution.icdf(quantile*torch.ones_like(out[:,:,0])).detach().cpu())
                results["quantiles"] = torch.stack(quantile_preds)

    return results


@dataclass
class ForecastConfig:
    model_name: str = "yinglong"
    device: Optional[str] = None
    context_len: int = 512
    horizon_len: Optional[int] = None
    hidden_size: Optional[int] = None
    step_size: Optional[int] = None
    intermediate_size: int = 1280
    quantiles: Tuple[float, ...] = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
    ckpt_dir: str = "models"
    ckpt_prefix: Optional[str] = None  # defaults to model_name


class HeadForecaster:
    """
    Forecasts using:
    - Backbone's default forecaster (returns median/quantiles)
    - Multiple custom trained heads (quantiles, studentst, gaussian, mixture, ...)

    Usage:
        cfg = ForecastConfig(model_name="yinglong")
        forecaster = HeadForecaster(cfg)
        forecaster.load_heads(["quantiles", "studentst", "gaussian", "mixture"])  # or provide explicit paths
        outputs = forecaster.forecast(context_tensor)  # dict with 'backbone' and per-head dicts
    """

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

    def __init__(self, cfg: ForecastConfig):
        self.cfg = self._populate_defaults(cfg)
        self.device = torch.device(self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.quantiles = list(self.cfg.quantiles)
        self._model = None
        self._heads: Dict[str, nn.Module] = {}
        self._mixture_output: Optional[MixtureOutput] = None

    def _populate_defaults(self, cfg: ForecastConfig) -> ForecastConfig:
        if cfg.horizon_len is None:
            cfg.horizon_len = self.HORIZON_LEN[cfg.model_name]
        if cfg.hidden_size is None:
            cfg.hidden_size = self.HIDDEN_SIZE[cfg.model_name]
        if cfg.step_size is None:
            cfg.step_size = self.STEP_SIZE[cfg.model_name]
        if cfg.ckpt_prefix is None:
            cfg.ckpt_prefix = cfg.model_name
        return cfg

    # ---------- Backbone loading and latent extraction ---------- #

    def _load_backbone(self):
        if self._model is not None:
            return self._model

        model_name = self.cfg.model_name
        device = self.device
        context_len = self.cfg.context_len
        pred_len = self.cfg.horizon_len

        if model_name == "timesfm":
            import timesfm
            if pred_len != 128: # pred_len is not default: Error out
                raise ValueError(F"pred_len for TimesFM must be 128, it is set as {pred_len}")
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
                    device=str(device),
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
            )
            model._model.eval()
            model._model.to(device)

        elif model_name == "moirai2":
            from uni2ts.model.moirai2 import (Moirai2Forecast, Moirai2Module)
            model = Moirai2Forecast(
                module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
                prediction_length=pred_len,
                context_length=context_len,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            if pred_len != 64: # pred_len is not default: Changing the AR step size
                num_predict_token = math.ceil(pred_len / self.PATCH_LEN[model_name])
                model.module.num_predict_token = num_predict_token
            model.module.eval()
            model.module.to(device)

        elif model_name == "chronos_bolt":
            if pred_len != 64: # pred_len is not default: Error out
                raise ValueError(F"pred_len for Chronos_Bolt must be 64, it is set as {pred_len}")
            from chronos import ChronosBoltPipeline
            model = ChronosBoltPipeline.from_pretrained(
                "amazon/chronos-bolt-base",
                device_map=str(device),
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            )
            model.model.eval()

        elif model_name == "tirex":
            from tirex import load_model as load_tirex_model
            model = load_tirex_model("NX-AI/TiRex")
            # model internal device handling

        elif model_name == "yinglong":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                'qcw2333/YingLong_110m',
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
            ).to(device)
            model.eval()
        else:
            raise KeyError(f"Unsupported model_name: {model_name}")

        self._model = model
        return self._model

    def _get_preds_latents(self, model, context: Tensor, pred_len: int):
        """
        Updated get_preds_latents as provided by you. Returns:
        - quantile_forecasts: [B, pred_len, Q]
        - transformer_output (latents): [B, D] or [B, P, D]
        - stats: [B, 2] or [B, P, 2]
        """
        model_name = self.cfg.model_name
        device = self.device
        batch_size = context.shape[0]
        context_len = self.cfg.context_len
        patch_len = self.PATCH_LEN[model_name]

        if model_name == "timesfm":
            freq = [0] * len(context)
            _, quantile_forecasts, (transformer_output, stats) = model.forecast(
                context, freq=freq, get_stacked_transformer=True
            )
            transformer_output = transformer_output[:, -1, :].detach().cpu()
            quantile_forecasts = torch.tensor(quantile_forecasts[:, :, 1:])

        elif "moirai" in model_name:
            context_ = context.to(device)[:, :, None]
            target, observed_mask, sample_id, time_id, variate_id, prediction_mask = model._convert(
                patch_size=patch_len,
                past_target=context_,
                past_observed_target=torch.isfinite(context_),
                past_is_pad=torch.full_like(context_[:, :, 0], False, dtype=bool),
            )
            if model_name == "moirai2":
                patch_sizes = False
            else:
                patch_sizes = torch.ones_like(time_id, dtype=torch.long) * patch_len
            model.module.get_reprs = True
            transformer_output, stats = model.module(
                target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_sizes
            )
            transformer_output = transformer_output[:,context_len//patch_len-1,:].detach().cpu()
            stats = stats[:,0,:].detach().cpu()
            model.module.get_reprs = False
            forecasts = model.module(target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_sizes)
            forecasts = einops.rearrange(forecasts[:,context_len//patch_len-1,:],
                                    "B (patches quantiles patch_len) -> B patches quantiles patch_len",
                                    quantiles = 9, patch_len = patch_len)
            quantile_forecasts = einops.rearrange(forecasts, "B patches quantiles patch_len -> B (patches patch_len) quantiles").detach().cpu()

        elif model_name == "chronos_bolt":
            transformer_output = torch.zeros(context.shape[0], model.model.model_dim)
            stats = torch.zeros(context.shape[0], 2)
            def save_decoder_hook(module, inputs, output):
                transformer_output[:] = output.last_hidden_state.squeeze().detach().cpu()
            def save_encoder_hook(module, inputs, output):
                stats[:] = torch.stack(output[1], dim=-1).squeeze().detach().cpu()
            decoder_hook_handle  = model.model.decoder.register_forward_hook(save_decoder_hook)
            instance_norm_hook_handle = model.model.instance_norm.register_forward_hook(save_encoder_hook)
            context_ = context.to(device)
            forecasts = model.model(context_)
            decoder_hook_handle.remove()
            instance_norm_hook_handle.remove()
            quantile_forecasts = einops.rearrange(forecasts.quantile_preds, "B Q h -> B h Q").detach().cpu()

        elif model_name == "tirex":
            d_model = model.model_config.block_kwargs.embedding_dim
            patch_size = model.model_config.output_patch_size
            out_patches = math.ceil(pred_len / patch_size)
            transformer_output = torch.zeros(context.shape[0], out_patches, d_model)
            stats = torch.zeros((context.shape[0], 2))
            forecasts = model.forecast(context=context, prediction_length=pred_len, batch_size=batch_size,
                                       max_accelerated_rollout_steps=out_patches,
                                       get_loc_scale=stats,
                                       get_hidden_states=transformer_output)
            quantile_forecasts = forecasts[0].detach().cpu()

        elif model_name == "yinglong":
            context_ = context.to(device)
            d_model = model.config.n_embd
            patch_size = model.config.patch_size
            out_patches = pred_len // patch_size
            transformer_output = torch.zeros((context.shape[0], out_patches, d_model))
            stats = torch.zeros((context.shape[0], 2))
            start_ind = context.shape[1] // patch_size
            end_ind = start_ind + out_patches
            def save_decoder_hook(module, inputs, output):
                transformer_output[:] = inputs[0][:, start_ind:end_ind].detach().cpu()
            def save_tokenizer_hook(module, inputs, output):
                x, x_raw, masks, mean, std, _ = output
                stats[:, 0] = mean.flatten().detach().cpu()
                stats[:, 1] = std.flatten().detach().cpu()
            if hasattr(model.lm_head, "_forward_hooks"):
                model.lm_head._forward_hooks.clear()
            if hasattr(model.tokenizer, "_forward_hooks"):
                model.tokenizer._forward_hooks.clear()
            lm_head_hook = model.lm_head.register_forward_hook(save_decoder_hook)
            tokenizer_hook = model.tokenizer.register_forward_hook(save_tokenizer_hook)
            forecast_len = 2048
            forecasts = model.generate(context_, future_token=forecast_len)
            lm_head_hook.remove()
            tokenizer_hook.remove()
            quantile_forecasts = forecasts[:, :pred_len, [9,19,29,39,49,59,69,79,89]].float().detach().cpu()
        else:
            raise KeyError(f"Unsupported model_name for forecasting: {model_name}")

        return quantile_forecasts, transformer_output, stats.detach().cpu()

    # ---------- Head loading/building ---------- #

    def _build_head(self, head_type: str) -> Tuple[nn.Module, Optional[MixtureOutput]]:
        if head_type == "mixture":
            components = [StudentTOutput(), NormalFixedScaleOutput(), LaplaceOutput(), LogNormalOutput()]
            mixture_output = MixtureOutput(components)
            head = MixtureHead(
                in_features=self.cfg.hidden_size,
                hidden_dims=self.cfg.intermediate_size,
                horizon_len=self.cfg.step_size,
                mixture_output=mixture_output
            ).to(self.device).eval()
            return head, mixture_output
        else:
            mixture_output = None
            output_dims = self.OUTPUT_DIMS[head_type]
            head = ResidualBlock(
                input_dims=self.cfg.hidden_size,
                output_dims=self.cfg.step_size * output_dims,
                hidden_dims=self.cfg.intermediate_size,
            ).to(self.device).eval()
            return head, mixture_output

    def load_heads(self, head_types: List[str], ckpt_paths: Optional[Dict[str, str]] = None):
        """
        Load multiple heads. If ckpt_paths is not provided, will look for:
            {ckpt_dir}/{ckpt_prefix}_{head_type}_head.pt
        """
        self._heads.clear()
        self._mixture_output = None  # rebuilt per mixture head
        for h in head_types:
            head, mixture_output = self._build_head(h)
            # pick checkpoint path
            if ckpt_paths and h in ckpt_paths:
                ckpt = ckpt_paths[h]
            else:
                ckpt = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.ckpt_prefix}_{h}_head.pt")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Checkpoint not found for head {h}: {ckpt}")
            state = torch.load(ckpt, map_location=self.device)
            head.load_state_dict(state)
            self._heads[h] = (head, mixture_output)
        return list(self._heads.keys())

    # ---------- Forecast API ---------- #

    def forecast(self, context: Tensor, heads: Optional[List[str]] = None) -> Dict[str, Dict[str, Tensor]]:
        """
        context: [B, context_len] tensor (float)
        Returns:
          {
            head_type: { "mean": [B,H], "median": [B,H], "quantiles": [Q,B,H], (maybe "loss" if computed off labels) }
          }
        """
        assert context.ndim == 2 and context.shape[1] == self.cfg.context_len, \
            f"context must be [B, {self.cfg.context_len}]"
        model = self._load_backbone()
        B = context.shape[0]
        pred_len = self.cfg.horizon_len

        # Backbone default forecasts + latents/stats
        with torch.no_grad():
            q_forecasts, latents, stats = self._get_preds_latents(model, context, pred_len)
            # q_forecasts: [B, H, Q]
            # latents: [B, D] or [B, P, D]
            # stats: [B, 2] or [B, P, 2]

        # Compute backbone outputs in a latent_forecast-like dict
        # Use the 0.5 (median) quantile and provide "quantiles" as [Q,B,H]
        if q_forecasts.ndim == 3:
            # determine median index (closest to 0.5)
            qs = self.quantiles
            try:
                median_idx = qs.index(0.5)
            except ValueError:
                # Pick nearest to 0.5
                median_idx = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5))
            backbone_quantiles = einops.rearrange(q_forecasts, "B H Q -> Q B H")
            backbone_median = backbone_quantiles[median_idx]
            backbone_mean = backbone_median  # no mean provided; use median as proxy
            backbone_out = {
                "mean": backbone_mean,
                "median": backbone_median,
                "quantiles": backbone_quantiles,
            }
        else:
            raise RuntimeError("Backbone did not return quantile forecasts of shape [B, H, Q].")

        # Prepare mu0/sigma0
        if stats.ndim == 3 and stats.shape[-1] == 2:
            mu0 = stats[..., 0].mean(dim=1).float().to(self.device)
            sigma0 = stats[..., 1].mean(dim=1).float().to(self.device)
        else:
            mu0 = stats[:, 0].float().to(self.device)
            sigma0 = stats[:, 1].float().to(self.device)

        # Ensure latents on device
        latents = latents.to(self.device)

        # Choose which heads to run
        if heads is None:
            heads = list(self._heads.keys())

        # Run each head
        heads_out: Dict[str, Dict[str, Tensor]] = {}
        for h in heads:
            if h not in self._heads:
                raise KeyError(f"Head not loaded: {h}. Call load_heads([...]) first.")
            head_module, mix_out = self._heads[h]
            output_dims = self.OUTPUT_DIMS[h]
            with torch.no_grad():
                results = latent_forecast(
                    head_module,
                    latents,
                    horizon_len=self.cfg.horizon_len,
                    head_type=h,
                    output_dims=output_dims,
                    labels=None,
                    mu0=mu0,     # pass for denorm/scaling
                    sigma0=sigma0,
                    quantiles=self.quantiles,
                    forecast=True,
                    mixture_output=mix_out,
                )
            heads_out[h] = results
        heads_out["backbone"] = backbone_out

        return heads_out
    

    def autoregressive_forecast(
        self,
        context: Tensor,
        pred_len: int,
        step_len: int,
        heads: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Autoregressive multi-quantile forecasting.

        Args:
          context: [B, context_len] float tensor
          pred_len: total forecast length to generate (> step_len)
          step_len: AR step size (<= self.cfg.horizon_len)
          heads: which heads to use (same semantics as forecast). If None, uses loaded heads.
                 As in `forecast`, the returned dict will always include a "backbone" entry.

        Returns:
          Dict[str, Dict[str, Tensor]] mapping:
            {
              head_type (incl. "backbone"): {
                "mean":      [B, pred_len] (CPU tensor),
                "median":    [B, pred_len] (CPU tensor),
                "quantiles": [Q, B, pred_len] (CPU tensor),
              },
              ...
            }
        """
        # Basic checks
        assert context.ndim == 2 and context.shape[1] == self.cfg.context_len, \
            f"context must be [B, {self.cfg.context_len}]"
        if not (1 <= step_len <= self.cfg.horizon_len):
            raise ValueError(f"step_len must be in [1, {self.cfg.horizon_len}], got {step_len}")
        if not (pred_len > 0 and pred_len >= step_len):
            raise ValueError(f"pred_len must be >= step_len (> 0). Got pred_len={pred_len}, step_len={step_len}")

        B = context.shape[0]
        Qs = self.quantiles
        Q = len(Qs)
        # Find index of 0.5 (or nearest) for median extraction
        try:
            median_idx = Qs.index(0.5)
        except ValueError:
            median_idx = min(range(Q), key=lambda i: abs(Qs[i] - 0.5))

        # 1) Initial forecast on the original context (one call)
        base_out = self.forecast(context, heads=heads)

        # Determine which entries to produce AR for (always include backbone)
        # base_out contains at least "backbone"; and potentially additional heads
        run_keys = [k for k in base_out.keys() if k == "backbone" or (heads is None or k in heads)]

        # Storage for AR outputs
        ar_quant_chunks: Dict[str, List[Tensor]] = {k: [] for k in run_keys}   # list of [Q, B, chunk_len]
        ar_mean_chunks: Dict[str, List[Tensor]] = {k: [] for k in run_keys}    # list of [B, chunk_len]

        # 2) Build q contexts per head by appending first step_len quantile paths from the initial forecast
        #    Also, store the first chunk outputs from base_out.
        # contexts_per_key: key -> contexts tensor of shape [Q, B, context_len]
        contexts_per_key: Dict[str, Tensor] = {}

        first_chunk_len = min(step_len, pred_len)
        for key in run_keys:
            bo = base_out[key]
            # bo["quantiles"]: [Q, B, H]
            if torch.is_tensor(bo["quantiles"]):
                q_init = bo["quantiles"][:, :, :first_chunk_len].detach().cpu()          # [Q, B, step_len or remaining]
                mean_init = bo["mean"][:, :first_chunk_len].detach().cpu()               # [B, chunk]
            else:
                q_init = torch.tensor(bo["quantiles"][:, :, :first_chunk_len])
                mean_init = torch.tensor(bo["mean"][:, :first_chunk_len])
            # Store outputs for the first chunk directly from the initial forecast
            ar_quant_chunks[key].append(q_init)
            ar_mean_chunks[key].append(mean_init)

            # Build q contexts for this key by appending each quantile path
            # Shift context by first_chunk_len and append q_init[k] for each k
            # Resulting shape: [Q, B, context_len]
            shifted = context[:, first_chunk_len:]  # [B, C - first_chunk_len]
            q_contexts = []
            for k in range(Q):
                q_ctx_k = torch.cat([shifted, q_init[k].to(context.dtype)], dim=1)
                q_contexts.append(q_ctx_k)
            contexts_per_key[key] = torch.stack(q_contexts, dim=0)  # [Q, B, context_len]

        generated = first_chunk_len

        # Helper for quantile aggregation across q^2 grid
        def aggregate_q2_to_q(q2_grid: Tensor) -> Tensor:
            """
            q2_grid: [Q_pred, Q_scen, B, L] -> aggregate across the first two dims (Q_pred*Q_scen)
            returns: [Q, B, L]
            """
            # Flatten (Q_pred, Q_scen) into samples dim
            samples = q2_grid.reshape(Q * Q, B, q2_grid.shape[-1])  # [Q*Q, B, L]
            # quantile over dim=0 (the "samples" dimension)
            probs = torch.tensor(Qs, dtype=samples.dtype, device=samples.device)
            # Output: [Q, B, L]
            return torch.quantile(samples, probs, dim=0)

        # 3-5) AR loop until we reach pred_len
        while generated < pred_len:
            chunk_len = min(step_len, pred_len - generated)
            for key in run_keys:
                # Prepare batch contexts [Q*B, context_len]
                q_contexts = contexts_per_key[key]               # [Q, B, C]
                qb, C = q_contexts.shape[0] * q_contexts.shape[1], q_contexts.shape[2]
                contexts_batch = q_contexts.reshape(qb, C)       # [Q*B, C]

                # Forecast on these q contexts for this specific key
                # - If key == "backbone": no extra heads -> pass heads=[]
                # - Else: pass heads=[key] to only compute that head (backbone will still be in output; we will select the needed key)
                if key == "backbone":
                    out = self.forecast(contexts_batch, heads=[])
                else:
                    out = self.forecast(contexts_batch, heads=[key])

                # Select the right entry
                this_out = out["backbone"] if key == "backbone" else out[key]

                # Quantiles: [Q_pred, Q*B, H] -> view as [Q_pred, Q_scen, B, H]
                if torch.is_tensor(this_out["quantiles"]):
                    q_pred_full = this_out["quantiles"][:, :, :chunk_len].detach().cpu()  # [Q, Q*B, chunk_len]
                else:
                    q_pred_full = torch.tensor(this_out["quantiles"][:, :, :chunk_len])
                q_pred = q_pred_full.reshape(Q, Q, B, chunk_len)                      # [Q_pred, Q_scen, B, chunk_len]

                # 4) Aggregate q^2 -> q quantiles
                q_agg = aggregate_q2_to_q(q_pred)  # [Q, B, chunk_len]

                # Mean aggregation: average the per-scenario means
                # this_out["mean"]: [Q*B, H] -> [Q, B, H] -> slice [:chunk_len] -> [Q, B, chunk_len]
                if torch.is_tensor(this_out["mean"]):
                    means_full = this_out["mean"][:, :chunk_len].detach().cpu().reshape(Q, B, chunk_len)
                else:
                    means_full = torch.tensor(this_out["mean"][:, :chunk_len]).reshape(Q, B, chunk_len)
                mean_agg = means_full.mean(dim=0)  # [B, chunk_len]

                # Store chunk
                ar_quant_chunks[key].append(q_agg)
                ar_mean_chunks[key].append(mean_agg)

                # 5) Update each scenario context by appending the aggregated quantile for that scenario (k)
                # Slide by chunk_len to keep context length constant.
                new_q_contexts = []
                for k in range(Q):
                    prev_ctx_k = q_contexts[k]  # [B, context_len]
                    ctx_k_next = torch.cat([prev_ctx_k[:, chunk_len:], q_agg[k].to(prev_ctx_k.dtype)], dim=1)
                    new_q_contexts.append(ctx_k_next)
                contexts_per_key[key] = torch.stack(new_q_contexts, dim=0)  # [Q, B, context_len]

            generated += chunk_len
            del out, this_out, q_pred_full, means_full
            torch.cuda.empty_cache()

        # 6) Collate chunks into final tensors for each key
        ar_results: Dict[str, Dict[str, Tensor]] = {}
        for key in run_keys:
            # Concatenate along the time dimension
            q_all = torch.cat(ar_quant_chunks[key], dim=2)  # [Q, B, pred_len]
            m_all = torch.cat(ar_mean_chunks[key], dim=1)   # [B, pred_len]
            ar_results[key] = {
                "mean": m_all,                 # [B, pred_len]
                "median": q_all[median_idx],   # [B, pred_len]
                "quantiles": q_all,            # [Q, B, pred_len]
            }

        return ar_results