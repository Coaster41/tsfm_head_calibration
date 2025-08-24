# cache_timesfm_features.py
import os
import math
import json
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
from torch.utils.data import DataLoader
from utils.data_loader import create_cached_tsmixup_datasets

import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
mp.set_sharing_strategy("file_system")

# Your snippet (works with stats [B, 2])
@torch.no_grad()
def process_transformer_output(stats: torch.Tensor, backbone_output: torch.Tensor, head: torch.nn.Module, output_patch_len, output_dim):
    output_ts = head(backbone_output)
    b, n, _ = output_ts.shape
    output_ts = output_ts.view(b, n, output_patch_len, output_dim)
    mu = stats[..., 0]   # [B]
    sigma = stats[..., 1]# [B]
    output_ts = output_ts * sigma[:, None, None, None] + mu[:, None, None, None]
    return output_ts[:, -1]

def default_collate_context(batch: List[Dict[str, Any]]):
    # Return the TimesFM-required context list and stacked labels if possible.
    context = [item["past_values"] for item in batch]
    try:
        labels = torch.stack([torch.as_tensor(item["future_values"]) for item in batch], dim=0)  # [B, H]
    except Exception:
        labels = [item["future_values"] for item in batch]
    return context, labels

def moirai_collate_context(batch: List[Dict[str, Any]]):
    # Return the Moirai-required context list and stacked labels if possible.
    context = torch.stack([item["past_values"] for item in batch], dim=0)[:, :, None]
    try:
        labels = torch.stack([torch.as_tensor(item["future_values"]) for item in batch], dim=0)  # [B, H]
    except Exception:
        labels = [item["future_values"] for item in batch]
    return context, labels

def chronos_bolt_collate_context(batch: List[Dict[str, Any]]):
    # Return the Chronos_bolt-required context list and stacked labels if possible.
    context = torch.stack([item["past_values"] for item in batch], dim=0)
    try:
        labels = torch.stack([torch.as_tensor(item["future_values"]) for item in batch], dim=0)  # [B, H]
    except Exception:
        labels = [item["future_values"] for item in batch]
    return context, labels

def sizeof(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

def target_samples_per_shard_full(N: int, latent_dim: int, latent_dtype: torch.dtype,
                                  include_stats: bool, label_numel: Optional[int],
                                  label_dtype: torch.dtype, target_bytes: int) -> int:
    bps = N * latent_dim * sizeof(latent_dtype)   # [N, D]
    if include_stats:
        bps += 2 * sizeof(torch.float32)          # stats [2] per sample
    if label_numel is not None:
        bps += label_numel * sizeof(label_dtype)
    return max(1, target_bytes // bps)

def target_samples_per_shard_pooled(latent_dim: int, latent_dtype: torch.dtype,
                                    include_stats: bool, label_numel: Optional[int],
                                    label_dtype: torch.dtype, target_bytes: int) -> int:
    bps = latent_dim * sizeof(latent_dtype)       # [D]
    if include_stats:
        bps += 2 * sizeof(torch.float32)          # stats [2]
    if label_numel is not None:
        bps += label_numel * sizeof(label_dtype)
    return max(1, target_bytes // bps)

def _concat_up_to(bufs: List[torch.Tensor], k: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    out, taken = [], 0
    new_bufs: List[torch.Tensor] = []
    for t in bufs:
        if taken >= k:
            new_bufs.append(t)
            continue
        need = k - taken
        if t.shape[0] <= need:
            out.append(t); taken += t.shape[0]
        else:
            out.append(t[:need]); new_bufs.append(t[need:]); taken += need
    return torch.cat(out, dim=0), new_bufs

@torch.no_grad()
def cache_timesfm_features(
    model,                      # TimesFM-like model
    dataset,                    # torch Dataset yielding dicts with keys "past_values" and "future_values"
    out_dir: str,
    mode: str = "full",         # "full" ([B, N, 1280]) or "pooled" ([B, 1280])
    batch_size: int = 512,
    num_workers: int = 8,
    pin_memory: bool = True,
    target_shard_bytes: int = 2 * 1024**3,   # ~2 GB shards
    verify_equivalence: bool = True,         # only meaningful for "full"
    output_patch_len: int = 128,
    output_dim: int = 10,
    latent_dim: int = 1280,
    to_dtype: torch.dtype = torch.float16,
    device: Optional[str] = "cuda:0",
    model_name: Optional[str] = "timesfm",
    inferred_N: Optional[int] = None
):
    os.makedirs(out_dir, exist_ok=True)
    if model_name == "timesfm":
        model._model.eval()
        if device is not None:
            model._model.to(device)
    elif "moirai" in model_name:
        moirai = model.module
        moirai.get_reprs = True
        moirai.eval()
        if device is not None:
            moirai.to(device)
    elif model_name == "chronos_bolt":
        model.model.eval()
    elif model_name == "yinglong":
        if device is not None:
            model.to(device)
        model.eval()

    print('start of loader')
    collate_dict = {
        "timesfm": default_collate_context,
        "chronos_bolt": chronos_bolt_collate_context,
        "moirai": moirai_collate_context,
        "moirai2": moirai_collate_context,
        "tirex": chronos_bolt_collate_context,
        "yinglong": chronos_bolt_collate_context,
        }
    collate_fn = collate_dict[model_name]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,          # try 2–4 for caching; you may not need many
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=1 if num_workers > 0 else None,                # keep small
        multiprocessing_context=mp.get_context("spawn") if num_workers >0 else None,
        collate_fn=collate_fn,
    )
    print('end of loader')

    # Buffers
    buf_latents: List[torch.Tensor] = []
    buf_stats: List[torch.Tensor] = []
    buf_labels: List[torch.Tensor] = []
    label_shape: Optional[torch.Size] = None
    shard_idx = 0
    total_written = 0
    # inferred_N: Optional[int] = None

    for step, (context, labels) in enumerate(loader):
        # print(f"context: {len(context)}, {context[0].shape}, labels: {labels.shape}")
        if step % 10 == 0:
            print(f"Current step: {step}")
        if model_name == "timesfm":
            freq = [0] * len(context)
            _, quantile_forecasts, (transformer_output, stats) = model.forecast(
                context, freq=freq, get_stacked_transformer=True
            )  # transformer_output: [B, N, 1280], stats: [B, 2]
        elif "moirai" in model_name:
            context = context.to(device)
            target, observed_mask, sample_id, time_id, variate_id, prediction_mask = model._convert(
                patch_size=output_patch_len,
                past_target=context,                                 # B x past_time x D
                past_observed_target=torch.isfinite(context),               # B x past_time x D (bool)
                past_is_pad=torch.full_like(context[:, :, 0], False, dtype=bool),                                 # B x past_time (bool)
            )
            if model_name == "moirai2":
                patch_sizes = False
            else:
                patch_sizes = torch.ones_like(time_id, dtype=torch.long) * output_patch_len
            transformer_output, stats = moirai(
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
                patch_sizes)
        elif model_name == "chronos_bolt":
            transformer_output = torch.zeros(context.shape[0], model.model.model_dim)
            stats = torch.zeros(context.shape[0], 2)            
            def save_decoder_hook(module, input, output):
                transformer_output[:] = output.last_hidden_state.squeeze().detach().cpu()
                
            def save_encoder_hook(module, input, output):
                stats[:] = torch.stack(output[1], dim=-1).squeeze().detach().cpu()

            model.model.decoder.register_forward_hook(save_decoder_hook)
            model.model.instance_norm.register_forward_hook(save_encoder_hook)
            context = context.to(device)
            _ = model.model(context)
        elif model_name == "tirex":
            d_model = model.model_config.block_kwargs.embedding_dim
            patch_size = model.model_config.output_patch_size
            out_patches = pred_len // patch_size
            transformer_output = torch.zeros(context.shape[0], out_patches, d_model)
            stats = torch.zeros((context.shape[0], 2))
            forecast = model.forecast(context=context, prediction_length=pred_len, batch_size=batch_size,
                                                max_accelerated_rollout_steps=4, 
                                                get_loc_scale=stats,
                                                get_hidden_states=transformer_output)
        elif model_name == "yinglong":
            context = context.to(device)
            d_model = model.config.n_embd # 768
            patch_size = model.config.patch_size # 32
            out_patches = pred_len // patch_size
            transformer_output = torch.zeros((context.shape[0], out_patches, d_model))
            stats = torch.zeros((batch_size, 2))
            start_ind = context.shape[0] // patch_size
            end_ind = start_ind + out_patches
            def save_decoder_hook(module, input, output):
                transformer_output[:] = input[0][:,start_ind:end_ind].detach().cpu()
            def save_tokenizer_hook(module, input, output):
                x, x_raw, masks, mean, std, _ = output
                stats[:,0] = mean.flatten().detach().cpu()
                stats[:,1] = std.flatten().detach().cpu()
            model.lm_head.register_forward_hook(save_decoder_hook)
            model.tokenizer.register_forward_hook(save_tokenizer_hook)
            forecast_len = 2048 # large number for DCoT
            forecast = model.generate(context, future_token=forecast_len)
            


        # print("post forecast")

        if inferred_N is None:
            inferred_N = transformer_output.shape[1]
            print(f"[cache] Inferred N={inferred_N}, latent_dim={latent_dim}")

        if verify_equivalence and mode == "full":
            computed_quantiles = process_transformer_output(
                stats, transformer_output, model._model.horizon_ff_layer, output_patch_len, output_dim
            )
            if not torch.allclose(computed_quantiles, quantile_forecasts, atol=1e-5, rtol=1e-4):
                raise RuntimeError("computed_quantiles != quantile_forecasts (sanity check failed)")

        # Prepare latents
        if mode == "full":
            lat = transformer_output.to(dtype=to_dtype).cpu().contiguous()  # [B, N, 1280]
        elif mode == "pooled":
            lat = transformer_output.mean(dim=1).to(dtype=to_dtype).cpu().contiguous()  # [B, 1280]
        elif mode == "last":
            lat = transformer_output[:,-1,:].to(dtype=to_dtype).cpu().contiguous()  # [B, 1280]
        elif mode == 'pred_mask':
            lat = transformer_output[prediction_mask].reshape([batch_size, -1, transformer_output.shape[-1]])
            lat = lat.to(dtype=to_dtype).cpu().contiguous()  # [B, D]
            stats = stats[prediction_mask].reshape([batch_size, -1, stats.shape[-1]])  # [B, D]
        elif mode == "last_ctx":
            lat = transformer_output[:,context_len//output_patch_len-1,:].to(dtype=to_dtype).cpu().contiguous()  # [B, D]
        elif mode == "squeeze":
            lat = transformer_output.to(dtype=to_dtype).contiguous()  # [B, N, 1280]
        else:
            raise ValueError("mode must be 'full' or 'pooled' or 'last' or 'pred_mask' or 'last_ctx' or 'squeeze")

        st = stats.float().cpu().contiguous()  # [B, 2]

        # Labels (expect [B, H])
        if isinstance(labels, list):
            try:
                labels = torch.stack([torch.as_tensor(y) for y in labels], dim=0)
            except Exception:
                labels = None
        if isinstance(labels, torch.Tensor):
            if label_shape is None:
                label_shape = labels.shape[1:]
            lbl = labels.cpu().contiguous()
        else:
            lbl = None

        buf_latents.append(lat)
        buf_stats.append(st)
        if lbl is not None:
            buf_labels.append(lbl)

        # Compute target shard size (samples)
        label_numel = int(math.prod(label_shape)) if label_shape is not None else None
        label_dtype = labels.dtype if isinstance(labels, torch.Tensor) else torch.float32
        if mode == "full" or mode == 'pred_mask':
            spp = target_samples_per_shard_full(
                N=inferred_N, latent_dim=latent_dim, latent_dtype=to_dtype,
                include_stats=True, label_numel=label_numel, label_dtype=label_dtype,
                target_bytes=target_shard_bytes
            )
        else:
            # print('before target samples per shard pooled')
            spp = target_samples_per_shard_pooled(
                latent_dim=latent_dim, latent_dtype=to_dtype,
                include_stats=True, label_numel=label_numel, label_dtype=label_dtype,
                target_bytes=target_shard_bytes
            )
            # print('after target samples per shard pooled')

        # Flush shards while enough samples buffered
        def buffered_count(lst: List[torch.Tensor]) -> int:
            return sum(t.shape[0] for t in lst) if lst else 0

        while buffered_count(buf_latents) >= spp:
            # print(f"buffered_count: {buffered_count(buf_latents)}")
            # concat up to spp for each buffer
            lat_cat, buf_latents[:] = _concat_up_to(buf_latents, spp)
            st_cat,  buf_stats[:]   = _concat_up_to(buf_stats, spp)
            if buf_labels:
                lbl_cat, buf_labels[:] = _concat_up_to(buf_labels, spp)
            else:
                lbl_cat = None

            shard_path = os.path.join(out_dir, f"{mode}_shard_{shard_idx:06d}.pt")
            meta = {
                "mode": mode,
                "latent_dim": latent_dim,
                "num_tokens": inferred_N,
                "latent_dtype": str(to_dtype).replace("torch.", ""),
                "stats_shape": [2],  # stats is [B, 2]
                "num_samples": lat_cat.shape[0],
                "label_shape": list(label_shape) if label_shape is not None else None,
            }
            torch.save({"latents": lat_cat, "stats": st_cat, "labels": lbl_cat, "meta": meta}, shard_path)
            total_written += lat_cat.shape[0]
            print(f"[cache] wrote {shard_path} ({lat_cat.shape[0]} samples). total={total_written}")
            shard_idx += 1

    # Final flush
    if buf_latents:
        lat_cat = torch.cat(buf_latents, dim=0)
        st_cat  = torch.cat(buf_stats, dim=0)
        lbl_cat = torch.cat(buf_labels, dim=0) if buf_labels else None
        shard_path = os.path.join(out_dir, f"{mode}_shard_{shard_idx:06d}.pt")
        meta = {
            "mode": mode,
            "latent_dim": latent_dim,
            "num_tokens": inferred_N,
            "latent_dtype": str(to_dtype).replace("torch.", ""),
            "stats_shape": [2],
            "num_samples": lat_cat.shape[0],
            "label_shape": list(label_shape) if label_shape is not None else None,
        }
        torch.save({"latents": lat_cat, "stats": st_cat, "labels": lbl_cat, "meta": meta}, shard_path)
        total_written += lat_cat.shape[0]
        print(f"[cache] wrote final {shard_path} ({lat_cat.shape[0]} samples). total={total_written}")


if __name__ == "__main__":

    context_len = 512
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    # Select the GPU device based on the local rank
    # device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}") 
    device = f"cuda:{local_rank % torch.cuda.device_count()}"

    model_name = "yinglong"
    if model_name == "timesfm":
        import timesfm
        pred_len = 128
        model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend='gpu',
                    # per_core_batch_size=32,
                    context_len=context_len,  # currently max supported
                    horizon_len=pred_len,  # number of steps to predict
                    input_patch_len=32,  # fixed parameters
                    output_patch_len=128,
                    num_layers=50,
                    model_dims=1280,
                    use_positional_embedding=False,
                    point_forecast_mode='mean',
                    device=device
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
            )
    elif model_name == "moirai":
        from uni2ts.model.moirai import (MoiraiForecast, MoiraiModule)
        pred_len = 128
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-small"),
            prediction_length=pred_len,
            context_length=context_len,
            patch_size=32,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    elif model_name == "moirai2":
        from uni2ts.model.moirai2 import (Moirai2Forecast, Moirai2Module)
        pred_len = 64
        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(
                f"Salesforce/moirai-2.0-R-small",
            ),
            prediction_length=pred_len,
            context_length=context_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    elif model_name == "chronos_bolt":
        from chronos import ChronosBoltPipeline
        pred_len = 64
        model = ChronosBoltPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    elif model_name == "tirex":
        from tirex import load_model as load_tirex_model
        pred_len = 128
        model = load_tirex_model("NX-AI/TiRex")
    elif model_name == "yinglong":
        from transformers import AutoModelForCausalLM
        pred_len = 128
        model = AutoModelForCausalLM.from_pretrained('qcw2333/YingLong_110m', trust_remote_code=True,torch_dtype=torch.bfloat16)

        

    train_dataset, val_dataset = create_cached_tsmixup_datasets(
        max_samples=None,
        context_length=context_len,
        prediction_length=pred_len, # 1 or 96
        num_workers=0,
        cache_dir="/extra/datalab_scratch0/ctadler/time_series_models/mechanistic_interpretability/data/tsmixup_cache/",
        processed_cache_path="/extra/datalab_scratch0/ctadler/time_series_models/mechanistic_interpretability/data/tsmixup_cache/tsmixup_processed_None_1024_128.pkl",
        batch_size=4000
    )

    # 1) Full token-level cache (~200GB fp16, sharded)
    # cache_timesfm_features(
    #     model=model,
    #     dataset=train_dataset,
    #     out_dir="./data/timesfm_cache_full_fp16",
    #     mode="full",
    #     batch_size=512,
    #     num_workers=0,
    #     target_shard_bytes=2 * 1024**3,  # ~2 GB shards
    #     verify_equivalence=False,         # ensures process_transformer_output == quantile_forecasts
    #     output_patch_len=128,
    #     output_dim=10,
    #     latent_dim=1280,
    #     to_dtype=torch.float16,
    #     device=device,
    #     pin_memory=False,
    # )

    # 2) Mean-pooled cache (~12–14GB fp16, sharded)
    # cache_timesfm_features(
    #     model=model,
    #     dataset=train_dataset,
    #     out_dir="./data/cache_pooled_fp16",
    #     mode="pooled",
    #     batch_size=2048//4,                 # can be larger since pooled is tiny
    #     num_workers=0,
    #     target_shard_bytes=int(0.5 * 1024**3),  # ~1 GB shards (optional)
    #     verify_equivalence=False,        # pooled won’t match quantile_forecasts
    #     latent_dim=1280,
    #     to_dtype=torch.float16,
    #     device=device,c
    # )

    # # 3) Last patch cache
    # cache_timesfm_features(
    #     model=model,
    #     dataset=train_dataset,
    #     out_dir="./data/timesfm_cache_last_fp16",
    #     mode="last",
    #     batch_size=512,
    #     num_workers=0,
    #     target_shard_bytes=500 * 1024**2,  # ~500 MB shards
    #     verify_equivalence=False,         # ensures process_transformer_output == quantile_forecasts
    #     output_patch_len=128,
    #     output_dim=10,
    #     latent_dim=1280,
    #     to_dtype=torch.float16,
    #     device=device,
    #     pin_memory=False,
    # )
    
    # 4) MOIRAI pred_mask patch cache
    # cache_timesfm_features(
    #     model=model,
    #     dataset=train_dataset,
    #     out_dir="./data/moirai_cache_fp16",
    #     mode="last_ctx",
    #     batch_size=512,
    #     num_workers=0,
    #     target_shard_bytes=500 * 1024**2,  # ~500 MB shards
    #     verify_equivalence=False,         # ensures process_transformer_output == quantile_forecasts
    #     output_patch_len=32,
    #     output_dim=10,
    #     latent_dim=384,
    #     to_dtype=torch.float16,
    #     device=device,
    #     pin_memory=False,
    #     model_name=model_name
    # )

    # # 5) MOIRAI2 last_ctx patch cache
    # cache_timesfm_features(
    #     model=model,
    #     dataset=train_dataset,
    #     out_dir="./data/moirai2_cache_fp16",
    #     mode="last_ctx",
    #     batch_size=512,
    #     num_workers=0,
    #     target_shard_bytes=500 * 1024**2,  # ~500 MB shards
    #     verify_equivalence=False,         # ensures process_transformer_output == quantile_forecasts
    #     output_patch_len=16,
    #     output_dim=10,
    #     latent_dim=384,
    #     to_dtype=torch.float16,
    #     device=device,
    #     pin_memory=False,
    #     model_name=model_name
    # )

    # 5) chronos_bolt squueze patch cache
    # cache_timesfm_features(
    #     model=model,
    #     dataset=train_dataset,
    #     out_dir="./data/chronos_bolt_cache_fp16",
    #     mode="squeeze",
    #     batch_size=512,
    #     num_workers=0,
    #     target_shard_bytes=500 * 1024**2,  # ~500 MB shards
    #     verify_equivalence=False,         # ensures process_transformer_output == quantile_forecasts
    #     output_patch_len=16,
    #     output_dim=10,
    #     latent_dim=384,
    #     to_dtype=torch.float16,
    #     device=device,
    #     pin_memory=False,
    #     model_name=model_name
    # )

    # # 5) Tirex full patch cache
    # cache_timesfm_features(
    #     model=model,
    #     dataset=train_dataset,
    #     out_dir="./data/tirex_cache_fp16",
    #     mode="full",
    #     batch_size=512,
    #     num_workers=0,
    #     target_shard_bytes=500 * 1024**2,  # ~500 MB shards
    #     verify_equivalence=False,         # ensures process_transformer_output == quantile_forecasts
    #     output_patch_len=32,
    #     output_dim=10,
    #     latent_dim=512,
    #     to_dtype=torch.float16,
    #     device=device,
    #     pin_memory=False,
    #     model_name=model_name,
    #     inferred_N=4,
    # )

    # 6) YingLong full patch cache
    cache_timesfm_features(
        model=model,
        dataset=train_dataset,
        out_dir="./data/yinglong_cache_fp16",
        mode="full",
        batch_size=512,
        num_workers=0,
        target_shard_bytes=500 * 1024**2,  # ~500 MB shards
        verify_equivalence=False,         # ensures process_transformer_output == quantile_forecasts
        output_patch_len=32,
        output_dim=10,
        latent_dim=768,
        to_dtype=torch.float16,
        device=device,
        pin_memory=False,
        model_name=model_name,
        inferred_N=4,
    )