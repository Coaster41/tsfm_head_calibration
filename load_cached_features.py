# load_cached_features.py
import glob
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import os
import psutil
import time

# Get the current process
process = psutil.Process(os.getpid())
def get_mem_usage():
    # Get memory info (including RSS)
    memory_info = process.memory_info()
    # RSS is in bytes, convert to MB for readability
    rss_mb = memory_info.rss / (1024 * 1024)
    print(f"Current memory usage (RSS): {rss_mb:.2f} MB")

class _ShardDS(Dataset):
    def __init__(self, shards_glob: str):
        self.files = sorted(glob.glob(shards_glob))
        assert self.files, f"No shards match: {shards_glob}"
        self.index: List[Tuple[int, int]] = []
        self.meta: List[Dict[str, Any]] = []
        self._cache: Dict[str, Any] = {}
        self._cache_idx: Optional[int] = None
        for fi, f in enumerate(self.files):
            # print(f"current file: {fi}, glob: {f}")
            # get_mem_usage()
            d = torch.load(f, map_location="cpu")
            n = d["meta"]["num_samples"]
            self.meta.append(d["meta"])
            # print(f"Meta: {d['meta']}")
            self.index += [(fi, j) for j in range(n)]

    def _load_file(self, fi: int):
        if self._cache_idx != fi:
            self._cache = torch.load(self.files[fi], map_location="cpu")
            self._cache_idx = fi

    def __len__(self):
        return len(self.index)

class FullLatentShardDataset(_ShardDS):
    # Yields: latents [N, 1280] (fp16), stats [2] (fp32), label [H] (if present)
    def __getitem__(self, i: int):
        # print(f'get_item {i}')
        # get_mem_usage()
        fi, off = self.index[i]
        self._load_file(fi)
        latents = self._cache["latents"][off].clone()  # [N, 1280], fp16
        stats   = self._cache["stats"][off].clone()    # [2], fp32
        labels  = self._cache.get("labels", None)
        label   = labels[off].clone() if isinstance(labels, torch.Tensor) else None
        return latents, stats, label

class PooledLatentShardDataset(_ShardDS):
    # Yields: pooled [1280] (fp16), stats [2] (fp32), label [H] (if present)
    def __getitem__(self, i: int):
        fi, off = self.index[i]
        self._load_file(fi)
        pooled = self._cache["latents"][off]   # [1280], fp16
        stats  = self._cache["stats"][off]     # [2], fp32
        labels = self._cache.get("labels", None)
        label  = labels[off] if isinstance(labels, torch.Tensor) else None
        return pooled, stats, label

@torch.no_grad()
def reconstruct_quantiles_from_full(
    latents_b: torch.Tensor,   # [B, N, 1280] fp16/32
    stats_b: torch.Tensor,     # [B, 2] fp32
    head: torch.nn.Module,
    output_patch_len: int,
    output_dim: int,
    device: str = "cuda:0",
) -> torch.Tensor:
    lat = latents_b.to(device)           # keep fp16 OK
    st  = stats_b.to(device).float()     # [B, 2]
    with torch.amp.autocast(device_type=device, dtype=torch.float16):
        # Uses your snippet with stats [B, 2]
        out = head(lat)                  # [B, N, output_patch_len * output_dim]
        b, n, _ = out.shape
        out = out.view(b, n, output_patch_len, output_dim)
        mu    = st[:, 0]                 # [B]
        sigma = st[:, 1]                 # [B]
        out = out * sigma[:, None, None, None] + mu[:, None, None, None]
        return out[:, -1]                # [B, output_patch_len, output_dim]
    
if __name__ == "__main__":
    # Full latents (token-level)
    import timesfm

    pred_len = 96
    context_len = 512
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    # Select the GPU device based on the local rank
    # device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}") 
    device = f"cuda:{local_rank % torch.cuda.device_count()}"

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

    print("Before Dataset")
    full_ds = FullLatentShardDataset("./data/timesfm_cache_full_fp16/full_shard_*.pt")
    print("After dataset setup")
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    print("Full Loader Setup Completed")


    # Sanity check: reconstruct quantiles from a batch of full latents
    start_time = time.time()
    for lat, st, _ in full_loader:
        print(f"first iteration: {lat.shape}, {st.shape}, {time.time()-start_time:.2f}")
        q = reconstruct_quantiles_from_full(lat, st, model._model.horizon_ff_layer, 128, 10, device=device)
        print(f"Completed recontruct_quantiles: {q.shape}")
        break

    # Pooled latents
    # pooled_ds = PooledLatentShardDataset("./data/timesfm_cache_pooled_fp16/pooled_shard_*.pt")
    # pooled_loader = DataLoader(pooled_ds, batch_size=4096, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)