import torch
import datasets
import numpy as np
import pickle

import multiprocessing as mp
import os
from torch.utils.data import Dataset
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc


from tqdm import tqdm
import time

class CachedTSMixupLoader:
    """
    Efficient loader for TSMixup dataset using cached data with parallel processing
    """
    def __init__(self, context_length=512, prediction_length=1, 
                 patch_length=16, patch_stride=16, num_workers=None,
                 cache_dir=None):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_workers = num_workers or mp.cpu_count()
        self.cache_dir = cache_dir
        
        # print(f"🔧 Initialized TSMixup loader:")
        # print(f"  Context length: {context_length}")
        # print(f"  Prediction length: {prediction_length}")
        # print(f"  Workers: {self.num_workers}")
        # print(f"  Cache dir: {cache_dir}")

    def load_cached_dataset(self, max_samples=None, force_redownload=False):
        """
        Load the TSMixup dataset from cache (streaming=False)
        """
        # print("📥 Loading TSMixup dataset from cache...")
        start_time = time.time()
        
        # Load with streaming=False for better performance
        dataset = datasets.load_dataset(
            "autogluon/chronos_datasets", 
            "training_corpus_tsmixup_10m", 
            streaming=False,  # Cache the entire dataset
            split="train",
            cache_dir=self.cache_dir,
        )
        
        load_time = time.time() - start_time
        # print(f"✅ Dataset loaded in {load_time:.2f} seconds")
        # print(f"📊 Total samples in dataset: {len(dataset):,}")
        
        if max_samples:
            total_samples = min(len(dataset), max_samples)
            # print(f"🎯 Will process {total_samples:,} samples")
        else:
            total_samples = len(dataset)
        
        return dataset, total_samples

    @staticmethod
    def process_sample_batch(batch_info):
        """
        Process a batch of samples (static method for multiprocessing)
        Args:
            batch_info: tuple of (batch_data, context_length, prediction_length, start_idx)
        Returns:
            list of (original_idx, processed_data) tuples
        """
        batch_data, context_length, prediction_length, start_idx = batch_info
        processed_batch = []
        
        for local_idx, sample in enumerate(batch_data):
            original_idx = start_idx + local_idx
            
            try:
                # Extract time series data
                ts_data = CachedTSMixupLoader._extract_time_series(sample)
                if ts_data is None:
                    continue
                
                # Preprocess time series
                processed_ts = CachedTSMixupLoader._preprocess_time_series(
                    ts_data, context_length, prediction_length
                )
                
                if processed_ts is not None:
                    processed_batch.append((original_idx, processed_ts))
                    
            except Exception as e:
                print(f"Error processing sample {original_idx}: {e}")
                continue
        
        return processed_batch

    @staticmethod
    def _extract_time_series(sample):
        """Extract time series data from a sample"""
        try:
            if 'target' in sample:
                data = np.array(sample['target'], dtype=np.float32)
                return data
            elif 'values' in sample:
                data = np.array(sample['values'], dtype=np.float32)
                return data
            else:
                # Find suitable numeric array
                for key, value in sample.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 100:
                        data = np.array(value, dtype=np.float32)
                        return data
            return None
        except Exception:
            return None

    @staticmethod
    def _preprocess_time_series(ts_data, context_length, prediction_length):
        """Preprocess time series data"""
        try:
            required_length = context_length + prediction_length
            
            # Length check
            if len(ts_data) < required_length:
                if len(ts_data) < context_length:
                    return None
                # Pad if reasonable
                padding_needed = required_length - len(ts_data)
                if padding_needed <= prediction_length // 2:
                    ts_data = np.pad(ts_data, (0, padding_needed), mode='edge')
                else:
                    return None
            
            # Quality checks
            if (np.any(np.isnan(ts_data)) or 
                np.any(np.isinf(ts_data)) or
                np.var(ts_data) == 0 or
                np.abs(ts_data).max() > 1e10):
                return None
            
            return ts_data.astype(np.float32)
            
        except Exception:
            return None

    def process_dataset_parallel(self, dataset, total_samples, batch_size=2000, 
                                save_path=None):
        """
        Process the cached dataset using multiple workers
        """

        def parallel_process(batches):
            # Process batches in parallel
            all_processed_data = []
            failed_count = 0
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all batch processing tasks
                future_to_batch = {
                    executor.submit(self.process_sample_batch, batch): batch_idx 
                    for batch_idx, batch in enumerate(batches)
                }
                
                # Collect results with progress tracking
                for future in tqdm(as_completed(future_to_batch), 
                                total=len(future_to_batch),
                                desc="Processing batches"):
                    try:
                        batch_results = future.result(timeout=120)  # 2 minute timeout
                        
                        for original_idx, processed_data in batch_results:
                            all_processed_data.append((original_idx, processed_data))
                        
                        # Calculate failed samples in this batch
                        batch_idx = future_to_batch[future]
                        expected_samples = len(batches[batch_idx][0])  # batch_data length
                        actual_samples = len(batch_results)
                        failed_count += (expected_samples - actual_samples)
                        
                    except Exception as e:
                        print(f"❌ Batch processing failed: {e}")
                        batch_idx = future_to_batch[future]
                        failed_count += len(batches[batch_idx][0])
            # current_memory = psutil.Process().memory_info().rss
            # print(f"before deletion: {current_memory / 1e9}")
            # del(batches)
            # gc.collect()
            # current_memory = psutil.Process().memory_info().rss
            # print(f"after deletion: {current_memory / 1e9}")
            # batches = []
            return all_processed_data, failed_count

        # print(f"🔄 Processing {total_samples:,} samples with {self.num_workers} workers...")
        # print(f"📦 Batch size: {batch_size}")
        
        start_time = time.time()
        
        # Create batches for parallel processing
        batches = []
        all_processed_data =  []
        failed_count = 0
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            current_memory = psutil.Process().memory_info().rss
            # print(f"{i}: {current_memory / 1e9}")
            
            # Extract batch data from cached dataset (fast random access)
            batch_data = [dataset[j] for j in range(i, end_idx)]
            
            batch_info = (
                batch_data, 
                self.context_length, 
                self.prediction_length, 
                i  # start_idx for original indexing
            )
            batches.append(batch_info)

            if ((i // batch_size + 1) % (self.num_workers)) == 0:
                # print(f"processing batches {len(batches)}")
                processed_batches, failed_num = parallel_process(batches)
                failed_count += failed_num
                all_processed_data.extend(processed_batches)
                del batches
                gc.collect()
                batches = []
                gc.collect()

        processed_batches, failed_num = parallel_process(batches)
        failed_count += failed_num        
        all_processed_data.extend(processed_batches)
        
        # print(f"📊 Created {len(batches)} batches for processing")
        
        
        
        # Sort by original index to maintain order
        all_processed_data.sort(key=lambda x: x[0])
        final_data = [data for _, data in all_processed_data]
        
        processing_time = time.time() - start_time
        
        # Print statistics
        # print(f"\n🎉 PROCESSING COMPLETE:")
        # print(f"  ✅ Successfully processed: {len(final_data):,} samples")
        # print(f"  ❌ Failed samples: {failed_count:,}")
        # print(f"  📈 Success rate: {len(final_data)/(len(final_data)+failed_count)*100:.1f}%")
        # print(f"  ⏱️  Processing time: {processing_time:.2f} seconds")
        # print(f"  🚀 Processing speed: {len(final_data)/processing_time:.1f} samples/second")
        
        # Save processed data if requested
        if save_path:
            self._save_processed_data(final_data, save_path, processing_time)
        
        return final_data
    
    def _save_processed_data(self, data, save_path, processing_time):
        """Save processed data to disk"""
        # print(f"💾 Saving processed data to {save_path}...")
        
        save_data = {
            'data': data,
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'patch_length': self.patch_length,
            'patch_stride': self.patch_stride,
            'processing_time': processing_time,
            'num_samples': len(data),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        # print(f"✅ Saved! File size: {file_size:.1f} MB")


class CachedTSMixupDataset(Dataset):
    """
    Dataset class for preprocessed TSMixup data loaded from cache
    """
    def __init__(self, preprocessed_data, context_length, prediction_length,
                 patch_length=16, patch_stride=16, augmentation=True):
        self.data = preprocessed_data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.augmentation = augmentation
        
        # print(f"🏗️  Dataset created with {len(self.data):,} samples")
        # print(f"📊 Augmentation: {'ON' if augmentation else 'OFF'}")
        
        if len(self.data) > 0:
            self._compute_statistics()

    def _compute_statistics(self):
        """Compute and display dataset statistics"""
        # Sample data for statistics (to avoid loading everything)
        sample_size = min(1000, len(self.data))
        sample_data = self.data[:sample_size]
        
        lengths = [len(ts) for ts in sample_data]
        all_values = np.concatenate(sample_data[:100])  # Even smaller sample for values
        
        # print(f"📈 Dataset Statistics (from {sample_size} samples):")
        # print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
        # print(f"  Value ranges: min={all_values.min():.4f}, max={all_values.max():.4f}")
        # print(f"  Value stats: mean={all_values.mean():.4f}, std={all_values.std():.4f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ts_data = self.data[idx]
        required_length = self.context_length + self.prediction_length
        
        # Create sliding window
        if len(ts_data) > required_length:
            if self.augmentation:
                # Random start position for training variability
                max_start = len(ts_data) - required_length
                start_idx = np.random.randint(0, max_start + 1)
            else:
                # Fixed start position for validation/testing
                start_idx = (len(ts_data) - required_length) // 2
            
            past_values = ts_data[start_idx:start_idx + self.context_length]
            future_values = ts_data[start_idx + self.context_length:start_idx + required_length]
        else:
            # Use all available data
            past_values = ts_data[:self.context_length]
            future_values = ts_data[self.context_length:self.context_length + self.prediction_length]
        
        # Ensure correct lengths
        assert len(past_values) == self.context_length, f"Past values length: {len(past_values)}"
        assert len(future_values) == self.prediction_length, f"Future values length: {len(future_values)}"
        
        return {
            'past_values': torch.tensor(past_values, dtype=torch.float32),
            'future_values': torch.tensor(future_values, dtype=torch.float32),
        }

    @classmethod
    def load_from_cache(cls, cache_path, augmentation=True):
        """Load preprocessed dataset from cache file"""
        # print(f"📂 Loading preprocessed dataset from {cache_path}...")
        
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # print(f"✅ Loaded cached data:")
        # print(f"  Samples: {cached_data['num_samples']:,}")
        # print(f"  Created: {cached_data.get('created_at', 'Unknown')}")
        # print(f"  Processing time: {cached_data.get('processing_time', 0):.2f}s")
        
        return cls(
            preprocessed_data=cached_data['data'],
            context_length=cached_data['context_length'],
            prediction_length=cached_data['prediction_length'],
            patch_length=cached_data.get('patch_length', 16),
            patch_stride=cached_data.get('patch_stride', 8),
            augmentation=augmentation
        )

def create_cached_tsmixup_datasets(max_samples=100000, context_length=512, 
                                 prediction_length=1, num_workers=None,
                                 train_val_split=0.9, cache_dir=None,
                                 processed_cache_path=None, batch_size=2000,
                                 force_reprocess=False):
    """
    Main function to create TSMixup datasets from cached data using multiple workers
    
    Args:
        max_samples: Maximum number of samples to process
        context_length: Length of input context
        prediction_length: Length of prediction horizon
        num_workers: Number of parallel workers (None = auto-detect)
        train_val_split: Train/validation split ratio
        cache_dir: Directory to cache the raw dataset
        processed_cache_path: Path to save/load preprocessed data
        batch_size: Batch size for parallel processing
        force_reprocess: Force reprocessing even if cache exists
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    
    # print("🚀 CREATING CACHED TSMIXUP DATASETS")
    # print("=" * 50)
    
    # Set default processed cache path
    if processed_cache_path is None:
        processed_cache_path = f"tsmixup_processed_{max_samples}_{context_length}_{prediction_length}.pkl"
    
    # Check if processed data already exists
    if os.path.exists(processed_cache_path) and not force_reprocess:
        # print(f"📂 Found existing processed data at {processed_cache_path}")
        try:
            # Load from existing processed cache
            # print("⚡ Loading preprocessed data from cache...")
            
            with open(processed_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            processed_data = cached_data['data']
            
            # print(f"✅ Loaded {len(processed_data):,} preprocessed samples")
            # print(f"📅 Cache created: {cached_data.get('created_at', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            print("🔄 Will reprocess data...")
            force_reprocess = True
    
    # Process data if needed
    if not os.path.exists(processed_cache_path) or force_reprocess:
        print("🔄 Processing raw dataset...")
        
        # Initialize loader
        loader = CachedTSMixupLoader(
            context_length=context_length,
            prediction_length=prediction_length,
            num_workers=num_workers,
            cache_dir=cache_dir
        )
        
        # Load cached dataset
        dataset, total_samples = loader.load_cached_dataset(
            max_samples=max_samples,
            force_redownload=False
        )
        
        # Process dataset with parallel workers
        processed_data = loader.process_dataset_parallel(
            dataset=dataset,
            total_samples=total_samples,
            batch_size=batch_size,
            save_path=processed_cache_path
        )
    else:
        # Data was loaded from cache above
        pass
    
    # Validate processed data
    if len(processed_data) == 0:
        raise ValueError("No valid samples found in dataset!")
    
    # print(f"\n📊 DATASET SUMMARY:")
    # print(f"  Total processed samples: {len(processed_data):,}")
    # print(f"  Context length: {context_length}")
    # print(f"  Prediction length: {prediction_length}")
    
    # Shuffle data for better train/val split
    # print("🔀 Shuffling data...")
    np.random.seed(42)  # For reproducible splits
    shuffled_indices = np.random.permutation(len(processed_data))
    shuffled_data = [processed_data[i] for i in shuffled_indices]
    
    # Split into train/validation
    split_point = int(len(shuffled_data) * train_val_split)
    train_data = shuffled_data[:split_point]
    val_data = shuffled_data[split_point:]
    
    # print(f"📈 Data split:")
    # print(f"  Training samples: {len(train_data):,}")
    # print(f"  Validation samples: {len(val_data):,}")
    # print(f"  Train ratio: {len(train_data)/len(shuffled_data)*100:.1f}%")
    
    # # Create dataset objects
    # print("🏗️  Creating PyTorch datasets...")
    
    train_dataset = CachedTSMixupDataset(
        preprocessed_data=train_data,
        context_length=context_length,
        prediction_length=prediction_length,
        augmentation=True  # Enable augmentation for training
    )
    
    val_dataset = CachedTSMixupDataset(
        preprocessed_data=val_data,
        context_length=context_length,
        prediction_length=prediction_length,
        augmentation=False  # Disable augmentation for validation
    )
    
    # print("\n🎉 DATASETS CREATED SUCCESSFULLY!")
    # print(f"✅ Train dataset: {len(train_dataset):,} samples")
    # print(f"✅ Validation dataset: {len(val_dataset):,} samples")
    
    # # Test datasets by loading a few samples
    # print("\n🧪 Testing dataset loading...")
    try:
        train_sample = train_dataset[0]
        val_sample = val_dataset[0]
        
        # print(f"✅ Sample shapes:")
        # print(f"  Train - Past: {train_sample['past_values'].shape}, Future: {train_sample['future_values'].shape}")
        # print(f"  Val   - Past: {val_sample['past_values'].shape}, Future: {val_sample['future_values'].shape}")
        
    except Exception as e:
        print(f"❌ Error testing datasets: {e}")
    
    return train_dataset, val_dataset