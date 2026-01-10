import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class VLAEmbeddingDataset(Dataset):
    def __init__(self, dataset_path, sequence_length=1):
        """
        Loads precomputed embeddings and actions.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        
        # Open file to read metadata (we will keep it open or reopen in __getitem__)
        # For HPC, it's safer to load indices in memory and read data on demand
        self.demos = []
        with h5py.File(dataset_path, "r") as f:
            demo_keys = list(f["data"].keys())
            # Store valid indices: (demo_key, start_index)
            # We iterate through demos to find all valid transitions
            for key in demo_keys:
                num_samples = f[f"data/{key}/obs_embedding"].shape[0]
                # We need at least one transition (t -> t+1)
                if num_samples > 1:
                    # We can take samples from 0 to num_samples - 2
                    # so that t+1 is at most num_samples - 1
                    for i in range(num_samples - 1):
                        self.demos.append((key, i))
                        
        print(f"🔹 Dataset loaded. Total transitions: {len(self.demos)}")

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        # Open file here to allow multiprocessing workers to have their own handles
        with h5py.File(self.dataset_path, "r") as f:
            demo_key, t = self.demos[idx]
            
            # Get Current State (z_t) and Action (a_t)
            z_t = f[f"data/{demo_key}/obs_embedding"][t]
            a_t = f[f"data/{demo_key}/actions"][t]
            
            # Get Next State (z_t+1) -> TARGET
            z_next = f[f"data/{demo_key}/obs_embedding"][t+1]
            
            # --- TEXT CONDITIONING (MOCKED FOR NOW) ---
            # In a full run, this would be a BERT/CLIP embedding of "Lift the cube"
            # Here we use a zero vector placeholder to ensure architecture compatibility
            text_emb = np.zeros(512, dtype=np.float32) 

        # Return as Torch Tensors
        return {
            "state": torch.from_numpy(z_t).float(),
            "action": torch.from_numpy(a_t).float(),
            "text": torch.from_numpy(text_emb).float(),
            "next_state": torch.from_numpy(z_next).float()
        }