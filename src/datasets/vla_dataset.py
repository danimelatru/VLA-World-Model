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
        
        self.demos = []
        self.has_text_embedding = False
        
        with h5py.File(dataset_path, "r") as f:
            demo_keys = list(f["data"].keys())
            
            # Check if text embeddings exist (support both old singular and new plural)
            first_key = demo_keys[0]
            if "text_embeddings" in f[f"data/{first_key}"]:
                self.text_mode = "multiple"
            elif "text_embedding" in f[f"data/{first_key}"]:
                self.text_mode = "single"
            else:
                self.text_mode = "none"
            
            for key in demo_keys:
                num_samples = f[f"data/{key}/obs_embedding"].shape[0]
                if num_samples > 1:
                    for i in range(num_samples - 1):
                        self.demos.append((key, i))
                        
        print(f"Dataset loaded. transitions: {len(self.demos)}, text_mode: {self.text_mode}")

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, "r") as f:
            demo_key, t = self.demos[idx]
            
            z_t = f[f"data/{demo_key}/obs_embedding"][t]
            a_t = f[f"data/{demo_key}/actions"][t]
            z_next = f[f"data/{demo_key}/obs_embedding"][t+1]
            
            # Load text embedding
            if self.text_mode == "multiple":
                # Data Augmentation: Pick one random synonym
                all_embs = f[f"data/{demo_key}/text_embeddings"][:] # (N, 512)
                idx = np.random.randint(len(all_embs))
                text_emb = all_embs[idx]
            elif self.text_mode == "single":
                text_emb = f[f"data/{demo_key}/text_embedding"][:]
            else:
                text_emb = np.zeros(512, dtype=np.float32)

        return {
            "state": torch.from_numpy(z_t).float(),
            "action": torch.from_numpy(a_t).float(),
            "text": torch.from_numpy(text_emb).float(),
            "next_state": torch.from_numpy(z_next).float()
        }