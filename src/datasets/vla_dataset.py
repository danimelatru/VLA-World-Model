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
                all_embs = f[f"data/{demo_key}/text_embeddings"][:] 
                idx = np.random.randint(len(all_embs))
                text_emb = all_embs[idx]
            elif self.text_mode == "single":
                text_emb = f[f"data/{demo_key}/text_embedding"][:]
            else:
                text_emb = np.zeros(512, dtype=np.float32)
                
            # VLA Support: We need raw text string too
            # Parse task from key "lift_demo_0" -> "lift"
            task_name = demo_key.split("_")[0]
            
            # Simple prompt map (ideally centralized config)
            PROMPTS = {
                "lift": ["Lift the red cube", "Pick up the red cube"],
                "can": ["Pick up the coke can", "Grasp the can"],
                "square": ["Push the square nut", "Insert the nut"]
            }
            # Fallback
            prompts = PROMPTS.get(task_name, ["Do the task"])
            raw_text = np.random.choice(prompts)

        return {
            "state": torch.from_numpy(z_t).float(),
            "action": torch.from_numpy(a_t).float(),
            "text": torch.from_numpy(text_emb).float(), # CLIP Vector
            "next_state": torch.from_numpy(z_next).float(),
            "raw_text": raw_text # String for LLM
        }

        return {
            "state": torch.from_numpy(z_t).float(),
            "action": torch.from_numpy(a_t).float(),
            "text": torch.from_numpy(text_emb).float(),
            "next_state": torch.from_numpy(z_next).float()
        }