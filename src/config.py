import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
    use_scheduler: bool = True
    warmup_steps: int = 1000
    save_dir: str = "./results/checkpoints"
    device: str = "cuda"

@dataclass
class DataConfig:
    dataset_path: str = "./data/lift_ph_embeddings.hdf5"
    num_workers: int = 4

@dataclass
class ModelConfig:
    text_dim: int = 512
    hidden_dim: int = 256
    state_dim: Optional[int] = None # Inferred from data
    action_dim: Optional[int] = None # Inferred from data

@dataclass
class Config:
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        
        return cls(
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {}))
        )
