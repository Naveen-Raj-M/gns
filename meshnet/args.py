from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    path: str = MISSING
    batch_size: int = 2
    noise_std: float = 2e-3
    input_sequence_length: int = 1
    node_type_embedding_size: int = 9
    dt: float = 0.01


@dataclass
class ModelConfig:
    path: str = "../gns-sample/Cylinder/models/"
    file: Optional[str] = None
    train_state_file: Optional[str] = None


@dataclass
class OutputConfig:
    path: str = "../gns-sample/Cylinder/rollouts/"
    filename: str = "rollout"


@dataclass
class LearningRateConfig:
    initial: float = 1e-4
    decay: float = 0.1
    decay_steps: int = 5000000


@dataclass
class TrainingConfig:
    steps: int = 100
    validation_interval: Optional[int] = None
    save_steps: int = 500
    resume: Optional[bool] = False
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)


@dataclass
class HardwareConfig:
    cuda_device_number: Optional[int] = None


@dataclass
class LoggingConfig:
    tensorboard_dir: str = "logs/"


@dataclass
class Config:
    mode: str = "train"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Hydra configuration
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)