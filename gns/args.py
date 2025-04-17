from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    path: str = MISSING
    batch_size: int = 2
    noise_std: float = 6.7e-4
    input_sequence_length: int = 6
    num_particle_types: int = 9
    kinematic_particle_id: int = 3


@dataclass
class ModelConfig:
    path: str = "models/"
    file: Optional[str] = None
    train_state_file: Optional[str] = None


@dataclass
class OutputConfig:
    path: str = "rollouts/"
    filename: str = "rollout"


@dataclass
class LearningRateConfig:
    initial: float = 1e-4
    decay: float = 0.1
    decay_steps: int = 50000


@dataclass
class TrainingConfig:
    steps: int = 2000
    validation_interval: Optional[int] = None
    save_steps: int = 500
    resume: Optional[bool] = False
    parameters: Optional[List[str]] = field(default_factory=lambda: ["encoder", "processor", "decoder"])
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    nmessage_passing_steps: Optional[int] = 10
    

@dataclass
class HardwareConfig:
    cuda_device_number: Optional[int] = None


@dataclass
class LoggingConfig:
    tensorboard_dir: str = "logs/"


@dataclass
class GifConfig:
    step_stride: int = 3
    vertical_camera_angle: int = 20
    viewpoint_rotation: float = 0.3
    change_yz: bool = False


@dataclass
class RenderingConfig:
    mode: Optional[str] = field(default="gif")
    gif: GifConfig = field(default_factory=GifConfig)


@dataclass
class InnerLoopConfig:
    n_examples: int = 2
    iterations: int = 5


@dataclass
class OuterLoopConfig:
    batch_size: int = 5
    step_size: int = 1
    iterations: int = 10000


@dataclass
class ReptileConfig:
    n_task: int = 3
    parameters: Optional[List[str]] = field(default_factory=lambda: ["encoder", "processor", "decoder"])
    inner_loop: InnerLoopConfig = field(default_factory=InnerLoopConfig)
    outer_loop: OuterLoopConfig = field(default_factory=OuterLoopConfig)


@dataclass
class PretrainedModelConfig:
    path: str = "models/"
    file: Optional[str] = None
    train_state_file: Optional[str] = None


@dataclass
class Config:
    mode: str = "train"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    reptile: ReptileConfig = field(default_factory=ReptileConfig)
    pretrained_model: PretrainedModelConfig = field(default_factory=PretrainedModelConfig)

# Hydra configuration
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
