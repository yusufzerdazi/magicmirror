from pydantic.v1 import BaseSettings, Field
from config import load_server_config
import os


class Settings(BaseSettings):
    # config, cannot be changed
    mode: str = Field(default="video")
    worker_id: int = Field(default=0)

    # Add server_password field
    server_password: str = Field(default="")

    output_fast: bool = Field(default=True)
    zmq_video_port: int = Field(default=5554)
    job_start_port: int = Field(default=5555)
    settings_port: int = Field(default=5556)
    job_finish_port: int = Field(default=5557)
    output_port: int = Field(default=5558)
    osc_port: int = Field(default=9091)
    primary_hostname: str = Field(default="0.0.0.0")
    websocket_port = 8765
    websocket_address = "ws://0.0.0.0:8765"

    safety: bool = Field(default=False)
    local_files_only: bool = Field(default=False)
    warmup: str = Field(default=None)
    threaded: bool = Field(default=True)

    # parameters for inference
    prompt: str = Field(default="A psychedelic landscape.")
    num_inference_steps: int = Field(default=2)
    fixed_seed: bool = Field(default=True)
    seed: int = Field(default=0)
    batch_size: int = Field(default=4)
    strength: float = Field(default=0.7)
    passthrough: bool = Field(default=False)
    compel: bool = Field(default=True)

    # can be changed dynamically
    opacity: float = Field(default=1.0)
    mirror: bool = Field(default=False)
    debug: bool = Field(default=False)
    pad: bool = Field(default=False)
    fps: int = Field(default=30)
    directory: str = Field(default="data/frames")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            # Load server config
            server_config = load_server_config()
            self.server_password = server_config.SERVER_PASSWORD
            print(f"Successfully loaded server password")  # Debug line
        except Exception as e:
            print(f"Error loading server password: {e}")
            raise

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
