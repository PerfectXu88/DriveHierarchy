import importlib.util

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .qwen3 import Qwen3Interface


class Qwen3LoRAInterface(Qwen3Interface):
    model_label = "Qwen3-VL-LoRA"

    def initialize(
        self,
        gpu_id: int,
        use_all_cameras: bool,
        no_history: bool,
        input_window: int,
        frame_rate: int,
        model_path: str,
        use_bev: bool = False,
        in_carla: bool = False,
        use_base64: bool = False,
        base_model_path: str = "",
    ):
        print(f"Initializing {self.model_label} on GPU {gpu_id}...")
        self.in_carla = in_carla
        self.use_bev = use_bev
        self.use_all_cameras = use_all_cameras
        self.input_window = input_window
        self.no_history = no_history
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.base_model_path = str(base_model_path).strip()
        self.frame_rate = frame_rate
        self.use_base64 = use_base64

        if not self.base_model_path:
            raise ValueError(
                f"{self.model_label} requires BASE_MODEL_PATH in the config file"
            )

        if importlib.util.find_spec("peft") is None:
            raise ImportError(
                f"{self.model_label} requires the `peft` package in the runtime environment"
            )

        self.visible_gpu_count = torch.cuda.device_count()
        if self.visible_gpu_count < 1:
            raise RuntimeError(f"{self.model_label} requires at least one visible CUDA device")
        if self.gpu_id < 0 or self.gpu_id >= self.visible_gpu_count:
            raise ValueError(
                f"GPU_ID={self.gpu_id} is out of range for "
                f"{self.visible_gpu_count} visible CUDA device(s)"
            )

        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        self.multi_image_flag = (
            self.use_all_cameras
            or (self.no_history is False and self.input_window > 1)
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.base_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        ).eval()
        self.model.to(self.device)
        self.model.load_adapter(
            self.model_path,
            adapter_name="default",
            device_map=self.gpu_id,
            low_cpu_mem_usage=True,
        )
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter("default")

        self.processor = AutoProcessor.from_pretrained(self.base_model_path)

        print(f"{self.model_label} loaded on GPU {gpu_id} successfully")
