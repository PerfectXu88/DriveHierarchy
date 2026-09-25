import base64
from io import BytesIO

from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .VLMInterface import VLMInterface
from .prompt_utils import (
    get_bev_img_desc,
    get_bev_traj_img_desc,
    get_concat_img_desc,
    get_concat_traj_img_desc,
    get_front_img_desc,
    get_front_traj_img_desc,
)


def load_image(image_ref, use_base64):
    if use_base64 and isinstance(image_ref, str) and image_ref.startswith("data:image/"):
        _, base64_data = image_ref.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(base64_data))).convert("RGB")
    return Image.open(image_ref).convert("RGB")


class MiniCPMV45Interface(VLMInterface):
    model_label = "MiniCPM-V-4_5"

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
    ):
        print(f"Initializing {self.model_label} on GPU {gpu_id}...")
        self.in_carla = in_carla
        self.use_bev = use_bev
        self.use_all_cameras = use_all_cameras
        self.input_window = input_window
        self.no_history = no_history
        self.primary_gpu_id = gpu_id
        self.input_gpu_id = gpu_id
        self.model_path = model_path
        self.frame_rate = frame_rate
        self.use_base64 = use_base64

        self.visible_gpu_count = torch.cuda.device_count()
        if self.visible_gpu_count < 1:
            raise RuntimeError(f"{self.model_label} requires at least one visible CUDA device")
        if self.primary_gpu_id < 0 or self.primary_gpu_id >= self.visible_gpu_count:
            raise ValueError(
                f"GPU_ID={self.primary_gpu_id} is out of range for "
                f"{self.visible_gpu_count} visible CUDA device(s)"
            )

        self.multi_image_flag = self.use_all_cameras or (
            self.no_history is False and self.input_window > 1
        )

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        if self.visible_gpu_count > 1:
            load_kwargs["device_map"] = "auto"
            print(
                f"Detected {self.visible_gpu_count} visible GPUs, "
                f"loading {self.model_label} with automatic sharding"
            )

        self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        if self.visible_gpu_count == 1:
            torch.cuda.set_device(self.primary_gpu_id)
            self.model = self.model.to(torch.device(f"cuda:{self.primary_gpu_id}"))
        else:
            self.input_gpu_id = self._resolve_input_gpu_id()

        self.device = torch.device(f"cuda:{self.input_gpu_id}")

        if self.visible_gpu_count == 1:
            print(f"{self.model_label} loaded on GPU {self.primary_gpu_id} successfully")
        else:
            print(
                f"{self.model_label} loaded successfully across "
                f"{self.visible_gpu_count} visible GPUs"
            )

    def _resolve_input_gpu_id(self):
        device_map = getattr(self.model, "hf_device_map", None)
        if not isinstance(device_map, dict):
            return self.primary_gpu_id

        gpu_ids = []
        for device_ref in device_map.values():
            gpu_id = self._device_ref_to_gpu_id(device_ref)
            if gpu_id is not None:
                gpu_ids.append(gpu_id)

        if not gpu_ids:
            return self.primary_gpu_id
        return min(gpu_ids)

    @staticmethod
    def _device_ref_to_gpu_id(device_ref):
        if isinstance(device_ref, int):
            return device_ref
        if isinstance(device_ref, str) and device_ref.startswith("cuda:"):
            return int(device_ref.split(":", 1)[1])
        if isinstance(device_ref, torch.device) and device_ref.type == "cuda":
            return device_ref.index
        return None

    def _append_image_block(self, content, description, image_path):
        content.append(description)
        content.append(load_image(image_path, self.use_base64))

    def _build_frame_content(self, frame_number, frame_images):
        content = []

        if self.in_carla:
            if self.use_bev:
                self._append_image_block(
                    content,
                    get_front_traj_img_desc(frame_number, self.frame_rate),
                    frame_images["CAM_FRONT"],
                )
                self._append_image_block(
                    content,
                    get_bev_traj_img_desc(frame_number, self.frame_rate),
                    frame_images["ANNO_BEV"],
                )
                return content

            if self.use_all_cameras:
                for key in ("CAM_FRONT_CONCAT", "CAM_BACK_CONCAT"):
                    if key in frame_images:
                        self._append_image_block(
                            content,
                            get_concat_traj_img_desc(frame_number, self.frame_rate),
                            frame_images[key],
                        )
                return content

            self._append_image_block(
                content,
                get_front_traj_img_desc(frame_number, self.frame_rate),
                frame_images["CAM_FRONT"],
            )
            return content

        if self.use_all_cameras:
            for key in ("CAM_FRONT_CONCAT", "CAM_BACK_CONCAT"):
                if key in frame_images:
                    self._append_image_block(
                        content,
                        get_concat_img_desc(frame_number, self.frame_rate),
                        frame_images[key],
                    )
            return content

        self._append_image_block(
            content,
            get_front_img_desc(frame_number, self.frame_rate),
            frame_images["CAM_FRONT"],
        )
        if self.use_bev and "ANNO_BEV" in frame_images:
            self._append_image_block(
                content,
                get_bev_img_desc(frame_number, self.frame_rate),
                frame_images["ANNO_BEV"],
            )
        return content

    def _extend_frame_range_content(
        self,
        content,
        images_dict,
        image_frame_list,
        start_frame,
        end_frame,
    ):
        for frame_number in image_frame_list:
            if start_frame < frame_number <= end_frame and frame_number in images_dict:
                content.extend(self._build_frame_content(frame_number, images_dict[frame_number]))

    def interact(self, bubble, conversation):
        torch.cuda.set_device(self.input_gpu_id)
        self.device = torch.device(f"cuda:{self.input_gpu_id}")

        messages = []
        images_list = bubble.get_full_images()
        image_frame_list = sorted(images_list.keys())
        current_frame = bubble.frame_number
        prev_frame = -1

        if self.no_history is False:
            for history_bubble in conversation:
                if history_bubble.actor == "User":
                    content = []
                    self._extend_frame_range_content(
                        content,
                        images_list,
                        image_frame_list,
                        prev_frame,
                        history_bubble.frame_number,
                    )
                    prev_frame = history_bubble.frame_number
                    content.append(
                        f"Q(frame {history_bubble.frame_number}): {history_bubble.words}"
                    )
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [f"A(frame {history_bubble.frame_number}): {history_bubble.words}"],
                        }
                    )

        current_content = []
        self._extend_frame_range_content(
            current_content,
            images_list,
            image_frame_list,
            prev_frame,
            current_frame,
        )
        current_content.append(f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}")
        messages.append({"role": "user", "content": current_content})

        response = self.model.chat(
            msgs=messages,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_new_tokens=1024,
            sampling=False,
            stream=False,
            enable_thinking=False,
        )

        if isinstance(response, list):
            return response[0]
        return response
