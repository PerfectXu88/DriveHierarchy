import math

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .VLMInterface import VLMInterface
from .interact_utils import get_carla_image_descriptions, get_image_descriptions
from .internvl3_5 import load_image


def image_template(image_path, use_base64):
    return ""


class InternVL3Interface(VLMInterface):
    model_label = "InternVL3"

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
            "use_flash_attn": True,
            "trust_remote_code": True,
        }
        if self.visible_gpu_count > 1:
            load_kwargs["device_map"] = self._build_large_model_device_map()
            print(
                f"Detected {self.visible_gpu_count} visible GPUs, "
                f"loading {self.model_label} with the official large-model sharding map"
            )

        self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
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

    def _build_large_model_device_map(self):
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        llm_config = getattr(config, "llm_config")
        if hasattr(llm_config, "num_hidden_layers"):
            num_layers = llm_config.num_hidden_layers
        else:
            num_layers = llm_config["num_hidden_layers"]

        world_size = self.visible_gpu_count
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

        device_map = {}
        layer_cnt = 0
        for gpu_idx, layer_budget in enumerate(num_layers_per_gpu):
            for _ in range(layer_budget):
                if layer_cnt >= num_layers:
                    break
                device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_idx
                layer_cnt += 1

        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.model.rotary_emb"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        return device_map

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

    def get_image_descriptions(self, images_dict, image_frame_list, start_frame, end_frame):
        if self.in_carla:
            return get_carla_image_descriptions(
                images_dict=images_dict,
                image_frame_list=image_frame_list,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_rate=self.frame_rate,
                template_func=image_template,
                use_all_cameras=self.use_all_cameras,
                use_bev=self.use_bev,
                use_base64=self.use_base64,
            )
        return get_image_descriptions(
            images_dict=images_dict,
            image_frame_list=image_frame_list,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_rate=self.frame_rate,
            template_func=image_template,
            use_all_cameras=self.use_all_cameras,
            use_base64=self.use_base64,
        )

    @staticmethod
    def _build_image_prefix(image_count: int):
        if image_count <= 0:
            return ""
        if image_count == 1:
            return "<image>\n"
        return "".join(
            f"Image-{index}: <image>\n" for index in range(1, image_count + 1)
        )

    def interact(self, bubble, conversation):
        torch.cuda.set_device(self.input_gpu_id)
        self.device = torch.device(f"cuda:{self.input_gpu_id}")

        images_list = bubble.get_full_images()
        image_frame_list = sorted(images_list.keys())
        input_image_files = []

        current_frame = bubble.frame_number
        prev_frame = -1
        all_context_str = ""

        if self.no_history is False:
            for history_bubble in conversation:
                is_user = history_bubble.actor == "User"

                if is_user and prev_frame < history_bubble.frame_number:
                    image_content, image_dirs = self.get_image_descriptions(
                        images_list,
                        image_frame_list,
                        prev_frame,
                        history_bubble.frame_number,
                    )
                    input_image_files.extend(image_dirs)
                    prev_frame = history_bubble.frame_number
                    all_context_str += image_content

                header = "Q" if is_user else "A"
                all_context_str += (
                    f"{header}(frame {history_bubble.frame_number}): "
                    f"{history_bubble.words}\n"
                )

        if prev_frame < current_frame:
            image_content, image_dirs = self.get_image_descriptions(
                images_list,
                image_frame_list,
                prev_frame,
                current_frame,
            )
            input_image_files.extend(image_dirs)
            all_context_str += image_content

        pixel_values_list = []
        num_patches_list = []
        for image_file in input_image_files:
            image_pixel_values = load_image(image_file, max_num=12)
            pixel_values_list.append(image_pixel_values)
            num_patches_list.append(image_pixel_values.size(0))

        pixel_values = None
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0).to(
                device=self.device, dtype=torch.bfloat16
            )

        input_conversation = (
            self._build_image_prefix(len(pixel_values_list))
            + all_context_str
            + f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}"
        )

        print(f"[debug] conversation = {input_conversation}")
        print(f"[debug] input_image_files = {input_image_files}")

        generation_config = dict(max_new_tokens=1024, do_sample=False)
        chat_kwargs = {}
        if len(num_patches_list) > 1:
            chat_kwargs["num_patches_list"] = num_patches_list

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            input_conversation,
            generation_config,
            **chat_kwargs,
        )

        if isinstance(response, list):
            return response[0]
        return response
