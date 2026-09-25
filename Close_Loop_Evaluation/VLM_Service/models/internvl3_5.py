import base64
from io import BytesIO

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .VLMInterface import VLMInterface
from .interact_utils import get_carla_image_descriptions, get_image_descriptions

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for idx in range(blocks):
        box = (
            (idx % (target_width // image_size)) * image_size,
            (idx // (target_width // image_size)) * image_size,
            ((idx % (target_width // image_size)) + 1) * image_size,
            ((idx // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str) and image_file.startswith("data:image/"):
        _, base64_data = image_file.split(",", 1)
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(processed_image) for processed_image in images]
    return torch.stack(pixel_values)


def image_template(image_path, use_base64):
    return ""


class InternVL35Interface(VLMInterface):
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
        print(f"Initializing InternVL3.5 on GPU {gpu_id}...")
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
            raise RuntimeError("InternVL3.5 requires at least one visible CUDA device")
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
            load_kwargs["device_map"] = "auto"
            print(
                f"Detected {self.visible_gpu_count} visible GPUs, "
                "loading InternVL3.5 with automatic sharding"
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
            print(f"InternVL3.5 loaded on GPU {self.primary_gpu_id} successfully")
        else:
            print(
                "InternVL3.5 loaded successfully across "
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

        input_conversation = (
            all_context_str + f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}"
        )

        print(f"[debug] conversation = {input_conversation}")
        print(f"[debug] input_image_files = {input_image_files}")

        input_image = input_image_files[-1] if input_image_files else None
        pixel_values = load_image(input_image, max_num=12).to(
            device=self.device, dtype=torch.bfloat16
        )
        generation_config = dict(max_new_tokens=1024, do_sample=False)

        response = self.model.chat(
            self.tokenizer, pixel_values, input_conversation, generation_config
        )

        if isinstance(response, list):
            return response[0]
        return response
