import base64
from io import BytesIO

from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .VLMInterface import VLMInterface
from .interact_utils import get_carla_image_descriptions, get_image_descriptions


def image_template(image_path, use_base64):
    return {"type": "image"}


def load_image(image_ref, use_base64):
    if use_base64 and isinstance(image_ref, str) and image_ref.startswith("data:image/"):
        _, base64_data = image_ref.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(base64_data))).convert("RGB")
    return Image.open(image_ref).convert("RGB")


class PixtralInterface(VLMInterface):
    model_label = "Pixtral-12B"

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
        }
        if self.visible_gpu_count > 1:
            load_kwargs["device_map"] = "auto"
            print(
                f"Detected {self.visible_gpu_count} visible GPUs, "
                f"loading {self.model_label} with automatic sharding"
            )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, **load_kwargs
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

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

        input_conversation = []
        input_image_files = []

        images_list = bubble.get_full_images()
        image_frame_list = sorted(images_list.keys())
        current_frame = bubble.frame_number
        prev_frame = -1

        if self.no_history is False:
            for history_bubble in conversation:
                if history_bubble.actor == "User":
                    content = []
                    if prev_frame < history_bubble.frame_number:
                        image_content, image_dirs = self.get_image_descriptions(
                            images_list,
                            image_frame_list,
                            prev_frame,
                            history_bubble.frame_number,
                        )
                        content.extend(image_content)
                        input_image_files.extend(image_dirs)
                        prev_frame = history_bubble.frame_number
                    content.append(
                        {
                            "type": "text",
                            "text": f"Q(frame {history_bubble.frame_number}): {history_bubble.words}",
                        }
                    )
                    input_conversation.append({"role": "user", "content": content})
                else:
                    input_conversation.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"A(frame {history_bubble.frame_number}): {history_bubble.words}",
                                }
                            ],
                        }
                    )

        current_content = []
        if prev_frame < current_frame:
            image_content, image_dirs = self.get_image_descriptions(
                images_list,
                image_frame_list,
                prev_frame,
                current_frame,
            )
            current_content.extend(image_content)
            input_image_files.extend(image_dirs)

        current_content.append(
            {
                "type": "text",
                "text": f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}",
            }
        )
        input_conversation.append({"role": "user", "content": current_content})

        prompt = self.processor.apply_chat_template(
            input_conversation, tokenize=False, add_generation_prompt=True
        )
        input_images = [
            load_image(image_ref, self.use_base64) for image_ref in input_image_files
        ]

        inputs = self.processor(
            text=prompt,
            images=input_images if input_images else None,
            return_tensors="pt",
        )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        result = output_text[0] if isinstance(output_text, list) else output_text
        return result
