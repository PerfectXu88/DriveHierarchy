import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from .VLMInterface import VLMInterface
from .interact_utils import get_carla_image_descriptions, get_image_descriptions


def image_template(image_path, use_base64):
    return {
        "type": "image",
        "image": image_path,
    }


class GemmaInterface(VLMInterface):
    model_label = "Gemma 3"

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

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
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

        messages = []
        images_list = bubble.get_full_images()
        image_frame_list = sorted(images_list.keys())
        current_frame = bubble.frame_number
        prev_frame = -1

        if self.no_history is False:
            for history_bubble in conversation:
                if history_bubble.actor == "User":
                    content = []
                    if prev_frame < history_bubble.frame_number:
                        image_content, _ = self.get_image_descriptions(
                            images_list,
                            image_frame_list,
                            prev_frame,
                            history_bubble.frame_number,
                        )
                        content.extend(image_content)
                        prev_frame = history_bubble.frame_number
                    content.append(
                        {
                            "type": "text",
                            "text": f"Q(frame {history_bubble.frame_number}): {history_bubble.words}",
                        }
                    )
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append(
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
            image_content, _ = self.get_image_descriptions(
                images_list,
                image_frame_list,
                prev_frame,
                current_frame,
            )
            current_content.extend(image_content)

        current_content.append(
            {
                "type": "text",
                "text": f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}",
            }
        )
        messages.append({"role": "user", "content": current_content})

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        result = decoded if isinstance(decoded, str) else decoded[0]
        return result
