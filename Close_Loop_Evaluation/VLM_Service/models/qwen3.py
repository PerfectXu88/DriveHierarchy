"""
Qwen3-VL model interface for close-loop evaluation.

Based on qwen25.py but uses Qwen3VLForConditionalGeneration
which is the native class for Qwen3-VL models in transformers >= 4.51.
"""

from .VLMInterface import VLMInterface
from .interact_utils import get_image_descriptions, get_carla_image_descriptions
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


def image_template(image_path, use_base64):
    return {
        "type": "image",
        "image": image_path if use_base64 else f"file://{image_path}"
    }


class Qwen3Interface(VLMInterface):
    model_label = "Qwen3-VL"

    def initialize(self, gpu_id: int, use_all_cameras: bool, no_history: bool,
                   input_window: int, frame_rate: int, model_path: str,
                   use_bev: bool = False, in_carla: bool = False,
                   use_base64: bool = False):
        print(f"Initializing {self.model_label} on GPU {gpu_id}...")
        self.in_carla = in_carla
        self.use_bev = use_bev
        self.use_all_cameras = use_all_cameras
        self.input_window = input_window
        self.no_history = no_history
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.frame_rate = frame_rate
        self.use_base64 = use_base64

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

        load_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
        }
        if self.visible_gpu_count > 1:
            load_kwargs["device_map"] = "auto"
            print(
                f"Detected {self.visible_gpu_count} visible GPUs, "
                f"loading {self.model_label} with automatic sharding"
            )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, **load_kwargs
        )
        if self.visible_gpu_count == 1:
            self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        if self.visible_gpu_count == 1:
            print(f"{self.model_label} loaded on GPU {gpu_id} successfully")
        else:
            print(
                f"{self.model_label} loaded successfully across "
                f"{self.visible_gpu_count} visible GPUs"
            )

    def get_image_descriptions(self, images_dict, image_frame_list,
                               start_frame, end_frame):
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
        else:
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
        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        input_conversation = []

        images_list = bubble.get_full_images()
        image_frame_list = sorted(images_list.keys())

        start_frame = bubble.frame_number
        current_frame = bubble.frame_number
        if conversation is not None and len(conversation) > 0:
            start_frame = conversation[0].frame_number

        prev_frame = -1

        # ---- Context (conversation history) ----
        context_str = ""
        if self.no_history is False:
            context_bb = {
                "role": "context",
                "content": []
            }

            for bb in conversation:
                is_user = (bb.actor == "User")

                if is_user and prev_frame < bb.frame_number:
                    image_content, _ = self.get_image_descriptions(
                        images_list, image_frame_list,
                        prev_frame, bb.frame_number)
                    prev_frame = bb.frame_number

                    if context_str:
                        context_bb['content'].append({
                            "type": "text",
                            "text": context_str
                        })
                        context_str = ""

                    context_bb['content'].extend(image_content)

                header = "Q" if is_user else "A"
                context_str += f"{header}(frame {bb.frame_number}): {bb.words}\n"

            if context_str:
                context_bb['content'].append({
                    "type": "text",
                    "text": context_str
                })
                context_str = ""

            input_conversation.append(context_bb)

        # ---- Current question ----
        bb_dict = {
            "role": "user",
            "content": []
        }
        if prev_frame < current_frame:
            image_content, _ = self.get_image_descriptions(
                images_list, image_frame_list,
                prev_frame, current_frame)
            prev_frame = current_frame
            bb_dict['content'].extend(image_content)

        bb_dict['content'].append({
            "type": "text",
            "text": f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}"
        })
        input_conversation.append(bb_dict)

        # ---- Generate ----
        prompts = self.processor.apply_chat_template(
            input_conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(input_conversation)

        inputs = self.processor(
            text=[prompts], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        result = output_text
        if isinstance(result, list):
            result = result[0]

        return result
