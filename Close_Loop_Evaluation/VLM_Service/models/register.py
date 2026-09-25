MODEL_MAP = {
    "gt": "VLMInterface",
    "Qwen2.5VL": "Qwen25Interface",
    "Qwen2.5-VL-7B-DriveLM": "Qwen25DriveLMInterface",
    "Qwen3VL": "Qwen3Interface",
    "Qwen3VL-LoRA": "Qwen3LoRAInterface",
    "ZwZ": "ZwZInterface",
    "Pixtral": "PixtralInterface",
    "MiniCPM-V-4_5": "MiniCPMV45Interface",
    "Gemma": "GemmaInterface",
    "InternVL": "InternVLInterface",
    "InternVL3": "InternVL3Interface",
    "InternVL3_5": "InternVL35Interface",
    "Mini-InternVL2-DA-DriveLM": "MiniInternVL2DADriveLMInterface",
}

def get_model_interface(model_name):
    """
    Retrieve the appropriate model interface class based on the model name.
    :param model_name: Name of the model.
    :return: An instance of the corresponding model interface.
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model {model_name} is not supported. Available models: {list(MODEL_MAP.keys())}")
    
    # Lazy import based on model_name
    if model_name == "gt":
        from .VLMInterface import VLMInterface
        return VLMInterface()
    elif model_name == "Qwen2.5VL":
        from .qwen25 import Qwen25Interface
        return Qwen25Interface()
    elif model_name == "Qwen2.5-VL-7B-DriveLM":
        from .qwen25_drivelm import Qwen25DriveLMInterface
        return Qwen25DriveLMInterface()
    elif model_name == "Gemma":
        from .gemma import GemmaInterface
        return GemmaInterface()
    elif model_name == "Qwen3VL":
        from .qwen3 import Qwen3Interface
        return Qwen3Interface()
    elif model_name == "Qwen3VL-LoRA":
        from .qwen3_lora import Qwen3LoRAInterface
        return Qwen3LoRAInterface()
    elif model_name == "ZwZ":
        from .zwz import ZwZInterface
        return ZwZInterface()
    elif model_name == "Pixtral":
        from .pixtral import PixtralInterface
        return PixtralInterface()
    elif model_name == "MiniCPM-V-4_5":
        from .minicpm_v_4_5 import MiniCPMV45Interface
        return MiniCPMV45Interface()
    elif model_name == "InternVL":
        from .intern import InternVLInterface
        return InternVLInterface()
    elif model_name == "InternVL3":
        from .internvl3 import InternVL3Interface
        return InternVL3Interface()
    elif model_name == "InternVL3_5":
        from .internvl3_5 import InternVL35Interface
        return InternVL35Interface()
    elif model_name == "Mini-InternVL2-DA-DriveLM":
        from .mini_internvl2_da_drivelm import MiniInternVL2DADriveLMInterface
        return MiniInternVL2DADriveLMInterface()
    raise ValueError(f"Model {model_name} is not supported. Available models: {list(MODEL_MAP.keys())}")
