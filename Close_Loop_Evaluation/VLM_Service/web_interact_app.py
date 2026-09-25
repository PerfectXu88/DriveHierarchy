import argparse
import json
import os
import sys
from typing import List

import uvicorn
import yaml
from fastapi import FastAPI

PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT)

from VLM_Service.inference_utils import Bubble
from VLM_Service.models import get_model_interface

def load_config(config_path):
    _, ext = os.path.splitext(config_path)
    with open(config_path, "r") as f:
        if ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format: use .yaml/.yml or .json")

def create_app(model_interface):
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/interact")
    def interact_route(request: dict):
        bubble = Bubble.from_dict(request["bubble"])
        conversation = [Bubble.from_dict(bb) for bb in request.get("conversation", [])]
        output = model_interface.interact(bubble, conversation)
        # print(f"[debug] VLM output: {output}")
        return {"response": output}

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)

    def get_cfg(path: List[str], default=None):
        ref = config
        try:
            for key in path:
                ref = ref[key]
            return ref
        except KeyError:
            return default

    model_interface = get_model_interface(config.get("MODEL_NAME"))

    initialize_kwargs = {
        "gpu_id": config.get("GPU_ID", 0),
        "use_all_cameras": get_cfg(["INFERENCE_BASICS", "USE_ALL_CAMERAS"], False),
        "no_history": get_cfg(["INFERENCE_BASICS", "NO_HISTORY_MODE"], False),
        "input_window": get_cfg(["INFERENCE_BASICS", "INPUT_WINDOW"], 1),
        "frame_rate": get_cfg(["TASK_CONFIGS", "FRAME_PER_SEC"], 10),
        "model_path": config.get("MODEL_PATH", ""),
        "use_bev": get_cfg(["INFERENCE_BASICS", "USE_BEV"], False),
        "in_carla": config.get("IN_CARLA", False),
        "use_base64": config.get("USE_BASE64", False),
    }
    if "BASE_MODEL_PATH" in config:
        initialize_kwargs["base_model_path"] = config.get("BASE_MODEL_PATH", "")

    model_interface.initialize(**initialize_kwargs)

    app = create_app(model_interface)
    host = "0.0.0.0"
    port = config.get("PORT", 7023)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
