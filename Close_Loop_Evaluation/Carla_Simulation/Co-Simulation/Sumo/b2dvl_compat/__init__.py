# -*- coding: utf-8 -*-
"""
Compatibility layer for carla_base (Python 3.8).

This package contains minimal copied helper modules needed by the VLM bridge
running inside the carla_base conda environment. Only pure-Python,
stdlib-only modules are included so that no extra dependencies
(torch, transformers, …) are required.
"""

from .inference_utils import Bubble, Context, create_query, create_response
from .waypoint_decoder import decode_xy_token, decode_polar_token
from .waypoint_encoder import generate_motion_tokens, generate_motion_and_direction_tokens
from .commands import SpeedCommand, DirectionCommand
