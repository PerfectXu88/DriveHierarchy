"""Shared constants for Open Loop evaluation."""

TASK_YES_NO = "yes_no"
TASK_NUMERIC = "numeric"
TASK_MULTI_CHOICE = "multi_choice"
TASK_BBOX = "bbox"
TASK_TEXT = "text"
TASK_SEQUENCE_ORDER = "sequence_order"

BACKEND_VLLM = "vllm"
BACKEND_TRANSFORMERS = "transformers"

DEFAULT_SYSTEM_PROMPT = (
    "You are a mature and cautious driver with many years of driving experience. "
    "You now need assistance answering questions related to the driving process."
)

