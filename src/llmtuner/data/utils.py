from enum import Enum, unique
from typing import Tuple

@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"
    FUNCTION = "function"


def infer_max_len(source_len: int, target_len: int, max_len: int, reserved_label_len: int) -> Tuple[int, int]:
    max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, reserved_label_len)
    max_source_len = max_len - max_target_len
    return max_source_len, max_target_len
