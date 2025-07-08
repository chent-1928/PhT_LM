from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    template: Optional[str] = field(
        default="qwen", metadata={"help": "Which template to use for constructing prompts in training and inference."}
    )