import os
from typing import TYPE_CHECKING, Tuple

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version

from ..extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ..hparams import ModelArguments


logger = get_logger(__name__)


require_version("transformers>=4.36.2", "To fix: pip install transformers>=4.36.2")
require_version("datasets>=2.14.3", "To fix: pip install datasets>=2.14.3")
require_version("accelerate>=0.21.0", "To fix: pip install accelerate>=0.21.0")
require_version("peft>=0.7.0", "To fix: pip install peft>=0.7.0")
require_version("trl>=0.7.6", "To fix: pip install trl>=0.7.6")


def load_model_and_tokenizer(model_args: "ModelArguments") -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **config_kwargs,
    )

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs,
    )

    model.requires_grad_(False)
    model = model.to(model_args.compute_dtype) if not getattr(model, "quantization_method", None) else model
    model.eval()

    for param in model.parameters():
        logger.info(param.dtype)  # 打印每个参数的精度
        break  # 只需要检查一个参数，因为所有参数通常具有相同的精度

    return model, tokenizer
