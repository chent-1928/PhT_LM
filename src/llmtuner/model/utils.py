import inspect

import torch
from transformers import PreTrainedModel
from ..extras.misc import get_current_device


def dispatch_model(model: "PreTrainedModel") -> "PreTrainedModel":
    r"""
    Dispatches a pre-trained model to GPUs with balanced memory when the GPU is available.
    Borrowed from: https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/modeling_utils.py#L3570
    """
    if getattr(model, "quantization_method", None):  # already set on current device
        return model

    if (
        torch.cuda.device_count() > 1
        and isinstance(model, PreTrainedModel)
        and model._no_split_modules is not None
        and model.config.model_type != "chatglm"
    ):
        from accelerate import dispatch_model
        from accelerate.utils import get_balanced_memory, infer_auto_device_map

        kwargs = {"dtype": model.dtype, "no_split_module_classes": model._get_no_split_modules("auto")}
        max_memory = get_balanced_memory(model, **kwargs)
        # Make sure tied weights are tied before creating the device map.
        model.tie_weights()
        device_map = infer_auto_device_map(model, max_memory=max_memory, **kwargs)
        device_map_kwargs = {"device_map": device_map}
        if "skip_keys" in inspect.signature(dispatch_model).parameters:
            device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
        return dispatch_model(model, **device_map_kwargs)
    else:
        return model.to(device=get_current_device())