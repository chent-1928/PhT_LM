import json
import os
from typing import Any, Dict, Optional

from peft.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME


ADAPTER_NAMES = {WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME}
DEFAULT_CACHE_DIR = "cache"
DEFAULT_DATA_DIR = "data"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user.config"



def get_config_path() -> os.PathLike:
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def load_config() -> Dict[str, Any]:
    try:
        with open(get_config_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"lang": None, "last_model": None, "path_dict": {}, "cache_dir": None}


def save_config(lang: str, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = load_config()
    user_config["lang"] = lang or user_config["lang"]
    if model_name:
        user_config["last_model"] = model_name
        user_config["path_dict"][model_name] = model_path
    with open(get_config_path(), "w", encoding="utf-8") as f:
        json.dump(user_config, f, indent=2, ensure_ascii=False)

