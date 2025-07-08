from typing import Any, Dict, Generator, Optional

import gradio as gr
from gradio.components import Component  # cannot use TYPE_CHECKING here

from .chatter import WebChatModel
from .common import load_config
from .locales import LOCALES
from .manager import Manager


class Engine:
    def __init__(self, demo_mode: Optional[bool] = False, pure_chat: Optional[bool] = False) -> None:
        self.demo_mode = demo_mode
        self.pure_chat = pure_chat
        self.manager = Manager()
        self.chatter = WebChatModel(manager=self.manager, demo_mode=demo_mode, lazy_init=(not pure_chat))

    def _form_dict(self, resume_dict: Dict[str, Dict[str, Any]]):
        return {self.manager.get_elem_by_name(k): gr.update(**v) for k, v in resume_dict.items()}

    def resume(self) -> Generator[Dict[Component, Dict[str, Any]], None, None]:
        user_config = load_config() if not self.demo_mode else {}
        lang = user_config.get("lang", None) or "en"

        init_dict = {"top.lang": {"value": lang}, "infer.chat_box": {"visible": self.chatter.loaded}}
        
        yield self._form_dict(init_dict)

    def change_lang(self, lang: str) -> Dict[Component, Dict[str, Any]]:
        return {
            component: gr.update(**LOCALES[name][lang])
            for elems in self.manager.all_elems.values()
            for name, component in elems.items()
            if name in LOCALES
        }
