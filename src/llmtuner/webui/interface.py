import gradio as gr
from transformers.utils.versions import require_version

from .common import save_config
from .components import (
    create_chat_box,
)
from .css import CSS
from .engine import Engine


require_version("gradio>=3.38.0,<4.0.0", 'To fix: pip install "gradio>=3.38.0,<4.0.0"')


def create_web_demo() -> gr.Blocks:
    engine = Engine(pure_chat=True)

    with gr.Blocks(title="Web Demo", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "zh"])
        engine.manager.all_elems["top"] = dict(lang=lang)

        chat_box, _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.all_elems["infer"] = dict(chat_box=chat_box, **chat_elems)

        demo.load(engine.resume, outputs=engine.manager.list_elems())
        lang.change(engine.change_lang, [lang], engine.manager.list_elems(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


if __name__ == "__main__":
    demo = create_web_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)
