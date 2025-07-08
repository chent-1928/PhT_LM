from typing import TYPE_CHECKING, Dict, Optional, Tuple

import gradio as gr
from ...extras.constants import EN_2_ZH, ZH_2_EN, DIRECTION, ES_AND_VEC, ES, RETRIEVAL_MODE


if TYPE_CHECKING:
    from gradio.blocks import Block
    from gradio.components import Component

    from ..engine import Engine


def create_chat_box(
    engine: "Engine", visible: Optional[bool] = False
) -> Tuple["Block", "Component", "Component", Dict[str, "Component"]]:
    with gr.Box(visible=visible) as chat_box:
        chatbot = gr.Chatbot()
        messages = gr.State([])
        with gr.Row():
            with gr.Column(scale=4):
                direction = gr.Dropdown(label=DIRECTION, choices=[EN_2_ZH, ZH_2_EN], value=EN_2_ZH)
                only_es = gr.Dropdown(label=RETRIEVAL_MODE, choices=[ES_AND_VEC, ES], value=ES_AND_VEC)
                query = gr.Textbox(show_label=False, lines=8)
                submit_btn = gr.Button(variant="primary")
                clear_btn = gr.Button()

            # with gr.Column(scale=1):
            #     clear_btn = gr.Button()
            #     gen_kwargs = engine.chatter.generating_args

    submit_btn.click(
        engine.chatter.predict,
        [chatbot, query, messages, direction, only_es],
        [chatbot, messages],
        show_progress=True,
    ).then(lambda: gr.update(value=""), outputs=[query])

    clear_btn.click(lambda: ([], []), outputs=[chatbot, messages], show_progress=True)

    return (
        chat_box,
        chatbot,
        messages,
        dict(
            query=query,
            submit_btn=submit_btn,
            clear_btn=clear_btn,
        ),
    )
