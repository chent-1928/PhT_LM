from retrieval.util import format_prompt
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Sequence, Tuple
from ..extras.constants import TOP_P, TEMPERATURE, MAX_TOKENS, ZH_2_EN, ES
from ..chat import ChatModel
from ..data import Role
from ..hparams import GeneratingArguments
from ..extras.misc import torch_gc
from ..extras.logging import get_logger

if TYPE_CHECKING:
    from .manager import Manager

logger = get_logger(__name__)

class WebChatModel(ChatModel):
    def __init__(
        self, manager: "Manager", demo_mode: Optional[bool] = False, lazy_init: Optional[bool] = True
    ) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        self.model = None
        self.tokenizer = None
        self.generating_args = GeneratingArguments()

        if not lazy_init:  # read arguments from command line
            super().__init__()


    @property
    def loaded(self) -> bool:
        return self.model is not None


    async def predict(
        self,
        chatbot: List[Tuple[str, str]],
        query: str,
        messages: Sequence[Tuple[str, str]],
        direction: str,
        retrieval_mode: str,
        topk: int=4,
        fusion_weight: float=0.5,
        max_new_tokens: int=MAX_TOKENS,
        top_p: float=TOP_P,
        temperature: float=TEMPERATURE
    ) -> Generator[Tuple[Sequence[Tuple[str, str]], Sequence[Tuple[str, str]]], None, None]:
        chatbot.append([query, ""])
        
        # prompt = self.find_vectore(query, kb_name)
        # 设置
        prompt = await format_prompt(query, direction == ZH_2_EN, topk, fusion_weight, retrieval_mode == ES)
        query_messages = [{"role": Role.USER, "content": prompt}]
        response = ""       
        for new_text in self.stream_chat(
            query_messages, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature
        ):
            response += new_text
            result = response

            output_messages = query_messages + [{"role": Role.ASSISTANT, "content": result}]
            bot_text = result

            chatbot[-1] = [query, self.postprocess(bot_text)]
            yield chatbot, output_messages
        torch_gc()


    def postprocess(self, response: str) -> str:
        blocks = response.split("```")
        for i, block in enumerate(blocks):
            if i % 2 == 0:
                blocks[i] = block.replace("<", "&lt;").replace(">", "&gt;")
        return "```".join(blocks)
