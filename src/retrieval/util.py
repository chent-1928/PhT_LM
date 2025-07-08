import json
import pandas as pd
from .pair_data import PairData


async def format_prompt(text, is_zh, topk, fusion_weight, is_es):
    f = PairData(text, is_zh=is_zh, is_es=is_es)
    results = f.get_weight_fusion_resp(text, topk, fusion_weight)

    if not results:
        # result为空时：
        if is_zh:
            prompt = "请将'{sentence}'翻译为英文，只能输出翻译后的句子，不允许添加编造成分。" 
        else:
            prompt = "请将'{sentence}'翻译为中文，只能输出翻译后的句子，不允许添加编造成分。" 
        prompt = prompt.replace("{sentence}", text)
    else:
        context = ""
        if is_zh:
            prompt = '''已知信息：
{context}
根据上述已知信息，请将'{sentence}'翻译为英文，只能输出翻译后的句子，不允许添加编造成分。'''
            for result in results:
                context += f"'{result['_source']['zh_text']}'可被翻译为：{result['_source']['en_text']}\n"
        else:
            prompt = '''已知信息：
{context}
根据上述已知信息，请将'{sentence}'翻译为中文，只能输出翻译后的句子，不允许添加编造成分。'''
            for result in results:
                context += f"'{result['_source']['en_text']}'可被翻译为：{result['_source']['zh_text']}\n"

        prompt = prompt.replace("{context}", context).replace("{sentence}", text)
    # print(prompt)
    return prompt

