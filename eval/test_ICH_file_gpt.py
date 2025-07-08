# encoding: utf-8
import jieba
import openai
import json
import time
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf

# sk-mgJ1aFGM84tjWEXv4OXZDNfbb1blofZ5qaoxX7z5ipgXLIIF
smooth = SmoothingFunction()
openai.api_base = "https://api.chatanywhere.tech"
openai.api_key = "sk-TkMY0C7ptbwu8Q1QvPD9qD2bGt8nsIqRvB3AymIYf2YH0L5G"

def send_request(system_prompt, prompt, is_zh):
        text = f"请将'{prompt}'翻译为英文，只能输出翻译后的句子，不允许添加编造成分。" if is_zh else f"请将'{prompt}'翻译为中文，只能输出翻译后的句子，不允许添加编造成分。"
        response = openai.ChatCompletion.create(
            messages=[
                {"role": "system",
                 "content": system_prompt},
                {"role": "user",
                 "content": text}],
            model="gpt-3.5-turbo",
            max_tokens=500,
            temperature=0.7,  # 决定输出的随机性，这个值越大，模型输出越随机
            stream=False,
        )
        # print(response["choices"][0]["message"]["content"])

        return response["choices"][0]["message"]["content"]


files = [
    "data/ICH_file_en_2_zh.json", 
    "data/ICH_file_zh_2_en.json",
]
for file in files:
    total, total_1, total_2, total_3, total_4 = 0, 0, 0, 0, 0
    with open(file, encoding='utf-8', mode='r') as f:
        datas = json.load(f)

    new_datas = []
    for m, data in enumerate(datas):
        data['instruction'] = data['instruction'].lower()
        data['output'] = data['output'].lower()
        
        predict = send_request("你是一个乐于助人的助手", data['instruction'], False if "en_2_zh" in file else True)
        if "en_2_zh" in file:
            pre = jieba.lcut(predict, cut_all=False)
            output = jieba.lcut(data['output'], cut_all=False)
        else:
            pre = predict.split(' ')
            output = data['output'].split(' ')

        total += sentence_chrf(output, pre)
        [score_1, score_2, score_3, score_4] = sentence_bleu([output], pre,
                                                             weights=[(1,), (1 / 2, 1 / 2), (1 / 3, 1 / 3, 1 / 3),
                                                                      (0.25, 0.25, 0.25, 0.25)])

        total_1 += score_1
        total_2 += score_2
        total_3 += score_3
        total_4 += score_4

    print(file + '---------------------------------------------------------------------------------------')
    # print("总chrf：", total)
    print("平均chrf：", 100 * total / len(datas))
    # print("总bleu分数：", total_1, total_2, total_3, total_4)
    # print("测试集个数：", len(datas))
    print("平均bleu分数：", 100 * total_1 / len(datas), 100 * total_2 / len(datas), 100 * total_3 / len(datas), 100 * total_4 / len(datas))
    print('---------------------------------------------------------------------------------------')

