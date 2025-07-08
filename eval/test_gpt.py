# encoding: utf-8
import jieba
import openai
import json
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf


smooth = SmoothingFunction()
openai.api_base = "https://api.chatanywhere.tech"
openai.api_key = "sk-xxx"

def send_request(system_prompt, prompt):
        response = openai.ChatCompletion.create(
            messages=[
                {"role": "system",
                 "content": system_prompt},
                {"role": "user",
                 "content": prompt}],
            model="gpt-3.5-turbo",
            max_tokens=500,
            temperature=0.7,  # 决定输出的随机性，这个值越大，模型输出越随机
            stream=False,
        )
        return response["choices"][0]["message"]["content"]


files = [
    "data/test_en_2_zh_without_context.json", 
    "data/test_zh_2_en_without_context.json",
]
for file in files:
    total, total_1, total_2, total_3, total_4 = 0, 0, 0, 0, 0
    with open(file, encoding='utf-8', mode='r') as f:
        datas = json.load(f)

    new_datas = []
    for m, data in enumerate(datas):
        predict = send_request("你是一个药学领域的语言专家", data['instruction'])
        if "en_2_zh" in file:
            pre = jieba.lcut(predict, cut_all=False)
            output = jieba.lcut(data['output'], cut_all=False)
        else:
            pre = predict.split(' ')
            output = data['output'].split(' ')

        total += sentence_chrf(output, pre)
        [score_1, score_2, score_3, score_4] = sentence_bleu([output], pre,
                                                             weights=[(1,), (1 / 2, 1 / 2), (1 / 3, 1 / 3, 1 / 3),
                                                                      (0.25, 0.25, 0.25, 0.25)], smoothing_function=smooth.method4)

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

