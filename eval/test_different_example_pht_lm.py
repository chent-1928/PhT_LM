from nltk.translate.bleu_score import sentence_bleu
import jieba
import json
import pandas as pd
import requests
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf


smooth = SmoothingFunction()
url = "http://10.4.0.141:8000/chat/test"

files = [
    "data/test_en_2_zh_1.json", "data/test_zh_2_en_1.json",
    "data/test_en_2_zh_2.json", "data/test_zh_2_en_2.json",
    "data/test_en_2_zh.json", "data/test_zh_2_en.json",
    "data/test_en_2_zh_8.json", "data/test_zh_2_en_8.json",
    "data/test_en_2_zh_16.json", "data/test_zh_2_en_16.json",
    "data/test_en_2_zh_es.json", "data/test_zh_2_en_es.json",
    "data/test_en_2_zh_vec.json", "data/test_zh_2_en_vec.json"
    ]


def request_api(url, request_body):
    response = requests.post(url, json=request_body)  # 使用GET请求示例

    if response.status_code == 200:  # 判断响应状态码为200表示成功
        resp = json.loads(response.content)
        return resp['content']
        # return resp['choices'][0]['message']['content'].strip("\n").strip(" ")
    else:
        print("Error occurred while calling the API.")
        return None


for file in files:
    total, total_1, total_2, total_3, total_4, total_5 = 0,0,0,0,0,0
    with open(file, encoding='utf-8', mode='r') as f:
        datas = json.load(f)

    new_datas = []
    for m,data in enumerate(datas):
        request_body =  {
                "model": 'str',
                "messages": [
                    {
                        "role": "user",
                        "content": data["instruction"]
                    }
                ]
            }
        predict = request_api(url, request_body)
        if "test_en_2_zh" in file:
            pre = jieba.lcut(predict, cut_all=False)
            output = jieba.lcut(data['output'], cut_all=False)
        else:
            pre = predict.split(' ')
            output = data['output'].split(' ')

        total += sentence_chrf(output, pre)
        [score_1, score_2, score_3, score_4] = sentence_bleu([output], pre, weights = [(1,), (1/2, 1/2), (1/3, 1/3, 1/3), (0.25, 0.25, 0.25, 0.25)], smoothing_function=smooth.method4)
        score_5 = sentence_bleu([output], pre, smoothing_function=smooth.method4)
        total_5 += score_5
        total_1 += score_1
        total_2 += score_2
        total_3 += score_3
        total_4 += score_4

    print(file + '----------------------------------------------------------------------------------------------------------------')
    # print("总chrf：", total)
    print("平均chrf：", total / len(datas))
    # print("总bleu分数：", total_1, total_2, total_3, total_4)
    # print("测试集个数：", len(datas))
    print("平均bleu分数：", total_1 / len(datas), total_2 / len(datas), total_3 / len(datas), total_4 / len(datas))
    # print(total_5 / len(datas))
    print('---------------------------------------------------------------------------------------------------------------------------------')