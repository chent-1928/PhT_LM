import re
import csv
import html
from urllib import parse
import requests

from nltk.translate.bleu_score import sentence_bleu
import jieba
import json
import requests
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf
 
GOOGLE_TRANSLATE_URL = 'http://translate.google.com/m?q=%s&tl=%s&sl=%s'
smooth = SmoothingFunction()
files = [
    "data/results_test_en_2_zh_without_context.json", "data/results_test_zh_2_en_without_context.json"
    ]
 
def translate(text, to_language="auto", text_language="auto"):
    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text,to_language,text_language)
    response = requests.get(url)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if (len(result) == 0):
        return ""
 
    return html.unescape(result[0])


for file in files:
    total, total_1, total_2, total_3, total_4, total_5 = 0,0,0,0,0,0
    with open(file, encoding='utf-8', mode='r') as f:
        datas = json.load(f)

    # datas = []
    # with open(file, newline='', encoding='utf-8') as csvfile:
    #     reader = csv.reader(csvfile)
    #     header = next(reader)
    #     for row in reader:
    #         if len(row) < 2:
    #             continue  # 跳过不完整的行
    #         instruction, output = row[0], row[1]
    #         datas.append({
    #             "instruction": instruction,
    #             "output": output
    #         })
    # print(len(datas))

    # new_datas = []
    for m,data in enumerate(datas):
        print(m)
        predict = translate(data["instruction"], "zh-CN" if "en_2_zh" in file else "en", "en" if "en_2_zh" in file else "zh-CN")
        # predict = translate(data["instruction"], "zh-CN" if "en2zh" in file else "en", "en" if "en2zh" in file else "zh-CN")

        # if "en_2_zh" in file:
        if "en2zh" in file:
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
    print("总chrf：", total)
    print("平均chrf：", total / len(datas))
    print("总bleu分数：", total_1, total_2, total_3, total_4)
    print("测试集个数：", len(datas))
    print("平均bleu分数：", total_1 / len(datas), total_2 / len(datas), total_3 / len(datas), total_4 / len(datas))
    print(total_5 / len(datas))
    print('---------------------------------------------------------------------------------------------------------------------------------')