import deepl
import csv

def translation(strs, language):

    auth_key = "469f3ab8-eec8-49ee-8292-1661dccd0fed:fx"  # use DeepL free API 
    target_language = language      #"EN-US"
    #调用deepl
    translator = deepl.Translator(auth_key)  #input the auth_key
    #print(time.time())
    result = translator.translate_text(strs,target_lang=target_language)
    return result.text


from nltk.translate.bleu_score import sentence_bleu
import jieba
import json
import pandas as pd
import requests
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf


smooth = SmoothingFunction()

files = [
    "data/results_test_en_2_zh_without_context.json", "data/results_test_zh_2_en_without_context.json"
    ]


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

    new_datas = []
    for m,data in enumerate(datas):
        predict = translation(data["instruction"], "ZH" if "en_2_zh" in file else "EN-US")
        if "en_2_zh" in file:
            pre = jieba.lcut(predict, cut_all=False)
            output = jieba.lcut(data['output'], cut_all=False)
        else:
            pre = predict.split(' ')
            output = data['output'].split(' ')

        total += sentence_chrf(output, pre)
        [score_1, score_2, score_3, score_4] = sentence_bleu([output], pre, weights = [(1,), (1/2, 1/2), (1/3, 1/3, 1/3), (0.25, 0.25, 0.25, 0.25)])
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