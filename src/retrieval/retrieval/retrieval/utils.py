import time
import pandas as pd
import re

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print('{}共耗时约 {:.2f} 秒'.format(func, time.time() - start))
        return res
    return wrapper

def replace_quote(value):
    value = str(value)
    pattern = re.compile(r'"(.*?)"')
    result = pattern.findall(value)
    for l in result:
        value = value.replace('"{}"'.format(l), '“{}”'.format(l))
    return value

def excel_parser(file):
    df = pd.read_excel(file, sheet_name=None)
    docs = []
    for sheet in df.keys():
        df_sheet = df[sheet].fillna('')
        # for col in df_sheet.columns:
        #     df_sheet[col] = df_sheet[col].apply(replace_quote)
        docs += df_sheet.to_dict(orient='records')
    return docs

def result_transfer(result) -> dict:
    source_docs = []
    for hit in result['hits']['hits']:
        id = hit['_id']
        score = hit['_score']
        source = hit['_source']
        # for field in source.keys():
            # source[field] = str(source[field]).replace('"', '&&temp&&').replace('\\', '@@temp@@')
        source_docs.append({"_id":id,"_score":score,"_source":source})
    # resp = {"total":result['hits']['total']['value'],"source_docs":source_docs}
    resp = {"total": 100, "source_docs": source_docs}
    return resp

def vec_result_transfer(result,from_kb):
    source_docs = []
    for hit in result['hits']['hits']:
        id = hit['_id']
        score = hit['_score']
        source =  from_kb.query_by_id(id)['_source']
        # for field in source.keys():
        #     source[field] = str(source[field]).replace('"', '&&temp&&').replace('\\', '@@temp@@')
        source_docs.append({"_id":id,"_score":score,"_source":source})
    resp = {"total": 100, "source_docs": source_docs}
    return resp


