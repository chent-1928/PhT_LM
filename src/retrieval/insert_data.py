import os

from retrieval.retrieval.kb import DocKB, ESVectorKB
from retrieval.retrieval.config import KB_NAME, VEC_NAME
from retrieval.retrieval.utils import excel_parser

question_kb = DocKB(KB_NAME)
question_veckb = ESVectorKB(VEC_NAME)

def delete_kb():
    question_kb.delete_kb()
    question_veckb.delete_kb()

def create_kb():
    question_kb.create_kb()
    question_veckb.create_kb()

def clear_kb():
    question_kb.clear_kb()
    question_veckb.clear_kb()

def insert_translation_data():
    current_dir = os.getcwd() 
    xlsx_path = current_dir + "/src/retrieval/data/translation_data.xlsx"
    datas = excel_parser(xlsx_path)
    for data in datas:
        id = question_kb.insert_one({'zh_text': data['Chinese'], 'en_text': data['English']})

def insert_veckb():
    question_veckb.insert_bulk(from_kb=question_kb)

if __name__ == '__main__':
    # clear_kb()
    create_kb()
    insert_translation_data()
    insert_veckb()