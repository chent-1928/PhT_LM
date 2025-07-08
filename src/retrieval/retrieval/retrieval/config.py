import torch

# ======================== es cinfig =======================
ES_API = "http://IP:9200"
BASIC_AUTH = ("username", "password")

# ======================== create knowledge base cinfig =======================

MAPPINGS = {
    "properties": {
        "zh_text": {"type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart_synonym"},
        "en_text": {"type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart_synonym"}
    }
}

GROUP_MAPPINGS = {
  "properties": {
        "standard_question_id": {"type": "keyword"},
        "similar_question_id": {"type": "keyword"},
        "answer": {"type": "keyword"},
    }
}

SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
        "filter": {
          "my_synonym_filter": {
            "type": "synonym",
            "synonyms_path": "analysis/synonyms.txt"
          }
        },
        "analyzer": {
          "ik_max_custom": {
            "filter": [
              "my_synonym_filter"
            ],
            "tokenizer": "ik_max_word"
          },
          "ik_smart_synonym": {
            "type": "custom",
            "tokenizer": "ik_smart",
            "filter": [
              "my_synonym_filter"
            ]
          }
        }
      }
}

# 文档块编码为向量时需要的字段
DEFAUT_FIELDS_VECTORUSED = ['zh_text', 'en_text']
# 向量库的向量字段名称
VECTOR_FIELD_NAME = "chunk_vector"
VECTOR_FIELD_NAME_EN = "chunk_vector_en"
VECTOR_MAPPINGS = {
    "properties": {
      VECTOR_FIELD_NAME: {"type": "dense_vector","dims": 1792},
      VECTOR_FIELD_NAME_EN: {"type": "dense_vector","dims": 1024}
    }
  }

# ====================== embedding model config =====================
import os

current_dir = os.getcwd()

EMBEDDING_MODLE_DICT = {
        "stella_base_zh_v3_1792d": current_dir + "/model/stella_base_zh_v3_1792d",
        "mxbai-embed-large-v1": current_dir + "/model/mxbai-embed-large-v1",
}

# 嵌入模型
EMBEDDING_MODEL = "stella_base_zh_v3_1792d"
EMBEDDING_MODEL_EN = "mxbai-embed-large-v1"
# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================== query cinfig =======================
KB_NAME = 'translation_kb'
VEC_NAME = 'translation_veckb'
QUERY_FIELDS = ['zh_text', 'en_text'] # 查询的字段
