from .documents_embedding import DocumentsEmbedding
from .utils import *

class SearchDSL:
    def __init__(self, query:str,res_from:int=0,res_size:int=10) -> None:
        if not query:
            raise Exception("query is empty")
        if not isinstance(query,str):
            raise Exception("query must be a str")
        self.res_from = res_from
        self.res_size = res_size
        self.query = query

    def match(self,fields:[str]):
        '''
        模糊匹配，单字段
        '''
        if not fields:
            raise Exception("fields is empty")
        if not isinstance(fields,list):
            raise Exception("fields must be a list")
        field = fields[0]
        body = {
            "query": {
                "match": {
                    field: self.query
                }
            },
            "from": self.res_from,
            "size": self.res_size
        }
        return body

    def term(self,fields:[str]):
        '''
        精确匹配，单字段
        '''
        if not fields:
            raise Exception("fields is empty")
        if not isinstance(fields,list):
            raise Exception("fields must be a list")
        field = fields[0]
        body = {
            "query": {
                "term": {
                    field: self.query
                }
            },
            "from": self.res_from,
            "size": self.res_size
        }
        return body

    def match_phrase(self,fields:[str]):
        '''
        匹配短语，单字段
        '''
        if not fields:
            raise Exception("fields is empty")
        if not isinstance(fields,list):
            raise Exception("fields must be a list")
        field = fields[0]
        body = {
            "query": {
                "match_phrase": {
                    field: self.query
                }
            }
        }
        return body

    def multi_match(self,fields:list[str]):
        '''
        多字段查询
        '''
        if not fields:
            raise Exception("fields is empty")
        if not isinstance(fields,list):
            raise Exception("fields must be a list")
        body = {
            "query": {
                "multi_match": {
                    "query": self.query,
                    "fields": fields
                }
            },
            "from": self.res_from,
            "size": self.res_size
        }
        return body

    def vector_search_cos(self, is_zh):
        '''
        向量搜索，余弦相似度
        '''
        if is_zh:
            query_vector = DocumentsEmbedding().query_embedding(self.query)
            body = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector,'chunk_vector')+1.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            },
            "from": self.res_from,
            "size": self.res_size
        }
        else:
            query_vector = DocumentsEmbedding().query_embedding_en(self.query)
            body = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector,'chunk_vector_en')+1.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            },
            "from": self.res_from,
            "size": self.res_size
        }
        
        return body

    def vector_search_dotproduct(self):
        '''
        向量搜索，点积
        '''
        query_vector = DocumentsEmbedding().query_embedding(self.query)
        body = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source":"dotProduct(params.query_vector,'chunk_vector')+1.0",
                        # "source":"double value = dotProduct(params.query_vector,'chunk_vector');return sigmoid(1, Math.E, -value);",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            },
            "from": self.res_from,
            "size": self.res_size
        }
        return body

    def vector_search_l1(self):
        '''
        向量搜索，L1距离
        '''
        query_vector = DocumentsEmbedding().query_embedding(self.query)
        body = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "1 / (1 + l1norm(params.query_vector, 'chunk_vector'))",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            },
            "from": self.res_from,
            "size": self.res_size
        }
        return body

    def vector_search_l2(self):
        '''
        向量搜索，L2距离
        '''
        query_vector = DocumentsEmbedding().query_embedding(self.query)
        body = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "1 / (1 + l2norm(params.query_vector, 'chunk_vector'))",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            },
            "from": self.res_from,
            "size": self.res_size
        }
        return body














