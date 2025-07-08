
from .kb import DocKB, ESVectorKB
from .search_dsl import SearchDSL
# from .config import QUERY_FIELDS, KB_NAME, VEC_NAME, GROUP_QUERY_FIELDS, ANS_KB_NAME
from .config import QUERY_FIELDS, KB_NAME, VEC_NAME

class Query:
    def __init__(self, query, is_zh):
        self.query = query
        self.kb_name = KB_NAME
        # self.ans_kb_name = ANS_KB_NAME
        self.vec_name = VEC_NAME
        self.query_fields =  QUERY_FIELDS[0] if is_zh else QUERY_FIELDS[1]
        # self.id_query_fields = GROUP_QUERY_FIELDS

    def doc_retrieval(self, res_from=0, res_size=10):
        dsl = SearchDSL(self.query,res_from, res_size).multi_match([self.query_fields])
        resp = DocKB(self.kb_name).query_by_dsl(dsl)
        return resp
    
    def doc_retrieval_by_id(self, id):
        resp = DocKB(self.kb_name).query_by_id(id)
        return resp

    def vec_retrieval(self, is_zh, res_from=0, res_size=10):
        dsl = SearchDSL(self.query, res_from, res_size).vector_search_cos(is_zh)
        veckb, dockb = ESVectorKB(self.vec_name), DocKB(self.kb_name)
        resp = veckb.query_by_dsl(dsl, dockb)
        return resp
    
    def group_retrieval(self, question_id: str):
        dsl = SearchDSL(query=question_id, res_from=0, res_size=1).multi_match(self.id_query_fields) # 关键字精准查询
        resp = DocKB(self.ans_kb_name).query_by_dsl(dsl)
        return resp

