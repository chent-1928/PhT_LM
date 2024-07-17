from .retrieval.retrieval.query import Query


class PairData:
    def __init__(self, question, is_zh=True, is_es=False) -> None:
        self.query = Query(question, is_zh)
        self.is_zh = is_zh
        self.is_es = is_es

    def _get_resp(self):
        if self.is_es:
            return self.query.doc_retrieval()['source_docs'], [] 
        else:
            return self.query.doc_retrieval()['source_docs'], self.query.vec_retrieval(self.is_zh)['source_docs']
    
    def get_weight_fusion_resp(self, query, topk, fusion_weight):
        doc_resp, vec_resp = self._get_resp()
        if not doc_resp and not vec_resp:
            return []
        
        # 测试集要去除和query一样的句子。训练完模型后和模型合并时需要删除：
        # for i, res in enumerate(doc_resp):
        #     if res['_source']['zh_text'] == query or res['_source']['en_text'] == query:
        #         del doc_resp[i]
        # for i, res in enumerate(vec_resp):
        #     if res['_source']['zh_text'] == query or res['_source']['en_text'] == query:
        #         del vec_resp[i]

        # vec_resp = []
        if not doc_resp:
            if len(vec_resp) > topk:
                return vec_resp[:topk]
            else:
                return vec_resp
        if not vec_resp:
            if len(doc_resp) > topk:
                return doc_resp[:topk]
            else:
                return doc_resp
        text_to_source = {}
        vec_k = round(topk * fusion_weight)

        print("vec_k：", vec_k)
        if len(vec_resp) > vec_k:
            for vec_sour in vec_resp[:vec_k]:
                text_to_source[vec_sour['_source']['zh_text'] + vec_sour['_source']['en_text']] = vec_sour
            i = len(text_to_source)
            for doc_sour in doc_resp:
                if i < topk:
                    text_to_source[doc_sour['_source']['zh_text'] + doc_sour['_source']['en_text']] = doc_sour
                    i = len(text_to_source)
            for vec_sour in vec_resp[vec_k:]:
                text_to_source[vec_sour['_source']['zh_text'] + vec_sour['_source']['en_text']] = vec_sour

        else:
            for doc_sour in doc_resp:
                text_to_source[doc_sour['_source']['zh_text'] + doc_sour['_source']['en_text']] = doc_sour

        resp = list(text_to_source.values())[:topk]
        return resp