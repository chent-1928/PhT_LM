
from elasticsearch import helpers
from .utils import *
from .config import *
from .documents_embedding import DocumentsEmbedding
from .get_client import Client


class KB:
    """
    创建知识库
    """

    def __init__(self,kb_name: str):
        """
        :param kb_name: 知识库名称
        """
        self.kb_name = kb_name
        self.client = None

    def __len__(self):
        """
        获取知识库文档数量
        """
        pass
    def create_kb(self):
        """
        创建知识库
        """
        pass
    def delete_kb(self):
        """
        删除知识库
        """
        pass
    def clear_kb(self):
        """
        清空知识库
        """
        pass
    def get_kb_info(self):
        """
        查询知识库基本信息
        """
        pass
    def insert_bulk(self,file):
        """
        向知识库中批量插入文档
        """
        pass
    def insert_one(self,doc: dict):
        """
        向知识库中插入单个文档
        """
        pass

    def query_by_id(self,id):
        pass

    def query_all(self):
        pass

    def query_by_dsl(self,dsl):
        pass


class DocKB(KB):
    """
    创建文档知识库用于文档内容检索
    """
    def __init__(self,kb_name: str):
        super().__init__(kb_name)
        self.kb_name = kb_name
        self.client = Client().es_connect(ES_API,BASIC_AUTH)

    def __len__(self):
        es = self.client
        res = es.count(index=self.kb_name)
        return res['count']

    def create_kb(self, mappings=MAPPINGS):
        es = self.client
        index_name = self.kb_name
        if es.indices.exists(index=index_name):
            print(f'创建失败，{index_name}索引已存在！')
            raise Exception
        else:
            es.indices.create(index=index_name, settings=SETTINGS, mappings=mappings)
            print(f'{self.kb_name}创建成功！')

    def delete_kb(self):
        es = self.client
        if es.indices.exists(index=self.kb_name):
            es.indices.delete(index=self.kb_name)
            print(f'{self.kb_name}删除成功！')
        else:
            print(f'{self.kb_name}索引不存在！')
            raise Exception

    def clear_kb(self):
        es = self.client
        index_name = self.kb_name
        res = es.delete_by_query(index=index_name, body={"query": {"match_all": {}}})
        if res['deleted'] == 0:
            print('No documents deleted')
        else:
            print('Deleted %d documents' % res['deleted'])

    def get_kb_info(self):
        es = self.client
        mapping = es.indices.get_mapping(index=self.kb_name)[self.kb_name]['mappings']
        setting = es.indices.get_settings(index=self.kb_name)[self.kb_name]['settings']
        res = {
           'mapping': mapping,
           'setting': setting
        }
        print(res)
    
    @timer
    def insert_bulk(self,file):
        es = self.client
        if es.indices.exists(index=self.kb_name) == False:
            self.create_kb()
        docs = excel_parser(file)
        action = ({
            '_index': self.kb_name,
            '_source': doc
        } for doc in docs)
        helpers.bulk(es, action)
        print(f'{len(docs)}条文档插入完成！')

    def insert_one(self,doc):
        es = self.client
        res = es.index(index=self.kb_name, body=doc)
        return res["_id"]
        # print(f'插入成功！id为{res["_id"]}')

    def query_by_id(self,id) -> dict:
        es = self.client
        res = es.get(index=self.kb_name, id=id)
        return res

    def query_all(self,res_from=0,res_size=100000):
        es = self.client
        res = es.search(index=self.kb_name,body={"from":res_from,"size":res_size,"query": {"match_all": {}}})
        hits = result_transfer(res)
        return hits

    def query_by_dsl(self,dsl):
        es = self.client
        res = es.search(index=self.kb_name,body=dsl)
        resp = result_transfer(res)
        return resp


class ESVectorKB(KB):
    def __init__(self,kb_name: str):
        super().__init__(kb_name)
        self.kb_name = kb_name
        self.client = Client().es_connect(ES_API,BASIC_AUTH)

    def __len__(self):
        es = self.client
        res = es.count(index=self.kb_name)
        return res['count']

    def create_kb(self):
        es = self.client
        if es.indices.exists(index=self.kb_name):
            print(f'创建失败，{self.kb_name}索引已存在！')
            raise Exception
        else:
            es.indices.create(index=self.kb_name, settings=SETTINGS,mappings=VECTOR_MAPPINGS)
            print(f'{self.kb_name}创建成功！')

    def delete_kb(self):
        es = self.client
        if es.indices.exists(index=self.kb_name):
            es.indices.delete(index=self.kb_name)
            print(f'{self.kb_name}删除成功！')
        else:
            print(f'{self.kb_name}索引不存在！')
            raise Exception

    def clear_kb(self):
        es = self.client
        res = es.delete_by_query(index=self.kb_name, body={"query": {"match_all": {}}})
        if res['deleted'] == 0:
            print('No documents deleted')
        else:
            print('Deleted %d documents' % res['deleted'])

    def get_kb_info(self):
        es = self.client
        mapping = es.indices.get_mapping(index=self.kb_name)[self.kb_name]['mappings']
        setting = es.indices.get_settings(index=self.kb_name)[self.kb_name]['settings']
        res = {
          'mapping': mapping,
          'setting': setting
        }
        print(res)

    @timer
    def insert_bulk(self, from_kb:KB, fields=DEFAUT_FIELDS_VECTORUSED):
        res_from = len(self)
        print(f'res_from: {res_from}')
        mapping = DocumentsEmbedding().get_embeddings(from_kb,res_from,fields)
        cnt = 0
        for map in mapping:
            id = map['_id']
            doc = {VECTOR_FIELD_NAME:map[VECTOR_FIELD_NAME], VECTOR_FIELD_NAME_EN: map[VECTOR_FIELD_NAME_EN]}
            # doc = {VECTOR_FIELD_NAME:map[VECTOR_FIELD_NAME]}
            self.insert_one(doc,id)
            cnt += 1
        print(f'{cnt} 条文档向量插入完成！')

    def insert_one(self,doc,id):
        es = self.client
        es.index(index=self.kb_name,id=id,body=doc)

    def query_by_dsl(self, dsl, from_kb:KB):
        es = self.client
        res = es.search(index=self.kb_name, body=dsl)
        # print(res)
        resp = vec_result_transfer(res,from_kb)
        return resp
