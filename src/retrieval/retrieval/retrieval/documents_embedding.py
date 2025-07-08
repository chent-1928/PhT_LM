from .config import *
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from .customize_exception import KBInconsistentError
from .utils import timer
def _embeddings_hash(self):
    return hash(self.model_name)

HuggingFaceEmbeddings.__hash__ = _embeddings_hash


class DocumentsEmbedding:
    embeddings: object = None
    embeddings_en: object = None
    chunk_conent: bool = False

    def __init__(self, embedding_model: str = EMBEDDING_MODEL,
                 embedding_model_en: str = EMBEDDING_MODEL_EN,
                 embedding_device=EMBEDDING_DEVICE,):
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODLE_DICT[embedding_model],
                                      model_kwargs={'device': embedding_device})
        self.embedding_model_en = HuggingFaceEmbeddings(model_name=EMBEDDING_MODLE_DICT[embedding_model_en],
                                      model_kwargs={'device': embedding_device})


    def _get_chunk_mapping(self,from_kb,res_from,fields:list[str]):
        id_mapping = from_kb.query_all(res_from)
        new_id_mapping,chunks, chunks_en = [],[],[]
        for map in id_mapping["source_docs"]:
            new_map,chunk,chunk_en = map,'',''
            # for field in fields:
            chunk += str(map['_source'][fields[0]]).replace('\n','')
            chunk_en += str(map['_source'][fields[1]]).replace('\n','')
            # print(f"用来编码向量的chunk为：{chunk}")
            new_map[VECTOR_FIELD_NAME] = chunk
            new_map[VECTOR_FIELD_NAME_EN] = chunk_en
            new_id_mapping.append(new_map)
            chunks.append(chunk)
            chunks_en.append(chunk_en)
        return new_id_mapping,chunks,chunks_en

    @timer
    def get_embeddings(self,from_kb,res_from,fields:list[str]=DEFAUT_FIELDS_VECTORUSED):
        mapping,chunks,chunks_en = self._get_chunk_mapping(from_kb,res_from,fields)
        # -----------------------------------------------------------------------
        self.embeddings = self.embedding_model.embed_documents(chunks)
        self.embeddings_en = self.embedding_model_en.embed_documents(chunks_en)
        embedding_mapping = []
        if len(mapping)==len(self.embeddings) and len(mapping)==len(self.embeddings_en):
            for i, map in enumerate(mapping):
                embedding_mapping.append({'_id':map['_id'],VECTOR_FIELD_NAME:self.embeddings[i],VECTOR_FIELD_NAME_EN:self.embeddings_en[i]})
        else:
            raise KBInconsistentError("文档编码前后不一致！")
        return embedding_mapping

    def query_embedding(self,query:str):
        query_embedding = self.embedding_model.embed_query(query)

        return query_embedding

    def query_embedding_en(self,query:str):
        query_embedding_en = self.embedding_model_en.embed_query(query)
    
        return query_embedding_en



