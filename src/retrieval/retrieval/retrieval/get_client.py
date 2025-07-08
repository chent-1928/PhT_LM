from elasticsearch import Elasticsearch

class Client:
    __es_conn_pool = None
    __esutil_instance__ = None

    def __new__(cls, *args, **kwargs):
        if not cls.__esutil_instance__:
            cls.__esutil_instance__ = super().__new__(cls)
        return cls.__esutil_instance__

    def __init__(self):
        pass

    def es_connect(self, api: str, auth_info: tuple):
        if self.__es_conn_pool is None:
            self.__es_conn_pool = Elasticsearch(api, basic_auth=auth_info, timeout=3600,
                                                maxsize=10)
        return self.__es_conn_pool
