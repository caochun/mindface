import os
import logging
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch
from typing import List, Dict, Optional, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunningCode(Enum):
    ES_ERROR = -1
    SUCCESS = 0
    MISSING_PARAMS = 1001
    MISSING_FILE = 1002
    ID_EXISTS = 1003
    ID_NOT_FOUND = 1004
    FACE_STORE_EXISTS = 1005
    FACE_STORE_NOT_EXISTS = 1006
    NO_MATCHED_FACE_DETECTED = 1007
    POOR_FACE_QUALITY = 1008

class FaceEmbeddingES:
    def __init__(self, host='localhost', port=9200, embedding_dim=512,
                 use_auth=True, username=None, password=None):
        es_config = f"http://{host}:{port}"
        if use_auth:
            username = username or os.getenv("ES_USERNAME", "elastic")
            password = password or os.getenv("ES_PASSWORD", "Si5KbfjF")
            if username and password:
                self.es = Elasticsearch(es_config, basic_auth=(username, password))
            else:
                logger.warning("认证信息不完整，尝试无认证连接")
                self.es = Elasticsearch(es_config)
        else:
            self.es = Elasticsearch(es_config)

        self.embedding_dim = embedding_dim

        try:
            if self.es.ping():
                logger.info("成功连接到 ElasticSearch")
            else:
                logger.error("无法连接到 ElasticSearch")
                raise ConnectionError("ES 连接失败")
        except Exception as e:
            logger.error(f"ES 连接错误: {e}")
            raise
    def _check_face_store_exists(self, reponame: str) -> bool:
        try:
            exists = self.es.indices.exists(index=reponame)
            return exists
        except Exception as e:
            logger.error(f"检索人脸库失败，请检查ES连接: {e}")
            return False
    def _check_face_exists(self, identity: str, reponame: str) -> bool:
        try:
            exists = self.es.exists(index=reponame, id=identity)
            return exists
        except Exception as e:
            logger.error(f"检索人脸信息失败，请检查ES连接: {e}")
            return False
    def create_face_store(self, reponame: str) -> RunningCode:
        try:
            if self._check_face_store_exists(reponame):
                return RunningCode.FACE_STORE_EXISTS
            
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {
                            "type": "keyword"
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embedding_dim,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "timestamp": {
                            "type": "date"
                        },
                        "metadata": {
                            "type": "object",
                            "enabled": False
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
            
            response = self.es.indices.create(index=reponame, body=mapping)
            logger.info(f"成功创建人脸库: {reponame}")
            return RunningCode.SUCCESS
        except Exception as e:
            logger.error(f"创建人脸库失败，请检查ES连接: {e}")
            return RunningCode.ES_ERROR
    def register_face(self, identity: str, embedding: np.ndarray, reponame: str) -> RunningCode:
        try:
            if not self._check_face_store_exists(reponame):
                return RunningCode.FACE_STORE_NOT_EXISTS
            if self._check_face_exists(identity, reponame):
                return RunningCode.ID_EXISTS
            doc = {
                "id": identity,
                "embedding": embedding.flatten().tolist(),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "embedding_shape": embedding.shape,
                    "embedding_norm": float(np.linalg.norm(embedding))
                }
            }
            response = self.es.index(
                index=reponame,
                id=identity,
                body=doc,
                refresh=True
            )
            logger.info(f"成功注册人脸: {identity} -> {reponame}")
            return RunningCode.SUCCESS
        except Exception as e:
            logger.error(f"注册人脸失败: {e}")
            return RunningCode.ES_ERROR
    def search_similar_faces(self, query_embedding: np.ndarray, reponame: str, 
                           k: int = 10, threshold: float = 0.9) -> Tuple[RunningCode, List[Dict]]:
        try:
            if not self._check_face_store_exists(reponame):
                return (RunningCode.FACE_STORE_NOT_EXISTS, [])
            
            search_body = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding.flatten().tolist(),
                    "k": k,
                    "num_candidates": min(k * 10, 100)
                },
                "_source": ["id", "timestamp", "metadata"],
                "size": k
            }
            response = self.es.search(index=reponame, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                # ES 的 cosine 相似度分数转换
                raw_score = hit['_score']
                # ES 返回的是 (1 + cosine_similarity) / 2，需要转换回来
                cosine_similarity = (raw_score * 2) - 1
                if cosine_similarity >= threshold:
                    results.append({
                        'id': hit['_source']['id'],
                        'similarity': cosine_similarity,
                        'timestamp': hit['_source'].get('timestamp'),
                        'metadata': hit['_source'].get('metadata', {})
                    })
            results.sort(key=lambda x: x['similarity'], reverse=True)
            if len(results) > 0:
                logger.info(f"搜索完成，找到 {len(results)} 个匹配结果")
                return (RunningCode.SUCCESS, results)
            else:
                logger.info("没有找到匹配的人脸")
                return (RunningCode.NO_MATCHED_FACE_DETECTED, results)
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return (RunningCode.ES_ERROR, []) 
    def delete_face(self, identity: str, reponame: str) -> RunningCode:
        try:
            if not self._check_face_store_exists(reponame):
                return RunningCode.FACE_STORE_NOT_EXISTS
            if not self._check_face_exists(identity, reponame):
                return RunningCode.ID_NOT_FOUND
            response = self.es.delete(index=reponame, id=identity, refresh=True)
            logger.info(f"成功删除人脸: {identity}")
            return RunningCode.SUCCESS
        except Exception as e:
            logger.error(f"删除人脸失败，请检查ES连接: {e}")
            return RunningCode.ES_ERROR
    def update_face(self, identity: str, embedding: np.ndarray, reponame: str) -> RunningCode:
        try:
            if not self._check_face_store_exists(reponame):
                return RunningCode.FACE_STORE_NOT_EXISTS
            if not self._check_face_exists(identity, reponame):
                return RunningCode.ID_NOT_FOUND
            doc = {
                "embedding": embedding.flatten().tolist(),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "embedding_shape": embedding.shape,
                    "embedding_norm": float(np.linalg.norm(embedding))
                }
            }
            response = self.es.update(index=reponame, id=identity, body={"doc": doc}, refresh=True)
            logger.info(f"成功更新人脸: {identity}")
            return RunningCode.SUCCESS
        except Exception as e:
            logger.error(f"更新人脸失败，请检查ES连接: {e}")
            return RunningCode.ES_ERROR
