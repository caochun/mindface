import os
import logging
import numpy as np
from datetime import datetime
from pymilvus import MilvusClient, DataType
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from running_code import RunningCode

class FaceEmbeddingMilvus:
    def __init__(self, host='localhost', port=19530, embedding_dim=512,
                 use_auth=True, username=None, password=None):
        uri = f"http://{host}:{port}"
        
        if use_auth:
            username = username or os.getenv("MILVUS_USERNAME", "root")
            password = password or os.getenv("MILVUS_PASSWORD", "Milvus")
            if username and password:
                token = f"{username}:{password}"
                self.client = MilvusClient(uri=uri, token=token)
            else:
                logger.warning("认证信息不完整，尝试无认证连接")
                self.client = MilvusClient(uri=uri)
        else:
            self.client = MilvusClient(uri=uri)

        self.embedding_dim = embedding_dim

        try:
            # 测试连接
            self.client.list_collections()
            logger.info("成功连接到 Milvus")
        except Exception as e:
            logger.error(f"Milvus 连接错误: {e}")
            raise ConnectionError("Milvus 连接失败")

    def _check_face_store_exists(self, reponame: str) -> bool:
        try:
            collections = self.client.list_collections()
            return reponame in collections
        except Exception as e:
            logger.error(f"检索人脸库失败，请检查Milvus连接: {e}")
            return False

    def _check_face_exists(self, identity: str, reponame: str) -> bool:
        try:
            if not self._check_face_store_exists(reponame):
                return False
            
            results = self.client.query(
                collection_name=reponame,
                filter=f'id == "{identity}"',
                output_fields=["id"]
            )
            return len(results) > 0
        except Exception as e:
            logger.error(f"检索人脸信息失败，请检查Milvus连接: {e}")
            return False

    def create_face_store(self, reponame: str) -> RunningCode:
        try:
            if self._check_face_store_exists(reponame):
                return RunningCode.FACE_STORE_EXISTS
            
            # 定义collection schema
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True
            )
            
            # 添加字段
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=64)
            
            # 创建索引参数
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 1024}
            )
            
            # 创建collection
            self.client.create_collection(
                collection_name=reponame,
                schema=schema,
                index_params=index_params
            )
            
            logger.info(f"成功创建人脸库: {reponame}")
            return RunningCode.SUCCESS
        except Exception as e:
            logger.error(f"创建人脸库失败，请检查Milvus连接: {e}")
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
                "embedding_shape": str(embedding.shape),
                "embedding_norm": float(np.linalg.norm(embedding))
            }
            
            self.client.insert(
                collection_name=reponame,
                data=[doc]
            )
            
            logger.info(f"成功注册人脸: {identity} -> {reponame}")
            return RunningCode.SUCCESS
        except Exception as e:
            logger.error(f"注册人脸失败: {e}")
            return RunningCode.ES_ERROR

    def delete_face(self, identity: str, reponame: str) -> RunningCode:
        try:
            if not self._check_face_store_exists(reponame):
                return RunningCode.FACE_STORE_NOT_EXISTS
            
            if not self._check_face_exists(identity, reponame):
                return RunningCode.ID_NOT_FOUND
            
            self.client.delete(
                collection_name=reponame,
                filter=f'id == "{identity}"'
            )
            
            logger.info(f"成功删除人脸: {identity}")
            return RunningCode.SUCCESS
        except Exception as e:
            logger.error(f"删除人脸失败，请检查Milvus连接: {e}")
            return RunningCode.ES_ERROR

    def update_face(self, identity: str, embedding: np.ndarray, reponame: str) -> RunningCode:
        try:
            if not self._check_face_store_exists(reponame):
                return RunningCode.FACE_STORE_NOT_EXISTS
            
            if not self._check_face_exists(identity, reponame):
                return RunningCode.ID_NOT_FOUND
            
            # Milvus的更新策略：先删除再插入
            self.client.delete(
                collection_name=reponame,
                filter=f'id == "{identity}"'
            )
            
            doc = {
                "id": identity,
                "embedding": embedding.flatten().tolist(),
                "timestamp": datetime.now().isoformat(),
                "embedding_shape": str(embedding.shape),
                "embedding_norm": float(np.linalg.norm(embedding))
            }
            
            self.client.insert(
                collection_name=reponame,
                data=[doc]
            )
            
            logger.info(f"成功更新人脸: {identity}")
            return RunningCode.SUCCESS
        except Exception as e:
            logger.error(f"更新人脸失败，请检查Milvus连接: {e}")
            return RunningCode.ES_ERROR
        
    def recognize_face(self, identity: str, embedding: np.ndarray, reponame: str, threshold: float = 0.9) -> Tuple[RunningCode, Dict]:
        try:
            if not self._check_face_store_exists(reponame):
                return (RunningCode.FACE_STORE_NOT_EXISTS, {})
            
            if not self._check_face_exists(identity, reponame):
                return (RunningCode.ID_NOT_FOUND, {})
            
            search_results = self.client.search(
                collection_name=reponame,
                data=[embedding.flatten().tolist()],
                limit=1,
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
                filter=f'id == "{identity}"',
                output_fields=["id", "timestamp", "embedding_shape", "embedding_norm"]
            )
            
            if not search_results or len(search_results[0]) == 0:
                return (RunningCode.ID_NOT_FOUND, {})
            
            hit = search_results[0][0]
            similarity = hit['distance']  # Milvus COSINE距离就是余弦相似度
            
            result = {
                'id': hit['entity']['id'],
                'similarity': similarity,
                'timestamp': hit['entity'].get('timestamp'),
                'metadata': {
                    'embedding_shape': hit['entity'].get('embedding_shape'),
                    'embedding_norm': hit['entity'].get('embedding_norm')
                },
                'threshold_passed': similarity >= threshold
            }
            
            if similarity >= threshold:
                logger.info(f"身份验证成功: {identity}, 相似度: {similarity:.4f}")
                return (RunningCode.SUCCESS, result)
            else:
                logger.info(f"身份验证失败: {identity}, 相似度: {similarity:.4f} < 阈值: {threshold}")
                return (RunningCode.ID_NOT_FOUND, result)
        except Exception as e:
            logger.error(f"身份识别失败: {e}")
            return (RunningCode.ES_ERROR, {})
        
    def search_similar_faces(self, query_embedding: np.ndarray, reponame: str, 
                    k: int = 10, threshold: float = 0.9) -> Tuple[RunningCode, List[Dict]]:
        try:
            if not self._check_face_store_exists(reponame):
                return (RunningCode.FACE_STORE_NOT_EXISTS, [])
            
            search_results = self.client.search(
                collection_name=reponame,
                data=[query_embedding.flatten().tolist()],
                limit=k,
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
                output_fields=["id", "timestamp", "embedding_shape", "embedding_norm"]
            )
            
            results = []
            for hits in search_results:
                for hit in hits:
                    similarity = hit['distance']  # Milvus COSINE距离就是余弦相似度
                    if similarity >= threshold:
                        results.append({
                            'id': hit['entity']['id'],
                            'similarity': similarity,
                            'timestamp': hit['entity'].get('timestamp'),
                            'metadata': {
                                'embedding_shape': hit['entity'].get('embedding_shape'),
                                'embedding_norm': hit['entity'].get('embedding_norm')
                            }
                        })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            if len(results) > 0:
                logger.info(f"搜索完成，找到 {len(results)} 个匹配结果")
                return (RunningCode.SUCCESS, results)
            else:
                logger.info("没有找到匹配的人脸")
                return (RunningCode.ID_NOT_FOUND, results)
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return (RunningCode.ES_ERROR, [])