# Milvus向量数据库管理器
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusManager:
    """Milvus向量数据库管理器"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "image_features",
        dimension: int = 2048,
        metric_type: str = "L2"
    ):
        """初始化Milvus管理器"""
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.collection = None
        self.connected = False
        
        try:
            from pymilvus import connections
            connections.connect(host=self.host, port=self.port)
            self.connected = True
            logger.info(f"成功连接到Milvus服务器: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        """搜索相似向量"""
        try:
            from pymilvus import Collection
            
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            search_params = {
                "metric_type": self.metric_type,
                "params": {"ef": 64}
            }
            
            results = self.collection.search(
                data=[query_vector],
                anns_field="feature_vector",
                param=search_params,
                limit=top_k,
                output_fields=["image_name", "image_path"]
            )
            
            formatted_results = []
            for i, result in enumerate(results[0], 1):
                formatted_results.append({
                    "rank": i,
                    "image_name": result.entity.get("image_name"),
                    "image_path": result.entity.get("image_path"),
                    "similarity_score": float(np.exp(-result.distance / 529.0)),
                    "distance": float(result.distance)
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        try:
            from pymilvus import Collection
            
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            self.collection.flush()
            num_entities = self.collection.num_entities
            
            stats = {
                "collection_name": self.collection_name,
                "dimension": self.dimension,
                "metric_type": self.metric_type,
                "num_entities": num_entities
            }
            
            return stats
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            raise
