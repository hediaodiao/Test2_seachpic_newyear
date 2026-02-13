# 向量数据库服务
import logging
from typing import Dict, List, Optional
from app.services.vector_db.milvus import MilvusManager

class VectorDBService:
    """向量数据库服务"""
    
    def __init__(self, model_names: List[str], collections: Dict[str, str], 
                 milvus_host: str, milvus_port: int):
        """
        初始化向量数据库服务
        
        参数:
            model_names: 模型名称列表
            collections: 集合名称映射
            milvus_host: Milvus主机地址
            milvus_port: Milvus端口
        """
        self.model_names = model_names
        self.collections = collections
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.managers = {}
        self._initialize_managers()
    
    def _initialize_managers(self):
        """初始化向量数据库管理器"""
        for model_name in self.model_names:
            collection_name = self.collections[model_name]
            logging.info(f"连接向量数据库: {collection_name}")
            
            # 只使用Docker中的Milvus
            manager = MilvusManager(
                host=self.milvus_host,
                port=self.milvus_port,
                collection_name=collection_name,
                dimension=2048  # 这里需要根据模型动态设置
            )
            
            self.managers[model_name] = manager
            logging.info(f"✓ 向量数据库连接成功: {collection_name}")
    
    def search(self, model_name: str, query_vector: List[float], top_k: int):
        """
        搜索相似向量
        
        参数:
            model_name: 模型名称
            query_vector: 查询向量
            top_k: 返回的最相似结果数量
            
        返回:
            搜索结果列表
        """
        if model_name not in self.managers:
            raise ValueError(f"模型 {model_name} 的向量数据库管理器未初始化")
        
        return self.managers[model_name].search(query_vector, top_k=top_k)
    
    async def check_health(self):
        """检查数据库健康状态"""
        try:
            if not self.managers:
                return False
            
            # 检查第一个管理器的健康状态
            model_name = next(iter(self.managers.keys()))
            manager = self.managers[model_name]
            
            # 尝试获取集合统计信息
            stats = manager.get_collection_stats()
            return stats is not None
        except Exception:
            return False
    
    def get_connections_status(self):
        """获取连接状态"""
        status = {}
        for model_name in self.model_names:
            status[model_name] = model_name in self.managers
        return status
    
    def get_collection_stats(self):
        """获取集合统计信息"""
        stats = {}
        for model_name, manager in self.managers.items():
            try:
                stats[model_name] = manager.get_collection_stats()
            except Exception:
                stats[model_name] = None
        return stats
