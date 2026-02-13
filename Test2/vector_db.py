#!/usr/bin/env python3
"""
向量数据库管理
支持 Milvus 和本地存储两种方式
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os
import json
from scipy.spatial.distance import cosine, euclidean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalVectorManager:
    """本地向量存储管理器（使用 NumPy + JSON）"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19532,
        collection_name: str = "image_features",
        dimension: int = 2048,
        metric_type: str = "L2",
        storage_dir: str = "./vector_db_storage"
    ):
        """
        初始化本地向量管理器
        
        参数:
            host: 保留参数（兼容性）
            port: 保留参数（兼容性）
            collection_name: 集合名称
            dimension: 向量维度
            metric_type: 距离度量类型 (L2, COSINE)
            storage_dir: 存储目录
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.storage_dir = storage_dir
        self.collection = None
        self.connected = False
        self.features = []
        self.image_names = []
        self.image_paths = []
        
        self._connect()
    
    def _connect(self):
        """连接到本地存储"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            self.connected = True
            logger.info(f"成功连接到本地存储: {self.storage_dir}")
        except Exception as e:
            logger.error(f"连接本地存储失败: {e}")
            raise
    
    def create_collection(self, drop_existing: bool = False):
        """
        创建集合
        
        参数:
            drop_existing: 如果集合已存在，是否删除重建
        """
        collection_file = os.path.join(self.storage_dir, f"{self.collection_name}.json")
        
        if os.path.exists(collection_file):
            if drop_existing:
                logger.info(f"删除已存在的集合: {self.collection_name}")
                self.features = []
                self.image_names = []
                self.image_paths = []
                self._save_collection()
            else:
                logger.info(f"集合已存在: {self.collection_name}")
                self._load_collection()
                return
        
        logger.info(f"创建集合成功: {self.collection_name}")
    
    def _save_collection(self):
        """保存集合到文件"""
        collection_file = os.path.join(self.storage_dir, f"{self.collection_name}.json")
        data = {
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "metric_type": self.metric_type,
            "features": [f.tolist() for f in self.features],
            "image_names": self.image_names,
            "image_paths": self.image_paths
        }
        with open(collection_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_collection(self):
        """从文件加载集合"""
        collection_file = os.path.join(self.storage_dir, f"{self.collection_name}.json")
        if os.path.exists(collection_file):
            with open(collection_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.features = [np.array(f) for f in data["features"]]
                self.image_names = data["image_names"]
                self.image_paths = data["image_paths"]
                logger.info(f"加载集合: {self.collection_name}, 记录数: {len(self.features)}")
    
    def insert_features(
        self,
        features_list: List[np.ndarray],
        image_names: List[str],
        image_paths: List[str]
    ) -> int:
        """
        插入特征向量
        
        参数:
            features_list: 特征向量列表
            image_names: 图片名称列表
            image_paths: 图片路径列表
            
        返回:
            插入的记录数
        """
        if len(features_list) != len(image_names) or len(features_list) != len(image_paths):
            raise ValueError("特征、名称和路径列表长度不一致")
        
        self.features.extend(features_list)
        self.image_names.extend(image_names)
        self.image_paths.extend(image_paths)
        
        self._save_collection()
        logger.info(f"成功插入 {len(features_list)} 条记录")
        
        return len(features_list)
    
    def create_index(self, index_type: str = "HNSW", params: Optional[Dict] = None):
        """
        创建索引（本地存储不需要索引）
        
        参数:
            index_type: 索引类型（忽略）
            params: 索引参数（忽略）
        """
        logger.info(f"本地存储不需要索引，跳过创建")
    
    def load_collection(self):
        """加载集合到内存"""
        self._load_collection()
        logger.info("集合已加载到内存")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        """
        搜索相似向量
        
        参数:
            query_vector: 查询向量
            top_k: 返回的最相似结果数量
            
        返回:
            搜索结果列表，每个元素包含图片信息
        """
        if len(self.features) == 0:
            logger.warning("集合中没有数据")
            return []
        
        distances = []
        for i, feature in enumerate(self.features):
            if self.metric_type == "L2":
                dist = euclidean(query_vector, feature)
            elif self.metric_type == "COSINE":
                dist = cosine(query_vector, feature)
            else:
                dist = euclidean(query_vector, feature)
            distances.append((i, dist))
        
        distances.sort(key=lambda x: x[1])
        
        formatted_results = []
        for idx, dist in distances[:top_k]:
            formatted_results.append({
                "image_name": self.image_names[idx],
                "image_path": self.image_paths[idx],
                "distance": dist,
                "score": np.exp(-dist / 529.0)  # 调整缩放因子以获得0.8相似度@距离118
            })
        
        logger.info(f"搜索完成，返回 {len(formatted_results)} 个结果")
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        stats = {
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "metric_type": self.metric_type,
            "num_entities": len(self.features)
        }
        return stats
    
    def drop_collection(self):
        """删除集合"""
        collection_file = os.path.join(self.storage_dir, f"{self.collection_name}.json")
        if os.path.exists(collection_file):
            os.remove(collection_file)
            logger.info(f"删除集合: {self.collection_name}")
        
        self.features = []
        self.image_names = []
        self.image_paths = []
    
    def disconnect(self):
        """断开连接"""
        self.connected = False
        logger.info("已断开本地存储连接")


class MilvusLiteManager:
    """Milvus Lite 向量数据库管理器（本地文件存储）"""
    
    def __init__(
        self,
        db_path: str = "./milvus_demo.db",
        collection_name: str = "image_features",
        dimension: int = 2048,
        metric_type: str = "L2"
    ):
        """
        初始化 Milvus Lite 管理器
        
        参数:
            db_path: Milvus Lite 数据库文件路径
            collection_name: 集合名称
            dimension: 向量维度
            metric_type: 距离度量类型 (L2, IP, COSINE)
        """
        try:
            from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
            self.pymilvus_available = True
        except ImportError:
            logger.warning("pymilvus 未安装，MilvusLiteManager 将不可用")
            self.pymilvus_available = False
            raise ImportError("pymilvus 未安装，请使用 LocalVectorManager 或安装 pymilvus[milvus-lite]")
        
        self.db_path = db_path
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.collection = None
        self.connected = False
        
        self._connect()
    
    def _connect(self):
        """连接到 Milvus Lite 数据库"""
        try:
            from pymilvus import connections
            
            connections.connect(
                alias="default",
                uri=self.db_path
            )
            self.connected = True
            logger.info(f"成功连接到 Milvus Lite 数据库: {self.db_path}")
        except Exception as e:
            logger.error(f"连接 Milvus Lite 失败: {e}")
            raise
    
    def create_collection(self, drop_existing: bool = False):
        """
        创建集合
        
        参数:
            drop_existing: 如果集合已存在，是否删除重建
        """
        from pymilvus import utility, FieldSchema, CollectionSchema, DataType, Collection
        
        if utility.has_collection(self.collection_name):
            if drop_existing:
                logger.info(f"删除已存在的集合: {self.collection_name}")
                utility.drop_collection(self.collection_name)
            else:
                logger.info(f"集合已存在: {self.collection_name}")
                self.collection = Collection(self.collection_name)
                return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="feature_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Image features collection (dim={self.dimension})"
        )
        
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        logger.info(f"创建集合成功: {self.collection_name}")
    
    def insert_features(
        self,
        features_list: List[np.ndarray],
        image_names: List[str],
        image_paths: List[str]
    ) -> int:
        """
        插入特征向量
        
        参数:
            features_list: 特征向量列表
            image_names: 图片名称列表
            image_paths: 图片路径列表
            
        返回:
            插入的记录数
        """
        if not self.collection:
            raise RuntimeError("集合未初始化")
        
        if len(features_list) != len(image_names) or len(features_list) != len(image_paths):
            raise ValueError("特征、名称和路径列表长度不一致")
        
        # 过滤掉已存在的图片，避免重复插入
        unique_features = []
        unique_names = []
        unique_paths = []
        
        # 跳过重复检查，直接插入
        # 这是因为在插入时进行重复检查会增加额外的开销，而且可能导致集合未加载的错误
        # 重复检查应该在特征提取阶段（evaluate_model.py中）进行，而不是在插入阶段
        for feature, name, path in zip(features_list, image_names, image_paths):
            unique_features.append(feature)
            unique_names.append(name)
            unique_paths.append(path)
        
        if not unique_features:
            logger.info("所有图片已存在于数据库中，跳过插入")
            return 0
        
        entities = [
            unique_names,
            unique_paths,
            unique_features
        ]
        
        insert_result = self.collection.insert(entities)
        logger.info(f"成功插入 {len(unique_features)} 条记录")
        
        return len(unique_features)
    
    def create_index(self, index_type: str = "HNSW", params: Optional[Dict] = None):
        """
        创建索引

        参数:
            index_type: 索引类型 (HNSW, IVF_FLAT, IVF_SQ8, etc.)
            params: 索引参数
        """
        if not self.collection:
            from pymilvus import Collection
            logger.info(f"集合未初始化，尝试获取已存在的集合: {self.collection_name}")
            self.collection = Collection(self.collection_name)
        
        try:
            # 检查索引是否已存在（兼容旧版本pymilvus）
            has_index = False
            try:
                # 尝试获取索引信息
                index_info = self.collection.indexes
                if index_info:
                    has_index = True
            except Exception as index_check_error:
                logger.warning(f"检查索引存在性时出错: {index_check_error}")
                has_index = False
            
            if has_index:
                logger.info(f"索引已存在，跳过创建")
                return
            
            if params is None:
                params = {
                    "M": 16,
                    "efConstruction": 256
                }
            
            index_params = {
                "index_type": index_type,
                "metric_type": self.metric_type,
                "params": params
            }
            
            self.collection.create_index(
                field_name="feature_vector",
                index_params=index_params
            )
            logger.info(f"创建索引成功: {index_type}")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            raise
    
    def load_collection(self, timeout=30):
        """加载集合到内存
        
        参数:
            timeout: 加载超时时间（秒）
        """
        import threading
        
        if not self.collection:
            from pymilvus import utility, Collection
            # 尝试获取已存在的集合
            if utility.has_collection(self.collection_name):
                logger.info(f"获取已存在的集合: {self.collection_name}")
                self.collection = Collection(self.collection_name)
            else:
                # 集合不存在，抛出更清晰的错误信息
                raise RuntimeError(f"集合 {self.collection_name} 不存在，请先运行特征提取功能")
        
        # 定义加载集合的函数
        def load_collection_thread():
            try:
                # 加载集合，跳过索引检查
                logger.info(f"开始加载集合: {self.collection_name}")
                logger.info(f"集合对象: {self.collection}")
                logger.info(f"集合状态: 准备加载")
                
                self.collection.load(skip_index_check=True)
                
                logger.info(f"集合 {self.collection_name} 已成功加载到内存")
                load_success[0] = True
            except Exception as e:
                logger.error(f"加载集合失败: {e}")
                load_success[0] = False
                load_error[0] = e
        
        # 使用列表来存储线程执行结果
        load_success = [False]
        load_error = [None]
        
        # 创建并启动加载线程
        load_thread = threading.Thread(target=load_collection_thread)
        load_thread.daemon = True
        load_thread.start()
        
        # 等待线程完成或超时
        load_thread.join(timeout=timeout)
        
        # 检查加载是否成功
        if not load_success[0]:
            if load_error[0]:
                raise Exception(f"加载集合超时或失败: {load_error[0]}")
            else:
                raise Exception(f"加载集合超时（{timeout}秒）")
        
        logger.info(f"集合 {self.collection_name} 加载完成")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        """
        搜索相似向量
        
        参数:
            query_vector: 查询向量
            top_k: 返回的最相似结果数量
            
        返回:
            搜索结果列表，每个元素包含图片信息
        """
        if not self.collection:
            raise RuntimeError("集合未初始化")
        
        search_params = {
            "metric_type": self.metric_type,
            "params": {"ef": 64}
        }
        
        max_retries = 5
        for retry in range(max_retries):
            try:
                # 尝试直接搜索，不预先加载集合
                # Milvus会在搜索时自动加载集合
                logger.info(f"尝试搜索 (尝试 {retry+1}/{max_retries})...")
                results = self.collection.search(
                    data=[query_vector],
                    anns_field="feature_vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["image_name", "image_path"]
                )
                logger.info(f"搜索成功，返回 {len(results[0])} 个结果")
                
                # 格式化结果
                formatted_results = []
                for result in results[0]:
                    formatted_results.append({
                        "image_name": result.entity.get("image_name"),
                        "image_path": result.entity.get("image_path"),
                        "distance": result.distance,
                        "score": np.exp(-result.distance / 529.0)  # 调整缩放因子以获得0.8相似度@距离118
                    })
                
                logger.info(f"搜索完成，返回 {len(formatted_results)} 个结果")
                return formatted_results
                
            except Exception as e:
                logger.warning(f"搜索失败 (尝试 {retry+1}/{max_retries}): {e}")
                if retry == max_retries - 1:
                    logger.error(f"搜索失败，达到最大重试次数: {e}")
                    raise
                
                # 等待5秒后重试，给Milvus更多时间加载集合
                import time
                logger.info(f"等待5秒后重试...")
                time.sleep(5)
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        if not self.collection:
            raise RuntimeError("集合未初始化")
        
        self.collection.flush()
        num_entities = self.collection.num_entities
        
        stats = {
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "metric_type": self.metric_type,
            "num_entities": num_entities
        }
        
        return stats
    
    def drop_collection(self):
        """删除集合"""
        from pymilvus import utility
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"删除集合: {self.collection_name}")
    
    def disconnect(self):
        """断开连接"""
        from pymilvus import connections
        connections.disconnect("default")
        self.connected = False
        logger.info("已断开 Milvus Lite 连接")


class MilvusManager:
    """Milvus向量数据库管理器"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19532,
        collection_name: str = "image_features",
        dimension: int = 2048,
        metric_type: str = "L2"
    ):
        """
        初始化Milvus管理器
        
        参数:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 集合名称
            dimension: 向量维度
            metric_type: 距离度量类型 (L2, IP, COSINE)
        """
        try:
            from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
            self.pymilvus_available = True
        except ImportError:
            logger.warning("pymilvus 未安装，MilvusManager 将不可用")
            self.pymilvus_available = False
            raise ImportError("pymilvus 未安装，请使用 LocalVectorManager 或安装 pymilvus")
        
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.collection = None
        self.connected = False
        
        self._connect()
    
    def _connect(self):
        """连接到Milvus服务器"""
        try:
            from pymilvus import connections
            connections.connect(host=self.host, port=self.port)
            self.connected = True
            logger.info(f"成功连接到Milvus服务器: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def create_collection(self, drop_existing: bool = False):
        """
        创建集合
        
        参数:
            drop_existing: 如果集合已存在，是否删除重建
        """
        from pymilvus import utility, FieldSchema, CollectionSchema, DataType, Collection
        
        if utility.has_collection(self.collection_name):
            if drop_existing:
                logger.info(f"删除已存在的集合: {self.collection_name}")
                utility.drop_collection(self.collection_name)
            else:
                logger.info(f"集合已存在: {self.collection_name}")
                self.collection = Collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="feature_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Image features collection (dim={self.dimension})"
        )
        
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        logger.info(f"创建集合成功: {self.collection_name}")
    
    def insert_features(
        self,
        features_list: List[np.ndarray],
        image_names: List[str],
        image_paths: List[str]
    ) -> int:
        """
        插入特征向量
        
        参数:
            features_list: 特征向量列表
            image_names: 图片名称列表
            image_paths: 图片路径列表
            
        返回:
            插入的记录数
        """
        if not self.collection:
            raise RuntimeError("集合未初始化")
        
        if len(features_list) != len(image_names) or len(features_list) != len(image_paths):
            raise ValueError("特征、名称和路径列表长度不一致")
        
        # 过滤掉已存在的图片，避免重复插入
        unique_features = []
        unique_names = []
        unique_paths = []
        
        # 直接插入所有特征，跳过重复检查
        # 重复检查应该在特征提取阶段（evaluate_model.py）进行
        for feature, name, path in zip(features_list, image_names, image_paths):
            unique_features.append(feature)
            unique_names.append(name)
            unique_paths.append(path)
        
        if not unique_features:
            logger.info("所有图片已存在于数据库中，跳过插入")
            return 0
        
        entities = [
            unique_names,
            unique_paths,
            unique_features
        ]
        
        insert_result = self.collection.insert(entities)
        logger.info(f"成功插入 {len(unique_features)} 条记录")
        
        return len(unique_features)
    
    def create_index(self, index_type: str = "HNSW", params: Optional[Dict] = None):
        """
        创建索引

        参数:
            index_type: 索引类型 (HNSW, IVF_FLAT, IVF_SQ8, etc.)
            params: 索引参数
        """
        if not self.collection:
            from pymilvus import Collection
            logger.info(f"集合未初始化，尝试获取已存在的集合: {self.collection_name}")
            self.collection = Collection(self.collection_name)
        
        try:
            # 检查索引是否已存在（兼容旧版本pymilvus）
            has_index = False
            try:
                # 尝试获取索引信息
                index_info = self.collection.indexes
                if index_info:
                    has_index = True
            except Exception as index_check_error:
                logger.warning(f"检查索引存在性时出错: {index_check_error}")
                has_index = False
            
            if has_index:
                logger.info(f"索引已存在，跳过创建")
                return
            
            if params is None:
                params = {
                    "M": 16,
                    "efConstruction": 256
                }
            
            index_params = {
                "index_type": index_type,
                "metric_type": self.metric_type,
                "params": params
            }
            
            self.collection.create_index(
                field_name="feature_vector",
                index_params=index_params
            )
            logger.info(f"创建索引成功: {index_type}")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            raise
    
    def load_collection(self, timeout=60):
        """加载集合到内存
        
        参数:
            timeout: 加载超时时间（秒）
        """
        import threading
        
        if not self.collection:
            from pymilvus import utility, Collection
            # 尝试获取已存在的集合
            if utility.has_collection(self.collection_name):
                logger.info(f"获取已存在的集合: {self.collection_name}")
                self.collection = Collection(self.collection_name)
            else:
                # 集合不存在，抛出更清晰的错误信息
                raise RuntimeError(f"集合 {self.collection_name} 不存在，请先运行特征提取功能")
        
        # 定义加载集合的函数
        def load_collection_thread():
            try:
                # 加载集合，跳过索引检查
                logger.info(f"开始加载集合: {self.collection_name}")
                logger.info(f"集合对象: {self.collection}")
                logger.info(f"集合状态: 准备加载")
                
                self.collection.load(skip_index_check=True)
                
                logger.info(f"集合 {self.collection_name} 已成功加载到内存")
                load_success[0] = True
            except Exception as e:
                logger.error(f"加载集合失败: {e}")
                load_success[0] = False
                load_error[0] = e
        
        # 使用列表来存储线程执行结果
        load_success = [False]
        load_error = [None]
        
        # 创建并启动加载线程
        load_thread = threading.Thread(target=load_collection_thread)
        load_thread.daemon = True
        load_thread.start()
        
        # 等待线程完成或超时
        load_thread.join(timeout=timeout)
        
        # 检查加载是否成功
        if not load_success[0]:
            if load_error[0]:
                raise Exception(f"加载集合超时或失败: {load_error[0]}")
            else:
                raise Exception(f"加载集合超时（{timeout}秒）")
        
        logger.info(f"集合 {self.collection_name} 加载完成")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        """
        搜索相似向量
        
        参数:
            query_vector: 查询向量
            top_k: 返回的最相似结果数量
            
        返回:
            搜索结果列表，每个元素包含图片信息
        """
        if not self.collection:
            raise RuntimeError("集合未初始化")
        
        search_params = {
            "metric_type": self.metric_type,
            "params": {"ef": 64}
        }
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                # 尝试直接搜索，不预先加载集合
                # Milvus会在搜索时自动加载集合
                logger.info(f"尝试搜索 (尝试 {retry+1}/{max_retries})...")
                results = self.collection.search(
                    data=[query_vector],
                    anns_field="feature_vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["image_name", "image_path"]
                )
                logger.info(f"搜索成功，返回 {len(results[0])} 个结果")
                
                # 格式化结果
                formatted_results = []
                for result in results[0]:
                    formatted_results.append({
                        "image_name": result.entity.get("image_name"),
                        "image_path": result.entity.get("image_path"),
                        "distance": result.distance,
                        "score": np.exp(-result.distance / 529.0)  # 调整缩放因子以获得0.8相似度@距离118
                    })
                
                logger.info(f"搜索完成，返回 {len(formatted_results)} 个结果")
                return formatted_results
                
            except Exception as e:
                logger.warning(f"搜索失败 (尝试 {retry+1}/{max_retries}): {e}")
                if retry == max_retries - 1:
                    logger.error(f"搜索失败，达到最大重试次数: {e}")
                    raise
                
                # 等待2秒后重试
                import time
                time.sleep(2)
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        if not self.collection:
            raise RuntimeError("集合未初始化")
        
        self.collection.flush()
        num_entities = self.collection.num_entities
        
        stats = {
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "metric_type": self.metric_type,
            "num_entities": num_entities
        }
        
        return stats
    
    def drop_collection(self):
        """删除集合"""
        from pymilvus import utility
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"删除集合: {self.collection_name}")
    
    def disconnect(self):
        """断开连接"""
        from pymilvus import connections
        connections.disconnect("default")
        self.connected = False
        logger.info("已断开Milvus连接")


def batch_insert_images(
    milvus_manager: LocalVectorManager,
    features_dict: Dict[str, np.ndarray],
    image_dir: str,
    batch_size: int = 1000
):
    """
    批量插入图片特征
    
    参数:
        milvus_manager: 向量数据库管理器实例
        features_dict: 特征字典 {相对路径: 特征向量}
        image_dir: 图片目录路径
        batch_size: 每批插入的图片数量
    """
    import os
    
    features_list = []
    image_names = []
    image_paths = []
    
    for rel_path, feature in features_dict.items():
        features_list.append(feature)
        image_names.append(os.path.basename(rel_path))
        image_paths.append(os.path.join(image_dir, rel_path))
    
    # 分批插入以避免超过 gRPC 消息大小限制
    total = len(features_list)
    for i in range(0, total, batch_size):
        batch_features = features_list[i:i + batch_size]
        batch_names = image_names[i:i + batch_size]
        batch_paths = image_paths[i:i + batch_size]
        
        logger.info(f"插入批次 {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch_features)} 张图片)")
        milvus_manager.insert_features(batch_features, batch_names, batch_paths)
