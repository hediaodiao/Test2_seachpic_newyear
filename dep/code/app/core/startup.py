# 服务启动和初始化
import logging
from app.services.feature import FeatureService
from app.services.vector_db import VectorDBService
from app.core.config import settings

from pymilvus import connections

import time
# 全局服务实例
feature_service = None
vector_db_service = None

async def wait_for_services():
    """等待依赖服务就绪"""
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 使用Docker中的Milvus
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=str(settings.milvus_port)
            )
            logging.info("Milvus连接成功")
            return True
        except Exception as e:
            retry_count += 1
            logging.warning(f"等待依赖服务... ({retry_count}/{max_retries}),"f'错误：{str(e)}')
            time.sleep(2)

    error_msg = (
        f"Milvus服务启动超时，无法连接到 {settings.milvus_host}:{settings.milvus_port}\n"
        f"请检查：\n"
        f"1. Milvus容器是否正常运行\n"
        f"2. 网络连接是否正常\n"
        f"3. Milvus服务是否配置正确"
    )
    logging.error(error_msg)
    raise Exception("依赖服务启动超时")

async def initialize_services():
    """初始化服务"""
    global feature_service, vector_db_service
    
    logging.info("初始化特征提取服务...")
    feature_service = FeatureService(
        model_names=settings.models,
        cache_dir=settings.model_cache_dir
    )
    
    logging.info("初始化向量数据库服务...")
    vector_db_service = VectorDBService(
        model_names=settings.models,
        collections=settings.collections,
        milvus_host=settings.milvus_host,
        milvus_port=settings.milvus_port
    )
    
    logging.info("服务初始化完成")
    return True
