# 配置加载和管理
import os
from pydantic_settings import BaseSettings
from config.settings import MODELS, COLLECTIONS, MODEL_CACHE_DIR, VECTOR_DB_STORAGE

class Settings(BaseSettings):
    """应用配置"""
    # 应用配置
    debug: bool = False
    app_port: int = 8000
    
    # Milvus配置
    milvus_host: str = "standalone"
    milvus_port: int = 19530
    
    # 路径配置
    model_cache_dir: str = MODEL_CACHE_DIR
    
    # 模型配置
    models: list = MODELS
    collections: dict = COLLECTIONS
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
