# 全局配置
import os
from typing import List, Dict

# 模型配置
MODELS: List[str] = [
    "resnet50", 
    "efficientnet_lite0", 
    "mobilenet_v3_small", 
    "convnext_tiny", 
    "openclip_vit_b_32"
]

COLLECTIONS: Dict[str, str] = {
    "resnet50": "image_features_resnet50",
    "efficientnet_lite0": "image_features_efficientnet_lite0",
    "mobilenet_v3_small": "image_features_mobilenet_v3_small",
    "convnext_tiny": "image_features_convnext_tiny",
    "openclip_vit_b_32": "image_features_openclip_vit_b_32"
}

# 存储配置
USE_MILVUS_LITE: bool = False
USE_DOCKER_MILVUS: bool = True

# 路径配置
MODEL_CACHE_DIR: str = "./models/cache"
VECTOR_DB_STORAGE: str = "./data/vector_db"
STATIC_DIR: str = "./static"
TEMPLATES_DIR: str = "./templates"

# Milvus配置
MILVUS_HOST: str = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))

# 应用配置
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
