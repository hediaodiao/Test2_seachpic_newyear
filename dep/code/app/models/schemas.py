# 数据模型和DTO
from pydantic import BaseModel
from typing import List, Optional, Dict

class SearchResult(BaseModel):
    """单个搜索结果"""
    rank: int
    image_name: str
    image_path: str
    similarity_score: float
    distance: float

class ModelResult(BaseModel):
    """单个模型的结果"""
    model_name: str
    similar_images: List[SearchResult]

class SearchResponse(BaseModel):
    """搜索结果响应模型"""
    success: bool
    message: str
    query_image: str
    results: Dict[str, ModelResult]

class StatusResponse(BaseModel):
    """状态响应模型"""
    success: bool
    models_loaded: Dict[str, bool]
    db_connected: Dict[str, bool]
    collection_stats: Dict[str, Optional[dict]]
