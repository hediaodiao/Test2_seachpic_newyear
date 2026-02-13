# API路由定义
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import SearchResponse, StatusResponse
from app.core.startup import feature_service, vector_db_service
from app.core.config import settings
import tempfile
import os
from pathlib import Path

router = APIRouter()

@router.post("/search", response_model=SearchResponse, tags=["图像搜索"])
async def search_similar_images(
    file: UploadFile = File(..., description="上传的图片文件"),
    top_k: int = 10
):
    """搜索相似的图片"""
    # 验证文件类型
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {', '.join(allowed_extensions)}"
        )
    
    # 保存上传的文件到临时目录
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"upload_{file.filename}")
    
    try:
        # 保存文件
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 提取特征并搜索
        results = await feature_service.search_similar_images(
            image_path=temp_path,
            top_k=top_k,
            vector_db_service=vector_db_service
        )
        
        return SearchResponse(
            success=True,
            message="搜索成功",
            query_image=file.filename,
            results=results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"搜索失败: {str(e)}"
        )
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/status", response_model=StatusResponse, tags=["系统状态"])
async def get_status():
    """获取系统状态"""
    models_loaded = feature_service.get_models_status() if feature_service else {}
    db_connected = vector_db_service.get_connections_status() if vector_db_service else {}
    collection_stats = vector_db_service.get_collection_stats() if vector_db_service else {}
    
    return StatusResponse(
        success=True,
        models_loaded=models_loaded,
        db_connected=db_connected,
        collection_stats=collection_stats
    )
