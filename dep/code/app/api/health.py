# 健康检查端点
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime
from app.core.startup import feature_service, vector_db_service

router = APIRouter()

@router.get("/health", tags=["系统状态"])
async def health_check():
    """健康检查端点"""
    try:
        # 检查服务状态
        milvus_healthy = False
        if vector_db_service:
            milvus_healthy = await vector_db_service.check_health()
        
        models_loaded = len(feature_service.models) > 0 if feature_service else False
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "milvus_healthy": milvus_healthy,
                "models_loaded": models_loaded
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
