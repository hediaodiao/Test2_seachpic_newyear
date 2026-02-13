from fastapi import APIRouter
from .routes import router as api_router
from .health import router as health_router

# 创建主路由
router = APIRouter()

# 包含所有子路由
router.include_router(api_router)
router.include_router(health_router)

__all__ = ["router"]
