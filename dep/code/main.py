# 应用入口
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import router as api_router
from app.core.startup import initialize_services, wait_for_services
from app.core.config import settings

import traceback
# 创建FastAPI应用
app = FastAPI(
    title="图像相似度搜索API",
    description="基于多种模型的图像特征提取与相似度搜索",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="./static"), name="static")

# 注册路由
app.include_router(api_router)

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    print("\n" + "="*60)
    print("FastAPI服务启动中...")
    print("="*60)
    
    try:
        # 等待依赖服务就绪
        await wait_for_services()
        
        # 初始化服务
        await initialize_services()
        
        print("\n" + "="*60)
        print("FastAPI服务启动完成")
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        
        traceback.print_exc()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=settings.debug,
        log_level="info"
    )
