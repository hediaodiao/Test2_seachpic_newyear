#!/usr/bin/env python3
"""
FastAPI Web服务-图像相似度搜索
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os
import tempfile
from pathlib import Path
import base64

from feature import FeatureExtractor
from vector_db import LocalVectorManager, MilvusLiteManager, MilvusManager


app = FastAPI(
    title="图像相似度搜索API",
    description="基于EfficientNet-Lite0、ResNet50、MobileNetV3-Small、ConvNeXt-Tiny和OpenCLIP的图像特征提取与相似度搜索",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录（用于显示图片）
app.mount("/static", StaticFiles(directory="./img"), name="static")

# 全局变量
extractors = {}
vector_managers = {}
models = ["resnet50", "efficientnet_lite0", "mobilenet_v3_small", "convnext_tiny", "openclip_vit_b_32"]
collections = {
    "resnet50": "image_features_resnet50",
    "efficientnet_lite0": "image_features_efficientnet_lite0",
    "mobilenet_v3_small": "image_features_mobilenet_v3_small",
    "convnext_tiny": "image_features_convnext_tiny",
    "openclip_vit_b_32": "image_features_openclip_vit_b_32"
}

# 存储方式配置
# True: 使用 Milvus Lite（本地文件存储）
# False: 使用本地存储（JSON文件）
# 注意：如果使用Docker中的Milvus，请修改下面的 USE_DOCKER_MILVUS = True
USE_MILVUS_LITE = False
USE_DOCKER_MILVUS = True



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


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global extractors, vector_managers
    
    print("\n" + "="*60)
    print("FastAPI服务启动中...")
    print("="*60)
    
    try:
        # 初始化两个特征提取器
        for model_name in models:
            print(f"\n加载模型: {model_name}")
            extractors[model_name] = FeatureExtractor(model_name=model_name)
            print(f"✓ 模型加载成功: {model_name}")
        
        # 连接向量数据库
        for model_name in models:      
            collection_name = collections[model_name]
            print(f"\n连接向量数据库: {collection_name}")
            
            if USE_DOCKER_MILVUS:
                # 使用Docker中的Milvus
                vector_managers[model_name] = MilvusManager(
                    host="localhost",
                    port=19532,
                    collection_name=collection_name,
                    dimension=extractors[model_name].feature_dim
                )
            elif USE_MILVUS_LITE:
                # 使用Milvus Lite
                vector_managers[model_name] = MilvusLiteManager(
                    db_path="./milvus_demo.db",
                    collection_name=collection_name,
                    dimension=extractors[model_name].feature_dim
                )
            else:
                # 使用本地存储
                vector_managers[model_name] = LocalVectorManager(
                    collection_name=collection_name,
                    dimension=extractors[model_name].feature_dim
                )
            
            # 检查集合是否存在
            if USE_DOCKER_MILVUS or USE_MILVUS_LITE:
                from pymilvus import utility
                collection_exists = utility.has_collection(collection_name)
            else:
                collection_file = f"./vector_db_storage/{collection_name}.json"
                collection_exists = os.path.exists(collection_file)
            
            if collection_exists:
                # 对于Milvus，需要先创建集合对象（不删除现有数据）
                if USE_DOCKER_MILVUS or USE_MILVUS_LITE:
                    vector_managers[model_name].create_collection(drop_existing=False)
                vector_managers[model_name].load_collection()
                print(f"✓ 向量数据库连接成功: {collection_name}")
            else:
                print(f"⚠ 警告: 集合不存在 {collection_name}，请先运行 init_db.py 初始化数据库")
        
        print("\n" + "="*60)
        print("FastAPI服务启动完成")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()


@app.get("/", tags=["根路径"])
async def root():
    """根路径 - 返回搜索页面"""
    try:
        with open("./templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return {
            "message": "图像相似度搜索API",
            "version": "1.0.0",
            "endpoints": {
                "status": "/status",
                "search": "/search",
                "docs": "/docs"
            }
        }


@app.get("/status", response_model=StatusResponse, tags=["系统状态"])
async def get_status():
    """获取系统状态"""
    global extractors, vector_managers
    
    models_loaded = {}
    db_connected = {}
    collection_stats = {}
    
    for model_name in models:
        models_loaded[model_name] = model_name in extractors
        db_connected[model_name] = model_name in vector_managers
        
        try:
            if model_name in vector_managers:
                collection_stats[model_name] = vector_managers[model_name].get_collection_stats()
            else:
                collection_stats[model_name] = None
        except:
            collection_stats[model_name] = None
    
    return StatusResponse(
        success=True,
        models_loaded=models_loaded,
        db_connected=db_connected,
        collection_stats=collection_stats
    )


@app.post("/search", response_model=SearchResponse, tags=["图像搜索"])
async def search_similar_images(
    file: UploadFile = File(..., description="上传的图片文件"),
    top_k: int = 10
):
    """
    搜索相似的图片，同时使用ResNet50和EfficientNet-Lite0两个模型
    
    参数:
        file: 上传的图片文件
        top_k: 返回的最相似图片数量（默认3）
    
    返回:
        SearchResponse: 包含两个模型相似图片信息的响应
    """
    global extractors, vector_managers
    
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
        
        print(f"\n{'='*60}")
        print("开始搜索相似图片")
        print(f"{'='*60}")
        print(f"查询图片: {file.filename}")
        print(f"使用模型: {', '.join(models)} ({len(models)} 个模型)")
        print(f"返回数量: {top_k}")
        
        results = {}
        
        # 使用两个模型分别进行搜索
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"使用 {model_name} 模型搜索")
            print(f"{'='*60}")
            
            # 提取查询图片的特征
            print(f"提取查询图片特征...")
            query_feature = extractors[model_name].extract_features(temp_path)
            print(f"✓ 特征提取完成，维度: {len(query_feature)}")
            
            # 在向量数据库中搜索
            print(f"在向量数据库中搜索...")
            search_results = vector_managers[model_name].search(query_feature, top_k=top_k)
            print(f"✓ 搜索完成，找到 {len(search_results)} 个结果")
            
            # 格式化结果
            similar_images = []
            for i, result in enumerate(search_results, 1):
                # 将本地路径转换为静态文件路径
                # 移除 ./img 前缀并添加 /static 前缀
                image_path = result["image_path"]
                
                # 标准化路径分隔符为正斜杠
                normalized_path = image_path.replace("\\", "/")
                
                # 移除 ./img/ 前缀
                if normalized_path.startswith("./img/"):
                    rel_path = normalized_path[6:]  # 移除 "./img/" 前缀
                else:
                    rel_path = os.path.basename(image_path)
                
                static_image_path = f"/static/{rel_path}"
                
                similar_images.append(SearchResult(
                    rank=i,
                    image_name=result["image_name"],
                    image_path=static_image_path,
                    similarity_score=round(result["score"], 4),
                    distance=round(result["distance"], 4)
                ))
            
            results[model_name] = ModelResult(
                model_name=model_name,
                similar_images=similar_images
            )
            
            print(f"\n{model_name} 搜索结果:")
            for img in similar_images:
                print(f"  {img.rank}. {img.image_name}")
                print(f"     相似度: {img.similarity_score:.4f}")
                print(f"     距离: {img.distance:.4f}")
        
        print(f"\n{'='*60}\n")
        
        return SearchResponse(
            success=True,
            message="搜索成功",
            query_image=file.filename,
            results=results
        )
        
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"搜索失败: {str(e)}"
        )
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/models", tags=["模型管理"])
async def get_available_models():
    """获取可用的模型列表"""
    return {
        "available_models": models,
        "collections": collections
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
