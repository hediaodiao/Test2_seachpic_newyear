#!/usr/bin/env python3
"""
初始化脚本：将img文件夹中的图片特征存入向量数据库
支持本地存储和 Milvus 两种方式
"""

import os
from pathlib import Path
from feature import FeatureExtractor
from vector_db import LocalVectorManager, MilvusLiteManager, MilvusManager, batch_insert_images


def initialize_database(
    image_dir: str = "./img",
    model_name: str = "resnet50",
    collection_name: str = "image_features",
    drop_existing: bool = False,
    use_milvus_lite: bool = False,
    use_docker_milvus: bool = True
):
    """
    初始化数据库，提取图片特征并存储到向量数据库
    
    参数:
        image_dir: 图片目录路径
        model_name: 使用的模型名称 (resnet50, efficientnet_b0, etc.)
        collection_name: 集合名称
        drop_existing: 是否删除已存在的集合
        use_milvus_lite: 是否使用 Milvus Lite（本地文件存储）
        use_docker_milvus: 是否使用 Docker 中的 Milvus
    """
    print(f"\n{'='*60}")
    print("开始初始化图像特征数据库")
    print(f"{'='*60}")
    
    # 1. 检查图片目录
    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"❌ 图片目录不存在: {image_dir}")
        return
                
    # 获取所有图片文件（递归扫描子目录）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = [
        f for f in image_path.rglob('*')
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ 在目录中未找到图片文件: {image_dir}")
        return
    
    print(f"✓ 找到 {len(image_files)} 张图片")
    
    # 2. 初始化特征提取器
    print(f"\n正在加载 {model_name} 模型...")
    extractor = FeatureExtractor(model_name=model_name)
    feature_dim = extractor.feature_dim
    print(f"✓ 特征维度: {feature_dim}")
    
    # 3. 提取所有图片的特征
    print(f"\n开始提取图片特征...")
    features_dict = extractor.extract_batch_features(
        [str(f) for f in image_files],
        show_progress=True,
        base_dir=image_dir
    )
    
    print(f"✓ 成功提取 {len(features_dict)} 张图片的特征")
    
    # 4. 初始化向量数据库管理器
    print(f"\n连接到向量数据库...")
    if use_docker_milvus:
        # 使用Docker中的Milvus
        vector_manager = MilvusManager(
            host="localhost",
            port=19532,
            collection_name=collection_name,
            dimension=feature_dim
        )
    elif use_milvus_lite:
        # 使用Milvus Lite
        vector_manager = MilvusLiteManager(
            db_path="./milvus_demo.db",
            collection_name=collection_name,
            dimension=feature_dim
        )
    else:
        # 使用本地存储
        vector_manager = LocalVectorManager(
            collection_name=collection_name,
            dimension=feature_dim
        )
    
    # 5. 创建集合
    print(f"\n创建集合: {collection_name}")
    vector_manager.create_collection(drop_existing=drop_existing)
    
    # 6. 批量插入特征
    print(f"\n插入特征向量到数据库...")
    batch_insert_images(
        milvus_manager=vector_manager,
        features_dict=features_dict,
        image_dir=image_dir
    )
    
    # 7. 创建索引
    print(f"\n创建索引...")
    vector_manager.create_index(index_type="HNSW")
    
    # 8. 加载集合到内存
    print(f"\n加载集合到内存...")
    vector_manager.load_collection()
    
    # 9. 显示统计信息
    stats = vector_manager.get_collection_stats()
    print(f"\n{'='*60}")
    print("数据库初始化完成")
    print(f"{'='*60}")
    print(f"集合名称: {stats['collection_name']}")
    print(f"向量维度: {stats['dimension']}")
    print(f"图片数量: {stats['num_entities']}")
    print(f"距离度量: {stats['metric_type']}")
    print(f"{'='*60}\n")
    
    return vector_manager


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="初始化图像特征数据库")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./img",
        help="图片目录路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["resnet50", "efficientnet_lite0", "mobilenet_v3_small", "convnext_tiny", "openclip_vit_b_32", "all"],
        help="使用的模型名称（all表示初始化所有模型）"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Milvus集合名称（不指定则使用默认名称）"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="删除已存在的集合并重建"
    )
    parser.add_argument(
        "--use-milvus-lite",
        action="store_true",
        default=False,
        help="使用 Milvus Lite（本地文件存储）"
    )
    parser.add_argument(
        "--use-docker-milvus",
        action="store_true",
        default=True,
        help="使用 Docker 中的 Milvus（默认）"
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="使用本地存储（JSON文件）"
    )
    
    args = parser.parse_args()
    
    # 确定使用哪种存储方式
    use_docker_milvus = args.use_docker_milvus
    use_milvus_lite = args.use_milvus_lite
    
    # 如果明确指定使用本地存储，则禁用其他选项
    if args.use_local:
        use_docker_milvus = False
        use_milvus_lite = False
    
    # 定义模型和对应的集合名称
    model_collections = {
        "resnet50": "image_features_resnet50",
        "efficientnet_lite0": "image_features_efficientnet_lite0",
        "mobilenet_v3_small": "image_features_mobilenet_v3_small",
        "convnext_tiny": "image_features_convnext_tiny",
        "openclip_vit_b_32": "image_features_openclip_vit_b_32"
    }
    
    try:
        if args.model == "all":
            # 初始化所有模型
            print(f"\n将初始化所有模型: {', '.join(model_collections.keys())}")
            for model_name, collection_name in model_collections.items():
                print(f"\n{'#'*60}")
                print(f"初始化模型: {model_name}")
                print(f"{'#'*60}")
                initialize_database(
                    image_dir=args.image_dir,
                    model_name=model_name,
                    collection_name=collection_name,
                    drop_existing=args.drop,
                    use_milvus_lite=use_milvus_lite,
                    use_docker_milvus=use_docker_milvus
                )
        else:
            # 初始化指定模型
            collection_name = args.collection if args.collection else model_collections[args.model]
            initialize_database(
                image_dir=args.image_dir,
                model_name=args.model,
                collection_name=collection_name,
                drop_existing=args.drop,
                use_milvus_lite=use_milvus_lite,
                use_docker_milvus=use_docker_milvus
            )
        
        print("✓ 初始化成功！")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
