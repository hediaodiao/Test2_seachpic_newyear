#!/usr/bin/env python3
"""
模型评估脚本
评估图像相似度模型的Precision@5、Recall@5、mAP指标
支持候选集图片特征提取和向量数据库存储
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path

# 导入现有的模块
from feature import FeatureExtractor
from vector_db import MilvusManager, batch_insert_images

def calculate_precision_recall_at_k(query_results, relevant_images, k=5):
    """
    计算Precision@k和Recall@k
    
    参数:
        query_results: 查询结果列表，每个元素包含image_path
        relevant_images: 相关图像列表
        k: 评估的k值
        
    返回:
        precision_at_k: Precision@k
        recall_at_k: Recall@k
    """
    # 提取前k个结果的图像路径（只取文件名部分进行比较）
    retrieved_images = [os.path.basename(result['image_path']) for result in query_results[:k]]
    relevant_images = [os.path.basename(img) for img in relevant_images]
    
    # 计算TP（真阳性）
    tp = len(set(retrieved_images) & set(relevant_images))
    
    # 计算Precision@k
    precision_at_k = tp / k if k > 0 else 0
    
    # 计算Recall@k
    recall_at_k = tp / len(relevant_images) if len(relevant_images) > 0 else 0
    
    return precision_at_k, recall_at_k

def calculate_ap(query_results, relevant_images, max_k=5):
    """
    计算平均精度(AP)
    
    参数:
        query_results: 查询结果列表，每个元素包含image_path
        relevant_images: 相关图像列表
        max_k: 最大评估的k值
        
    返回:
        ap: 平均精度值
    """
    if not relevant_images:
        return 0.0
    
    retrieved_images = [os.path.basename(result['image_path']) for result in query_results[:max_k]]
    relevant_images = [os.path.basename(img) for img in relevant_images]
    
    relevant_set = set(relevant_images)
    ap = 0.0
    tp_count = 0
    
    for i, img in enumerate(retrieved_images):
        if img in relevant_set:
            tp_count += 1
            precision_at_i = tp_count / (i + 1)
            ap += precision_at_i
    
    # 使用实际找出的相关项数量作为分母（标准AP计算方法）
    if tp_count == 0:
        return 0.0
    return ap / tp_count

def evaluate_model(model_name, annotations_path, image_dir, device='auto'):
    """
    评估指定模型
    
    参数:
        model_name: 要评估的模型名称
        annotations_path: 标注文件路径
        image_dir: 图像目录路径
        device: 运行设备
        
    返回:
        results: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"🔍 开始评估模型: {model_name}")
    print(f"{'='*60}")
    
    # 1. 加载特征提取器
    print(f"\n📦 加载特征提取器...")
    feature_extractor = FeatureExtractor(model_name=model_name, device=device)
    print(f"✅ 模型加载完成: {feature_extractor.get_model_info()}")
    
    # 2. 加载标注文件
    print(f"\n📄 加载标注文件...")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    print(f"✅ 加载完成: {len(annotations)}个查询样本")
    
    # 3. 连接向量数据库
    print(f"\n🗄️ 连接向量数据库...")
    
    # 模型对应的集合名称（与init_db.py保持一致）
    collection_name = f"image_features_{model_name}"
    
    # 使用Docker中的Milvus（使用余弦相似度）
    vector_db = MilvusManager(
        host="localhost",
        port=19532,
        collection_name=collection_name,
        dimension=feature_extractor.feature_dim,
        metric_type="COSINE"
    )
    
    # 检查集合是否存在（评估阶段集合应该已经存在）
    try:
        from pymilvus import utility
        if not utility.has_collection(collection_name):
            print(f"❌ 集合不存在: {collection_name}")
            print(f"⚠ 请先运行特征提取功能创建集合")
            print(f"   使用命令: python evaluate_model.py --model {model_name} --extract_candidates <候选集目录>")
            return None
        
        print(f"✅ 集合存在: {collection_name}")
        
    except Exception as e:
        print(f"❌ 检查集合存在性失败: {e}")
        return None
    
    # 加载集合（评估阶段集合应该已经存在）
    try:
        # 尝试创建集合（如果不存在）
        vector_db.create_collection(drop_existing=False)
        
        # 检查索引是否存在
        try:
            # 检查索引是否存在
            vector_db.create_index()
            print(f"✅ 索引已存在或创建成功")
        except Exception as e:
            print(f"   检查索引存在性时出错: {e}")
        
        # 然后加载集合
        vector_db.load_collection()
        print(f"✅ 集合加载成功")
        print(f"✅ 数据库连接成功，集合名称: {collection_name}")
    except Exception as e:
        print(f"❌ 加载集合失败: {e}")
        print(f"⚠ 尝试继续执行，跳过索引和加载步骤...")
        # 继续执行，不返回错误
        print(f"✅ 数据库连接成功，集合名称: {collection_name}")
    
    # 4. 评估过程
    print(f"\n📊 开始评估...")
    
    total_queries = 0
    total_precision_at_5 = 0.0
    total_recall_at_5 = 0.0
    total_precision_at_10 = 0.0
    total_recall_at_10 = 0.0
    total_ap = 0.0
    
    # 初始化类别指标字典
    category_metrics = {}
    
    for i, annotation in enumerate(annotations, 1):
        query_image = annotation['query_image']
        relevant_images = annotation['relevant_images']
        # 获取类别信息，默认为'unknown'
        category = annotation.get('category', 'unknown')
        
        # 跳过没有相关图像的查询（无法计算有意义的召回率）
        if not relevant_images:
            continue
        
        # 构建完整的查询图像路径（使用用户指定的image_dir）
        query_image_path = os.path.join(image_dir, query_image)
        
        if not os.path.exists(query_image_path):
            print(f"⚠ 跳过不存在的查询图像: {query_image}")
            print(f"   完整路径: {query_image_path}")
            continue
        
        try:
            # 提取查询图像的特征
            query_features = feature_extractor.extract_features(query_image_path)
            
            # L2归一化处理（用于余弦相似度）
            norm = np.linalg.norm(query_features)
            if norm > 0:
                query_features = query_features / norm
            
            # 在向量数据库中搜索前10个相似图像
            search_results = vector_db.search(query_vector=query_features, top_k=10)
            
            if not search_results:
                print(f"⚠ 查询 {query_image} 没有返回结果")
                continue
            
            # 计算指标
            precision_at_5, recall_at_5 = calculate_precision_recall_at_k(search_results, relevant_images, k=5)
            precision_at_10, recall_at_10 = calculate_precision_recall_at_k(search_results, relevant_images, k=10)
            ap = calculate_ap(search_results, relevant_images, max_k=10)
            
            # 累加总指标
            total_precision_at_5 += precision_at_5
            total_recall_at_5 += recall_at_5
            total_precision_at_10 += precision_at_10
            total_recall_at_10 += recall_at_10
            total_ap += ap
            total_queries += 1  # 只有成功完成评估的样本才计入
            
            # 累加类别指标
            if category not in category_metrics:
                category_metrics[category] = {
                    'count': 0,
                    'precision_at_5': 0.0,
                    'recall_at_5': 0.0,
                    'precision_at_10': 0.0,
                    'recall_at_10': 0.0,
                    'ap': 0.0
                }
            
            category_metrics[category]['count'] += 1
            category_metrics[category]['precision_at_5'] += precision_at_5
            category_metrics[category]['recall_at_5'] += recall_at_5
            category_metrics[category]['precision_at_10'] += precision_at_10
            category_metrics[category]['recall_at_10'] += recall_at_10
            category_metrics[category]['ap'] += ap
            
            if i % 50 == 0:
                print(f"  进度: {i}/{len(annotations)} 查询完成")
                print(f"  当前平均 - Precision@5: {total_precision_at_5/total_queries:.4f}, Recall@5: {total_recall_at_5/total_queries:.4f}")
                print(f"  当前平均 - Precision@10: {total_precision_at_10/total_queries:.4f}, Recall@10: {total_recall_at_10/total_queries:.4f}")
                print(f"  当前平均 - mAP: {total_ap/total_queries:.4f}")
                
        except Exception as e:
            print(f"⚠ 处理查询 {query_image} 时出错: {e}")
            continue
    
    # 5. 计算平均指标
    print(f"\n{'='*60}")
    print(f"📈 评估结果")
    print(f"{'='*60}")
    
    if total_queries == 0:
        print("❌ 没有有效的查询样本")
        return None
    
    # 计算每个类别的平均指标
    category_results = {}
    for category, metrics in category_metrics.items():
        count = metrics['count']
        if count > 0:
            category_results[category] = {
                'count': count,
                'precision_at_5': metrics['precision_at_5'] / count,
                'recall_at_5': metrics['recall_at_5'] / count,
                'precision_at_10': metrics['precision_at_10'] / count,
                'recall_at_10': metrics['recall_at_10'] / count,
                'mAP': metrics['ap'] / count
            }
    
    # 打印每个类别的评估结果
    if category_results:
        print(f"\n{'='*60}")
        print(f"📋 按类别评估结果")
        print(f"{'='*60}")
        
        for category, result in category_results.items():
            print(f"\n📊 类别: {category}")
            print(f"├── 评估样本数: {result['count']}")
            print(f"├── Precision@5: {result['precision_at_5']:.4f}")
            print(f"├── Recall@5: {result['recall_at_5']:.4f}")
            print(f"├── Precision@10: {result['precision_at_10']:.4f}")
            print(f"├── Recall@10: {result['recall_at_10']:.4f}")
            print(f"└── mAP: {result['mAP']:.4f}")
    
    # 计算总平均指标
    avg_precision_at_5 = total_precision_at_5 / total_queries
    avg_recall_at_5 = total_recall_at_5 / total_queries
    avg_precision_at_10 = total_precision_at_10 / total_queries
    avg_recall_at_10 = total_recall_at_10 / total_queries
    mAP = total_ap / total_queries
    
    results = {
        'model_name': model_name,
        'total_queries': total_queries,
        'precision_at_5': avg_precision_at_5,
        'recall_at_5': avg_recall_at_5,
        'precision_at_10': avg_precision_at_10,
        'recall_at_10': avg_recall_at_10,
        'mAP': mAP,
        'category_results': category_results
    }
    
    # 打印总结果
    print(f"\n{'='*60}")
    print(f"📊 总评估结果")
    print(f"{'='*60}")
    print(f"模型名称: {model_name}")
    print(f"评估样本数: {total_queries}")
    print(f"Precision@5: {avg_precision_at_5:.4f}")
    print(f"Recall@5: {avg_recall_at_5:.4f}")
    print(f"Precision@10: {avg_precision_at_10:.4f}")
    print(f"Recall@10: {avg_recall_at_10:.4f}")
    print(f"mAP: {mAP:.4f}")
    print(f"{'='*60}")
    
    return results

def extract_candidate_features(model_name, candidate_dir, device='auto', overwrite=False, batch_size=32, refresh=False):
    """
    提取候选集图片特征并存储到向量数据库
    
    参数:
        model_name: 模型名称
        candidate_dir: 候选集图片目录
        device: 运行设备
        overwrite: 是否覆盖现有集合（True: 删除现有集合并重新创建, False: 增量更新）
        batch_size: 批量处理大小
        refresh: 是否刷新特征（True: 检查并更新新图片的特征, False: 不执行任何操作）
        
    返回:
        bool: 是否成功
    """
    # 如果未配置refresh参数，则不重新提取特征，也不更新Milvus数据库
    if not refresh:
        print(f"\n{'='*60}")
        print(f"📸 跳过特征提取：未配置refresh参数")
        print(f"{'='*60}")
        return True
    import os
    
    print(f"\n{'='*60}")
    print(f"📸 开始提取候选集图片特征")
    print(f"模型: {model_name}")
    print(f"候选集目录: {candidate_dir}")
    print(f"{'='*60}")
    
    # 1. 检查候选集目录是否存在
    if not os.path.exists(candidate_dir):
        print(f"❌ 候选集目录不存在: {candidate_dir}")
        return False
    
    # 2. 加载特征提取器
    print(f"\n📦 加载特征提取器...")
    try:
        feature_extractor = FeatureExtractor(model_name=model_name, device=device)
        print(f"✅ 模型加载完成: {feature_extractor.get_model_info()}")
    except Exception as e:
        print(f"❌ 特征提取器加载失败: {e}")
        return False
    
    # 3. 连接向量数据库
    print(f"\n🗄️ 连接向量数据库...")
    
    # 模型对应的集合名称
    collection_name = f"image_features_{model_name}"
    
    vector_db = MilvusManager(
        host="localhost",
        port=19532,
        collection_name=collection_name,
        dimension=feature_extractor.feature_dim,
        metric_type="COSINE"
    )
    
    # 初始化集合
    if overwrite:
        print(f"   模式: 覆盖现有集合,先删除现有集合再创建同名空集合")
        vector_db.create_collection(drop_existing=True)
        print(f"✅ 创建集合成功: {collection_name}")
    else:
        print(f"   模式: 增量更新现有集合")
        vector_db.create_collection(drop_existing=False)
        print(f"✅ 初始化集合成功: {collection_name}")
    
    # 注意：不立即加载集合，避免索引检查问题
    # 只有在需要搜索时才加载集合，插入数据时不需要加载集合
    print(f"✅ 数据库连接成功，集合名称: {collection_name}")
    
    # 4. 扫描候选集图片
    print(f"\n🔍 扫描候选集图片...")
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 收集所有图片文件
    all_candidate_images = []
    for root, dirs, files in os.walk(candidate_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                all_candidate_images.append(os.path.join(root, file))
    
    if not all_candidate_images:
        print(f"❌ 在目录 {candidate_dir} 中没有找到图片文件")
        return False
    
    print(f"✅ 找到 {len(all_candidate_images)} 张候选集图片")
    
    # 获取已存在的图片路径（增量更新模式）
    existing_images = set()
    if not overwrite:
        try:
            # 从向量数据库获取所有已存在的图片路径
            from pymilvus import connections, utility, Collection
            connections.connect(host='localhost', port=19532)
            collection = Collection(collection_name)
            
            # 检查索引是否存在（兼容旧版本pymilvus）
            has_index = False
            try:
                from pymilvus import utility
                # 尝试获取索引信息
                index_info = collection.indexes
                if index_info:
                    has_index = True
                    print(f"   ✅ 索引已存在")
                else:
                    has_index = False
            except Exception as e:
                print(f"   检查索引存在性时出错: {e}")
                has_index = False
            
            # 如果索引不存在，创建索引
            if not has_index:
                print(f"   索引不存在，正在创建索引...")
                # 创建索引
                index_params = {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 256}
                }
                collection.create_index(field_name="feature_vector", index_params=index_params)
                print(f"   ✅ 索引创建成功")
            
            # 尝试加载集合
            try:
                collection.load()
                loaded = True
            except Exception as load_error:
                print(f"   尝试加载集合失败: {load_error}")
                print(f"   继续执行，但无法获取已存在图片列表")
                loaded = False
            
            # 获取集合中的实体数量
            collection.flush()
            existing_entity_count = collection.num_entities
            
            # 如果加载成功，则执行查询
            if loaded:
                # 使用有效的表达式，避免空表达式错误
                expr = "id >= 0"  # 合法的条件，获取所有记录
                result = collection.query(expr=expr, output_fields=["image_path"], limit=existing_entity_count + 1000)
                
                for item in result:
                    existing_image_path = item.get("image_path", "")
                    if existing_image_path:
                        # 归一化路径格式，确保跨平台一致性
                        normalized_path = existing_image_path.replace('\\', '/')
                        existing_images.add(normalized_path)
                
                print(f"   已存在 {len(existing_images)} 张图片的特征")
                
                # 释放集合资源
                collection.release()
            else:
                # 如果加载失败，则假设没有已存在的图片
                print(f"   无法加载集合，假设没有已存在的图片")
                existing_images = set()
        except Exception as e:
            print(f"   获取已存在图片列表失败: {e}")
            print(f"   继续执行增量更新，假设没有已存在的图片")
            existing_images = set()
            # 不设置overwrite=True，继续使用增量更新模式
    
    # 执行数据库与实际图片同步（只在refresh模式下执行）
    if refresh:
        try:
            # 获取当前实际存在的图片路径
            current_images = set()
            for image_path in all_candidate_images:
                rel_path = os.path.relpath(image_path, candidate_dir)
                # 归一化路径格式，确保跨平台一致性
                normalized_path = rel_path.replace('\\', '/')
                current_images.add(normalized_path)
            
            # 找出数据库中存在但实际不存在的图片路径
            removed_images = existing_images - current_images
            
            if removed_images:
                print(f"   发现 {len(removed_images)} 张图片已从本地删除，将从数据库中移除对应的向量")
                
                # 连接到Milvus
                from pymilvus import connections, utility, Collection
                connections.connect(host='localhost', port=19532)
                collection = Collection(collection_name)
                
                # 加载集合到内存中（删除前必须加载集合）
                collection.load(skip_index_check=True)
                
                # 逐个删除数据库中不存在的图片对应的向量
                successfully_deleted = set()
                for image_path in removed_images:
                    print(f"   删除图片 {image_path} 对应的向量")
                    try:
                        # 同时尝试两种路径格式（正斜杠和反斜杠）
                        check_exprs = [
                            f"image_path == '{image_path}'",  # 正斜杠格式
                            f"image_path == '{image_path.replace('/', '\\\\')}'"  # 反斜杠格式
                        ]
                        
                        for check_expr in check_exprs:
                            # 使用相同的表达式删除图片
                            result = collection.delete(expr=check_expr)
                            if result.delete_count > 0:
                                print(f"   删除图片 {image_path} 对应的向量，删除行数: {result.delete_count}")
                                successfully_deleted.add(image_path)
                                break
                        
                        # 如果没有使用任何表达式删除成功，假设图片不存在
                        if image_path not in successfully_deleted:
                            successfully_deleted.add(image_path)
                    except Exception as delete_error:
                        print(f"   删除图片 {image_path} 时出错: {delete_error}")
                        # 即使删除失败，也从existing_images中移除，避免下次重复尝试
                        successfully_deleted.add(image_path)
                
                # 刷新集合，确保删除操作生效
                collection.flush()
                
                # 更新existing_images集合，移除已成功删除的图片
                existing_images -= successfully_deleted
                
                # 释放集合资源
                collection.release()
            else:
                print(f"   数据库与本地图片一致，无需删除操作")
        except Exception as e:
            print(f"   同步数据库与本地图片失败: {e}")
    
    # 过滤出需要处理的新图片
    candidate_images = []
    for image_path in all_candidate_images:
        rel_path = os.path.relpath(image_path, candidate_dir)
        # 归一化路径格式，确保跨平台一致性
        normalized_rel_path = rel_path.replace('\\', '/')
        if normalized_rel_path not in existing_images:
            candidate_images.append(image_path)
    
    if not candidate_images:
        print(f"✅ 所有图片特征已存在，无需更新")
        return True
    
    print(f"   需要处理 {len(candidate_images)} 张新图片")
    
    # 5. 批量插入图片特征
    print(f"\n💾 开始批量插入图片特征...")
    
    try:
        # 提取所有图片的特征，使用批量处理
        print(f"   使用批量处理进行特征提取，批量大小: {batch_size}")
        features_dict = feature_extractor.extract_batch_features(
            candidate_images,
            show_progress=True,
            base_dir=candidate_dir,
            batch_size=batch_size
        )
        
        # 统计成功和失败的数量
        success_count = len(features_dict)
        error_count = len(candidate_images) - success_count
    
        # 批量插入特征到向量数据库
        if features_dict:
            # 分批插入特征
            batch_size = 128
            features_list = list(features_dict.values())
            image_paths_list = list(features_dict.keys())
            
            for i in range(0, len(features_list), batch_size):
                batch_features = features_list[i:i + batch_size]
                batch_paths = image_paths_list[i:i + batch_size]
                
                # 对特征进行L2归一化处理（用于余弦相似度）
                normalized_batch_features = []
                for feature in batch_features:
                    norm = np.linalg.norm(feature)
                    if norm > 0:
                        normalized_feature = feature / norm
                    else:
                        normalized_feature = feature
                    normalized_batch_features.append(normalized_feature)
                
                # 插入到向量数据库
                vector_db.insert_features(normalized_batch_features, batch_paths, batch_paths)
                
                print(f"   插入批次 {i//batch_size + 1}/{(len(features_list) + batch_size - 1)//batch_size} 完成")
        
        print(f"✅ 特征提取完成!")
        print(f"   成功: {success_count} 张图片")
        print(f"   失败: {error_count} 张图片")
        
        if success_count > 0:
            print(f"\n📊 数据库统计信息:")
            try:
                stats = vector_db.get_collection_stats()
                print(f"   集合名称: {stats.get('collection_name', 'N/A')}")
                print(f"   向量数量: {stats.get('num_entities', 0)}")
                print(f"   向量维度: {stats.get('dimension', 0)}")
            except Exception as e:
                print(f"⚠ 无法获取统计信息: {e}")
        
        return success_count > 0
    
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='评估图像相似度模型')
    parser.add_argument('--model', type=str, required=True, help='要评估的模型名称')
    parser.add_argument('--annotations', type=str, default='similarity_annotations.json', help='标注文件路径')
    parser.add_argument('--image_dir', type=str, default='querySet', help='查询集图像目录路径')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备 (auto, cpu, cuda, mps)')
    parser.add_argument('--extract_candidates', type=str, help='候选集图片目录路径（如果提供，则只提取特征不进行评估）')
    parser.add_argument('--overwrite', action='store_true', help='是否覆盖现有集合（True: 删除现有集合并重新创建, False: 增量更新）')
    parser.add_argument('--batch_size', type=int, default=8, help='批量处理大小（默认: 8）')
    parser.add_argument('--refresh', action='store_true', help='是否刷新特征（True: 检查并更新新图片的特征, False: 不执行任何操作）')
    """
    python evaluate_model.py --model resnet50 --extract_candidates img --refresh
    pyton evaluate_model.py --model resnet50 
    """
    
    args = parser.parse_args()
    
    # 如果指定了候选集目录，则只提取特征
    if args.extract_candidates:
        success = extract_candidate_features(
            model_name=args.model,
            candidate_dir=args.extract_candidates,
            device=args.device,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
            refresh=args.refresh
        )
        if success:
            print("\n✅ 候选集特征提取完成!")
        else:
            print("\n❌ 候选集特征提取失败!")
            
        return
    
    # 验证输入路径
    if not os.path.exists(args.annotations):
        print(f"❌ 标注文件不存在: {args.annotations}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"❌ 图像目录不存在: {args.image_dir}")
        return
    
    # 执行评估
    results = evaluate_model(
        model_name=args.model,
        annotations_path=args.annotations,
        image_dir=args.image_dir,
        device=args.device
    )
    
    if results:
        print("\n✅ 评估完成!")
    else:
        print("\n❌ 评估失败!")

if __name__ == "__main__":
    main() 