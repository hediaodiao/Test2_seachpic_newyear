#!/usr/bin/env python3
"""
验证模型在数据增强上的性能指标
使用增强图片作为查询，在Milvus数据库中查找相似度top-k的图片，评估模型性能
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from feature import FeatureExtractor

# 尝试导入Milvus客户端
try:
    from pymilvus import connections, Collection, utility
except ImportError:
    print("警告: 未安装pymilvus库，请先安装: pip install pymilvus")
    sys.exit(1)


def parse_args():
    """
    解析命令行参数
    python validate_augmented.py --input_dir ../../querySet_step2_profile --original_dir ../../querySet_step2 --output_dir ../../querySet_step2_profile/evaluation_results --model resnet50 --augmentation_type rotation --rotation_angle 5
    python validate_augmented.py --augmentation_type rotation --rotation_angle 10
    python validate_augmented.py --augmentation_type cutting --crop_type offset --crop_ratio 0.8 --offset_x_percent 10 --offset_y_percent 10
    """
    parser = argparse.ArgumentParser(description='验证模型在数据增强上的性能指标')
    parser.add_argument('--input_dir', type=str, default='../../querySet_step2_profile',
                        help='增强图片输入目录')
    parser.add_argument('--original_dir', type=str, default='../../querySet_step2',
                        help='原始图片目录')
    parser.add_argument('--output_dir', type=str, default='../../querySet_step2_profile/evaluation_results',
                        help='评估结果输出目录')
    parser.add_argument('--model', type=str, nargs='*', default=['resnet50'],
                        choices=['resnet50', 'efficientnet_lite0', 'mobilenet_v3_small', 'convnext_tiny', 'openclip_vit_b_32', 'dinov2_vit_s'],
                        help='使用的特征提取模型，多个值用空格分隔，如：--model resnet50 efficientnet_lite0')
    parser.add_argument('--augmentation_type', type=str, default='',
                        choices=['', 'rotation', 'cutting', 'rotation_cutting'],
                        help='数据增强类型')
    parser.add_argument('--rotation_angle', type=int, nargs='*', default=[],
                        help='旋转角度列表，多个值用空格分隔，如：--rotation_angle 5 10 15')
    parser.add_argument('--crop_type', type=str, default='',
                        choices=['', 'center', 'offset'],
                        help='裁剪类型')
    parser.add_argument('--crop_ratio', type=float, default=0.0,
                        help='裁剪比例（0表示不指定）')
    parser.add_argument('--offset_x_percent', type=int, default=0,
                        help='X轴偏移百分比，用于偏移裁剪')
    parser.add_argument('--offset_y_percent', type=int, default=0,
                        help='Y轴偏移百分比，用于偏移裁剪')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='使用的设备')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量处理大小')
    return parser.parse_args()


def ensure_dir(directory):
    """
    确保目录存在
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_json_mappings(input_dir, augmentation_type='', rotation_angle=0, crop_type='', crop_ratio=0.0, offset_x_percent=0, offset_y_percent=0):
    """
    加载JSON文件，构建变体图片路径到原始图片信息的映射
    
    参数:
        input_dir: 输入目录，包含JSON文件
        augmentation_type: 数据增强类型
        rotation_angle: 旋转角度
        crop_type: 裁剪类型
        crop_ratio: 裁剪比例
        offset_x_percent: X轴偏移百分比
        offset_y_percent: Y轴偏移百分比
    
    返回:
        dict: 变体图片路径到原始图片信息的映射
    """
    mappings = {}
    
    # 根据增强类型和参数构建JSON文件路径
    json_paths = []
    
    if augmentation_type == 'rotation':
        # 旋转增强的JSON文件路径
        json_dir = os.path.join(input_dir, 'rotation', f"angle_{rotation_angle}")
        json_filename = f"augmentation_annotations_rotation_angle_{rotation_angle}.json"
        json_path = os.path.join(json_dir, json_filename)
        json_paths.append(json_path)
    elif augmentation_type == 'cutting' and crop_type and crop_ratio > 0:
        # 裁剪增强的JSON文件路径
        if crop_type == 'center':
            crop_subdir = f"offset{int(crop_ratio * 100)}_center"
            json_filename = f"augmentation_annotations_cutting_offset{int(crop_ratio * 100)}_center.json"
        elif crop_type == 'offset':
            crop_subdir = f"offset{int(crop_ratio * 100)}_offset_x{offset_x_percent}_y{offset_y_percent}"
            json_filename = f"augmentation_annotations_cutting_offset{int(crop_ratio * 100)}_offset_x{offset_x_percent}_y{offset_y_percent}.json"
        else:
            crop_subdir = f"{crop_type}_{int(crop_ratio * 100)}"
            json_filename = f"augmentation_annotations_cutting_{crop_type}_{int(crop_ratio * 100)}.json"
        json_dir = os.path.join(input_dir, 'cutting', crop_subdir)
        json_path = os.path.join(json_dir, json_filename)
        json_paths.append(json_path)
    elif augmentation_type == 'rotation_cutting' and crop_type and crop_ratio > 0:
        # 旋转和裁剪组合增强的JSON文件路径
        if crop_type == 'center':
            crop_subdir = f"offset{int(crop_ratio * 100)}_center_angle{rotation_angle}"
            json_filename = f"augmentation_annotations_rotation_cutting_offset{int(crop_ratio * 100)}_center_angle{rotation_angle}.json"
        elif crop_type == 'offset':
            crop_subdir = f"offset{int(crop_ratio * 100)}_offset_x{offset_x_percent}_y{offset_y_percent}_angle{rotation_angle}"
            json_filename = f"augmentation_annotations_rotation_cutting_offset{int(crop_ratio * 100)}_offset_x{offset_x_percent}_y{offset_y_percent}_angle{rotation_angle}.json"
        else:
            crop_subdir = f"{crop_type}_{int(crop_ratio * 100)}_angle{rotation_angle}"
            json_filename = f"augmentation_annotations_rotation_cutting_{crop_type}_{int(crop_ratio * 100)}_angle{rotation_angle}.json"
        json_dir = os.path.join(input_dir, 'rotation_cutting', crop_subdir)
        json_path = os.path.join(json_dir, json_filename)
        json_paths.append(json_path)
    else:
        # 如果没有指定增强类型或参数，遍历所有可能的JSON文件
        print("未指定增强类型或参数，遍历所有JSON文件...")
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.json') and 'augmentation_annotations' in file:
                    json_paths.append(os.path.join(root, file))
    
    # 加载每个JSON文件
    for json_path in json_paths:
        if os.path.exists(json_path):
            print(f"加载JSON文件: {json_path}")
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 解析变体信息
                if 'variants' in data:
                    for variant in data['variants']:
                        # 获取变体路径
                        variant_path = variant.get('variant_path', '')
                        if variant_path:
                            # 构建完整的变体图片路径
                            if variant_path.startswith('/'):
                                variant_path = variant_path[1:]  # 移除开头的斜杠
                            full_variant_path = os.path.join(input_dir, variant_path.replace('/', os.sep))
                            
                            # 存储原始图片信息
                            mappings[full_variant_path] = {
                                'original_id': variant.get('original_id', ''),
                                'original_path': variant.get('original_path', ''),
                                'category': variant.get('category', '')
                            }
                
            except Exception as e:
                print(f"加载JSON文件 {json_path} 时出错: {e}")
        else:
            print(f"JSON文件不存在: {json_path}")
    
    print(f"加载了 {len(mappings)} 个变体图片映射")
    return mappings


def get_original_image_path(augmented_path, original_dir, mappings):
    """
    根据增强图片路径获取原始图片路径
    
    参数:
        augmented_path: 增强图片路径
        original_dir: 原始图片目录
        mappings: 变体图片路径到原始图片信息的映射
    
    返回:
        原始图片路径
    """
    # 首先尝试从映射中获取原始图片信息
    if augmented_path in mappings:
        original_info = mappings[augmented_path]
        original_path = original_info.get('original_path', '')
        
        # 如果原始路径是相对路径，构建完整路径
        if original_path:
            if not os.path.isabs(original_path):
                # 尝试从原始目录构建路径
                parts = original_path.split('/')
                if len(parts) > 1 and parts[0] in ['animal_model', 'buildingBlock', 'plush_toy', 'remote_control_car', 'watergun']:
                    return os.path.join(original_dir, parts[0], parts[1])
                else:
                    # 尝试从文件名构建路径
                    filename = os.path.basename(original_path)
                    # 从增强图片路径中提取类别信息
                    parts = augmented_path.split(os.sep)
                    category = None
                    for part in parts:
                        if part in ['animal_model', 'buildingBlock', 'plush_toy', 'remote_control_car', 'watergun']:
                            category = part
                            break
                    if category:
                        return os.path.join(original_dir, category, filename)
    
    # 如果映射中没有，使用旧的逻辑作为后备
    filename = os.path.basename(augmented_path)
    if '_rot_' in filename:
        original_filename = filename.split('_rot_')[0] + '.jpg'
    elif '_crop_' in filename:
        original_filename = filename.split('_crop_')[0] + '.jpg'
    else:
        original_filename = filename
    
    # 从增强图片路径中提取类别信息
    parts = augmented_path.split(os.sep)
    category = None
    for part in parts:
        if part in ['animal_model', 'buildingBlock', 'plush_toy', 'remote_control_car', 'watergun']:
            category = part
            break
    
    if category:
        return os.path.join(original_dir, category, original_filename)
    else:
        # 如果找不到类别，尝试在所有类别中查找
        for cat in ['animal_model', 'buildingBlock', 'plush_toy', 'remote_control_car', 'watergun']:
            cat_path = os.path.join(original_dir, cat, original_filename)
            if os.path.exists(cat_path):
                return cat_path
    
    return None


def calculate_metrics(results, total_query_count):
    """
    计算评估指标
    
    参数:
        results: 评估结果列表，每个元素包含 {'rank': 排名}
        total_query_count: 查询的总图数，包括未找到的图片
    
    返回:
        包含各指标的字典
    """
    successful_queries = len(results)
    if total_query_count == 0:
        return {
            'top1_accuracy': 0.0,
            'top5_recall': 0.0,
            'top10_recall': 0.0,
            'top20_recall': 0.0,
            'mrr': 0.0,
            'total_queries': 0,
            'successful_queries': 0
        }
    
    # 计算 Top-1 准确率
    top1_correct = sum(1 for r in results if r['rank'] == 1)
    top1_accuracy = top1_correct / total_query_count
    
    # 计算 Top-K 召回率
    top5_correct = sum(1 for r in results if r['rank'] <= 5)
    top5_recall = top5_correct / total_query_count
    
    top10_correct = sum(1 for r in results if r['rank'] <= 10)
    top10_recall = top10_correct / total_query_count
    
    top20_correct = sum(1 for r in results if r['rank'] <= 20)
    top20_recall = top20_correct / total_query_count
    
    # 计算 MRR（分母是查询的总图数）
    mrr_sum = sum(1.0 / r['rank'] for r in results if r['rank'] > 0)
    mrr = mrr_sum / total_query_count
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_recall': top5_recall,
        'top10_recall': top10_recall,
        'top20_recall': top20_recall,
        'mrr': mrr,
        'total_queries': total_query_count,
        'successful_queries': successful_queries
    }


def find_augmented_images(input_dir, augmentation_type='', rotation_angle=0, crop_type='', crop_ratio=0.0, offset_x_percent=0, offset_y_percent=0):
    """
    查找符合条件的增强图片
    
    参数:
        input_dir: 输入目录
        augmentation_type: 数据增强类型
        rotation_angle: 旋转角度
        crop_type: 裁剪类型
        crop_ratio: 裁剪比例
        offset_x_percent: X轴偏移百分比
        offset_y_percent: Y轴偏移百分比
    
    返回:
        增强图片路径列表
    """
    augmented_images = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 检查是否符合数据增强类型条件
                if augmentation_type:
                    if augmentation_type == 'rotation' and 'rotation' not in root:
                        continue
                    if augmentation_type == 'cutting' and 'cutting' not in root:
                        continue
                    if augmentation_type == 'rotation_cutting' and 'rotation_cutting' not in root:
                        continue
                
                # 检查旋转角度条件
                if augmentation_type == 'rotation':
                    if f'angle_{rotation_angle}' not in root:
                        continue
                elif augmentation_type == 'rotation_cutting':
                    if f'angle{rotation_angle}' not in root:
                        continue
                
                # 检查裁剪类型和比例条件
                if (augmentation_type == 'cutting' or augmentation_type == 'rotation_cutting') and crop_type and crop_ratio > 0:
                    if crop_type == 'center':
                        crop_dir = f'offset{int(crop_ratio * 100)}_center'
                    elif crop_type == 'offset':
                        crop_dir = f'offset{int(crop_ratio * 100)}_offset_x{offset_x_percent}_y{offset_y_percent}'
                    else:
                        crop_dir = f'{crop_type}_{int(crop_ratio * 100)}'
                    if crop_dir not in root:
                        continue
                
                augmented_images.append(os.path.join(root, file))
    
    return augmented_images


def initialize_milvus(model_name):
    """
    初始化Milvus连接
    
    返回:
        Collection: Milvus集合对象
    """
    try:
        # 连接到Milvus服务
        connections.connect(
            alias="default",
            host="localhost",  # Docker容器的IP地址
            port="19532"         # Milvus默认端口
        )
        print("成功连接到Milvus数据库")
        
        # 指定集合名称
        collection_name = f"image_features_{model_name}"
        
        # 检查集合是否存在
        if not utility.has_collection(collection_name):
            print(f"警告: 集合 {collection_name} 不存在")
            print("警告: Milvus集合不存在，将使用本地特征匹配作为后备")
            return None
        
        # 获取集合对象
        collection = Collection(collection_name)
        print(f"成功获取集合: {collection_name}")
        
        # 检查并创建索引
        print(f"检查集合 {collection_name} 的索引...")
        try:
            # 获取集合的索引信息
            indexes = collection.indexes
            print(f"当前索引数量: {len(indexes)}")
            
            # 如果没有索引，创建索引
            if len(indexes) == 0:
                print("集合没有索引，正在创建索引...")
                # 尝试不同的字段名
                for field_name in ["feature_vector", "vector", "embedding", "feature"]:
                    try:
                        # 创建索引
                        index_params = {
                            "index_type": "IVF_FLAT",
                            "params": {"nlist": 128},
                            "metric_type": "COSINE"
                        }
                        collection.create_index(
                            field_name=field_name,
                            index_params=index_params
                        )
                        print(f"成功为字段 {field_name} 创建索引")
                        break
                    except Exception as index_error:
                        print(f"为字段 {field_name} 创建索引时出错: {index_error}")
                        continue
            else:
                print("集合已有索引，跳过索引创建")
        except Exception as index_check_error:
            print(f"检查索引时出错: {index_check_error}")
        
        # 尝试加载集合到内存
        print(f"尝试加载集合 {collection_name} 到内存...")
        try:
            collection.load()
            print(f"集合 {collection_name} 成功加载到内存")
        except Exception as load_error:
            print(f"加载集合时出错: {load_error}")
            print("注意: 集合可能在首次搜索时自动加载")
        
        return collection
        
    except Exception as e:
        print(f"连接Milvus数据库时出错: {e}")
        print("警告: Milvus连接失败，将使用本地特征匹配作为后备")
        return None


def search_milvus(collection, feature, top_k=20, anns_field="feature_vector", output_fields=["image_path"]):
    """
    在Milvus中搜索相似度top-k的图片
    
    参数:
        collection: Milvus集合对象
        feature: 查询特征向量
        top_k: 返回的结果数量
        anns_field: 特征字段名
        output_fields: 返回的字段列表
    
    返回:
        list: 包含id和距离的搜索结果
    """
    if not collection:
        print("警告: Milvus集合对象为None，搜索失败")
        return []
    
    try:
        # 尝试加载集合到内存，最多重试3次
        max_retries = 3
        for retry in range(max_retries):
            try:
                print(f"尝试加载集合到内存 (尝试 {retry+1}/{max_retries})...")
                collection.load()
                print("集合成功加载到内存")
                break
            except Exception as load_error:
                print(f"尝试加载集合失败: {load_error}")
                if retry == max_retries - 1:
                    print("已达到最大重试次数，继续执行搜索")
                else:
                    import time
                    time.sleep(1)  # 等待1秒后重试
        
        # 设置搜索参数
        search_params = {
            "metric_type": "COSINE",  # 距离度量类型
            "params": {"nprobe": 10}  # 搜索参数
        }
        
        # 执行搜索
        print("执行Milvus搜索...")
        results = collection.search(
            data=[feature],
            anns_field=anns_field,  # 特征字段名
            param=search_params,
            limit=top_k,
            expr=None,  # 可选的过滤表达式
            output_fields=output_fields  # 返回的字段
        )
        
        # 处理搜索结果
        search_results = []
        for hits in results:
            for hit in hits:
                # 尝试获取图片路径，支持不同的字段名
                image_path = ""
                for field in output_fields:
                    try:
                        # 尝试直接获取字段值
                        if hasattr(hit.entity, field):
                            image_path = getattr(hit.entity, field)
                            break
                        # 尝试使用get方法
                        elif hasattr(hit.entity, 'get'):
                            field_value = hit.entity.get(field)
                            if field_value is not None:
                                image_path = field_value
                                break
                    except Exception as e:
                        # 忽略获取字段时的错误
                        pass
                
                search_results.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "image_path": image_path
                })
        
        print(f"Milvus搜索成功，找到 {len(search_results)} 个结果")
        return search_results
        
    except Exception as e:
        print(f"在Milvus中搜索时出错: {e}")
        # 尝试使用不同的字段名
        print("尝试使用不同的字段名...")
        
        # 尝试常见的字段名组合
        field_combinations = [
            ("vector", ["path"]),
            ("embedding", ["path"]),
            ("feature", ["image_path"]),
            ("vector", ["image_path"]),
            ("embedding", ["image_path"])
        ]
        
        for anns_field, output_fields in field_combinations:
            try:
                print(f"尝试使用字段: anns_field={anns_field}, output_fields={output_fields}")
                # 设置搜索参数
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10}
                }
                
                # 执行搜索
                results = collection.search(
                    data=[feature],
                    anns_field=anns_field,
                    param=search_params,
                    limit=top_k,
                    expr=None,
                    output_fields=output_fields
                )
                
                # 处理搜索结果
                search_results = []
                for hits in results:
                    for hit in hits:
                        # 尝试获取图片路径
                        image_path = ""
                        for field in output_fields:
                            try:
                                if hasattr(hit.entity, field):
                                    image_path = getattr(hit.entity, field)
                                    break
                                elif hasattr(hit.entity, 'get'):
                                    field_value = hit.entity.get(field)
                                    if field_value is not None:
                                        image_path = field_value
                                        break
                            except Exception:
                                pass
                        
                        search_results.append({
                            "id": hit.id,
                            "distance": hit.distance,
                            "image_path": image_path
                        })
                
                print(f"成功使用字段: anns_field={anns_field}, output_fields={output_fields}")
                return search_results
                
            except Exception as e:
                print(f"尝试使用字段组合失败: {e}")
        
        print("所有字段组合尝试失败，搜索失败")
        return []


def load_original_features(original_dir, model_name, device):
    """
    加载原始图片特征
    
    参数:
        original_dir: 原始图片目录
        model_name: 模型名称
        device: 设备
    
    返回:
        原始图片特征字典 {图片路径: 特征向量}
    """
    try:
        # 遍历原始图片目录
        original_images = []
        for root, dirs, files in os.walk(original_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    original_images.append(os.path.join(root, file))
        
        print(f"找到 {len(original_images)} 张原始图片")
        
        # 如果没有原始图片，返回空字典
        if len(original_images) == 0:
            print("警告: 没有找到原始图片")
            return {}
        
        # 计算项目根目录路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # 模型缓存目录
        model_cache_dir = os.path.join(project_root, 'model_cache')
        
        # 提取特征
        extractor = FeatureExtractor(model_name, device, cache_dir=model_cache_dir)
        
        # 限制批量处理大小，避免资源竞争
        max_workers = min(4, os.cpu_count())
        print(f"使用 {max_workers} 个线程提取特征")
        
        # 分批处理，每批32张图片
        batch_size = 32
        features_dict = {}
        
        for i in range(0, len(original_images), batch_size):
            batch = original_images[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(original_images) + batch_size - 1)//batch_size}")
            
            # 单线程处理，避免线程竞争
            batch_features = {}
            for img_path in batch:
                try:
                    feature = extractor.extract_features(img_path)
                    # 对特征进行L2归一化
                    norm = np.linalg.norm(feature)
                    if norm > 0:
                        feature = feature / norm
                    batch_features[img_path] = feature
                except Exception as e:
                    print(f"提取图片 {img_path} 特征时出错: {e}")
            
            features_dict.update(batch_features)
        
        return features_dict
        
    except Exception as e:
        print(f"加载原始图片特征时出错: {e}")
        # 返回空字典，让程序继续执行
        return {}


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 确保输出目录存在
    ensure_dir(args.output_dir)
    
    # 处理旋转角度
    rotation_angles = args.rotation_angle
    if not rotation_angles:
        # 如果没有指定旋转角度，使用默认值0
        rotation_angles = [0]
    
    # 对每个模型单独处理
    for model_name in args.model:
        print(f"=" * 60)
        print(f"验证模型: {model_name}")
        print(f"=" * 60)
        
        # 对每个旋转角度单独处理
        for rotation_angle in rotation_angles:
            print(f"\n{'=' * 60}")
            print(f"处理旋转角度: {rotation_angle}")
            print(f"{'=' * 60}")
            
            # 查找符合条件的增强图片
            augmented_images = find_augmented_images(
                args.input_dir,
                args.augmentation_type,
                rotation_angle,
                args.crop_type,
                args.crop_ratio,
                args.offset_x_percent,
                args.offset_y_percent
            )
            
            print(f"找到 {len(augmented_images)} 张增强图片")
            
            if len(augmented_images) == 0:
                print("没有找到符合条件的增强图片")
                continue
            
            # 初始化结果列表
            results = []
            
            # 加载JSON文件，构建变体图片到原始图片的映射
            print("加载JSON文件，构建变体图片到原始图片的映射...")
            mappings = load_json_mappings(
                args.input_dir,
                args.augmentation_type,
                rotation_angle,
                args.crop_type,
                args.crop_ratio,
                args.offset_x_percent,
                args.offset_y_percent
            )
            
            # 初始化Milvus连接
            print("初始化Milvus连接...")
            collection = initialize_milvus(model_name)
            
            # 加载原始图片特征作为后备
            print("加载原始图片特征作为后备...")
            original_features = load_original_features(args.original_dir, model_name, args.device)
            print(f"成功加载 {len(original_features)} 张原始图片特征")
            
            # 初始化特征提取器
            print("初始化特征提取器...")
            # 计算项目根目录路径
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            # 模型缓存目录
            model_cache_dir = os.path.join(project_root, 'model_cache')
            extractor = FeatureExtractor(model_name, args.device, cache_dir=model_cache_dir)
            
            # 处理增强图片
            for i, augmented_path in enumerate(augmented_images, 1):
                print(f"处理图片 {i}/{len(augmented_images)}: {os.path.basename(augmented_path)}")
                
                # 获取原始图片路径
                original_path = get_original_image_path(augmented_path, args.original_dir, mappings)
                if not original_path or not os.path.exists(original_path):
                    print(f"警告: 找不到原始图片: {original_path}")
                    continue
                
                # 提取增强图片特征
                augmented_feature = extractor.extract_features(augmented_path)
                
                # 对特征进行L2归一化
                norm = np.linalg.norm(augmented_feature)
                if norm > 0:
                    augmented_feature = augmented_feature / norm
                
                # 在Milvus中搜索相似度top-k的图片
                print(f"在Milvus中搜索相似度top-20的图片...")
                search_results = search_milvus(collection, augmented_feature, top_k=20)
                
                # 如果Milvus搜索失败，使用本地特征匹配
                if not search_results:
                    print("Milvus搜索失败，使用本地特征匹配...")
                    # 计算本地特征相似度
                    local_results = []
                    for img_path, img_feature in original_features.items():
                        # 计算余弦相似度
                        similarity = np.dot(augmented_feature, img_feature)
                        local_results.append({
                            "id": 0,
                            "distance": similarity,
                            "image_path": img_path
                        })
                    
                    # 按相似度排序
                    local_results.sort(key=lambda x: x["distance"], reverse=True)
                    # 取top-k结果
                    search_results = local_results[:20]
                    print(f"本地特征匹配成功，找到 {len(search_results)} 个结果")
                
                if not search_results:
                    print(f"警告: 未找到搜索结果")
                    continue
                
                # 验证结果，计算排名
                rank = -1
                top_k_results = []
                
                for j, result in enumerate(search_results, 1):
                    # 构建返回图片的路径
                    returned_image_path = result.get('image_path', '')
                    top_k_results.append({
                        'id': result.get('id', ''),
                        'distance': result.get('distance', 0),
                        'image_path': returned_image_path
                    })
                    
                    # 检查是否返回了正确的原始图片
                    # 比较文件名（忽略路径差异）
                    if returned_image_path:
                        returned_filename = os.path.basename(returned_image_path)
                        original_filename = os.path.basename(original_path)
                        if returned_filename == original_filename:
                            rank = j
                            break
                
                # 如果找到正确的原始图片，添加结果
                if rank > 0:
                    results.append({
                        'augmented_path': augmented_path,
                        'original_path': original_path,
                        'rank': rank,
                        'top_k_results': top_k_results
                    })
                    print(f"✓ 找到正确的原始图片，排名: {rank}")
                else:
                    print(f"✗ 未找到正确的原始图片")
            
            # 计算指标（传递查询的总图数）
            total_query_count = len(augmented_images)
            metrics = calculate_metrics(results, total_query_count)
            
            # 生成评估结果
            evaluation_result = {
                'model': model_name,
                'augmentation_params': {
                    'type': args.augmentation_type,
                    'rotation_angle': rotation_angle,
                    'crop_type': args.crop_type,
                    'crop_ratio': args.crop_ratio,
                    'offset_x_percent': args.offset_x_percent,
                    'offset_y_percent': args.offset_y_percent
                },
                'metrics': metrics,
                'results': results
            }
            
            # 保存评估结果
            output_filename = f"evaluation_{model_name}"
            if args.augmentation_type:
                output_filename += f"_{args.augmentation_type}"
            if rotation_angle != 0:
                output_filename += f"_angle{rotation_angle}"
            if args.crop_type:
                if args.crop_type == 'center':
                    output_filename += f"_offset{int(args.crop_ratio * 100)}_center"
                elif args.crop_type == 'offset':
                    output_filename += f"_offset{int(args.crop_ratio * 100)}_x{args.offset_x_percent}_y{args.offset_y_percent}"
                else:
                    output_filename += f"_{args.crop_type}_{int(args.crop_ratio * 100)}"
            output_filename += ".json"
            
            output_path = os.path.join(args.output_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
            
            print(f"\n评估完成！")
            print(f"评估结果保存到: {output_path}")
            print(f"\n指标结果:")
            print(f"Top-1 准确率: {metrics['top1_accuracy']:.4f}")
            print(f"Top-5 召回率: {metrics['top5_recall']:.4f}")
            print(f"Top-10 召回率: {metrics['top10_recall']:.4f}")
            print(f"Top-20 召回率: {metrics['top20_recall']:.4f}")
            print(f"MRR: {metrics['mrr']:.4f}")
            print(f"总查询数: {metrics['total_queries']}")
            print(f"成功查询数: {metrics['successful_queries']}")


if __name__ == "__main__":
    main()