#!/usr/bin/env python3
"""
清理Milvus数据库中的重复记录
"""

from pymilvus import connections, Collection, utility
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_duplicate_records(collection_name):
    """
    清理指定集合中的重复记录
    
    参数:
        collection_name: 集合名称
    """
    print(f"\n{'='*60}")
    print(f"🧹 开始清理集合 {collection_name} 中的重复记录")
    print(f"{'='*60}")
    
    # 连接到Milvus服务器
    print("连接到Milvus服务器...")
    connections.connect(host='localhost', port=19531)
    
    # 检查集合是否存在
    if not utility.has_collection(collection_name):
        print(f"❌ 集合 {collection_name} 不存在")
        return False
    
    # 获取集合
    collection = Collection(collection_name)
    print(f"✅ 集合加载成功: {collection_name}")
    
    # 加载集合
    collection.load()
    
    # 获取集合中的实体数量
    collection.flush()
    total_entities = collection.num_entities
    print(f"📊 清理前集合中的实体数量: {total_entities}")
    
    # 查询所有图片路径
    print("查询所有图片路径...")
    result = collection.query(expr='id >= 0', output_fields=['id', 'image_path'])
    print(f"✅ 查询到 {len(result)} 条记录")
    
    # 找出重复的图片路径
    print("找出重复的图片路径...")
    path_to_ids = {}
    duplicate_paths = set()
    
    for item in result:
        image_path = item['image_path']
        image_id = item['id']
        
        if image_path not in path_to_ids:
            path_to_ids[image_path] = []
        path_to_ids[image_path].append(image_id)
        
        if len(path_to_ids[image_path]) > 1:
            duplicate_paths.add(image_path)
    
    print(f"📋 找到 {len(duplicate_paths)} 个重复的图片路径")
    
    # 删除重复记录，保留每个图片路径的第一个记录
    deleted_count = 0
    for path in duplicate_paths:
        ids = path_to_ids[path]
        # 保留第一个记录，删除其他记录
        ids_to_delete = ids[1:]
        
        if ids_to_delete:
            print(f"🔄 清理图片 {path} 的重复记录...")
            print(f"   图片 {path} 有 {len(ids)} 个重复记录，保留ID: {ids[0]}, 删除ID: {ids_to_delete}")
            
            # 逐个删除重复记录
            for image_id in ids_to_delete:
                expr = f"id == {image_id}"
                delete_result = collection.delete(expr=expr)
                deleted_count += delete_result.delete_count
                
                if delete_result.delete_count > 0:
                    print(f"   ✅ 成功删除ID: {image_id}")
                else:
                    print(f"   ❌ 删除ID: {image_id} 失败")
    
    # 刷新集合，确保删除操作生效
    collection.flush()
    
    # 释放集合资源
    collection.release()
    
    # 获取清理后的实体数量
    collection.load()
    collection.flush()
    final_entities = collection.num_entities
    collection.release()
    
    print(f"\n{'='*60}")
    print(f"🧹 清理完成")
    print(f"{'='*60}")
    print(f"📊 清理前实体数量: {total_entities}")
    print(f"📊 清理后实体数量: {final_entities}")
    print(f"🗑️  共删除 {deleted_count} 条重复记录")
    print(f"📈 减少了 {total_entities - final_entities} 条记录")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='清理Milvus数据库中的重复记录')
    parser.add_argument('--model', type=str, default='resnet50', help='模型名称')
    parser.add_argument('--collection', type=str, help='集合名称（如果提供，则忽略model参数）')
    
    args = parser.parse_args()
    
    if args.collection:
        collection_name = args.collection
    else:
        collection_name = f"image_features_{args.model}"
    
    cleanup_duplicate_records(collection_name)
