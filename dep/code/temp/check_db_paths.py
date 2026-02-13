#!/usr/bin/env python3
"""
检查Milvus数据库中的图片路径
"""

from pymilvus import connections, Collection

# 连接到Milvus服务器
print("连接到Milvus服务器...")
connections.connect(host='localhost', port=19531)

# 获取集合
collection_name = 'image_features_resnet50'
collection = Collection(collection_name)
print(f"加载集合: {collection_name}")
collection.load()

# 查询数据库中的图片路径
print("查询数据库中的图片路径...")
result = collection.query(expr='id >= 0', output_fields=['image_path'], limit=10)

print("数据库中前10条图片路径:")
for item in result:
    print(f"  {item['image_path']}")

# 统计数据库中的图片数量
collection.flush()
print(f"\n数据库中的图片总数: {collection.num_entities}")

# 释放集合资源
collection.release()
print("完成检查!")
