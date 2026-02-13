#!/usr/bin/env python3
"""
检查Milvus中的集合
"""

from pymilvus import connections, utility, Collection
import sys

def check_milvus_collections():
    """
    检查并显示Milvus中的所有集合
    """
    print("=" * 60)
    print("🔍 检查Milvus中的集合")
    print("=" * 60)
    
    try:
        # 连接到Milvus服务器
        print("正在连接到Milvus服务器...")
        connections.connect(host="localhost", port=19532)
        print("✅ 成功连接到Milvus服务器")
        
        # 列出所有集合
        print("\n📋 Milvus中的集合：")
        collections = utility.list_collections()
        
        if not collections:
            print("❌ 没有找到任何集合")
            return False
        print(f"Milvus中共有 {len(collections)} 个集合。")
        print("集合列表:", collections)
        
        for i, collection_name in enumerate(collections, 1):
            print(f"{i}. {collection_name}")
            
            # 获取集合信息
            try:
                collection = Collection(collection_name)
                collection.flush()
                num_entities = collection.num_entities
                
                # 检查索引
                has_index = False
                try:
                    index_info = collection.indexes
                    if index_info:
                        has_index = True
                except Exception:
                    pass
                
                print(f"   - 向量数量: {num_entities}")
                print(f"   - 是否已建立索引: {'是' if has_index else '否'}")
                
                # 尝试获取集合的第一个元素，查看结构
                try:
                    if num_entities > 0:
                        # 加载集合
                        collection.load()
                        # 查询前3个元素
                        result = collection.query(expr="id >= 0", limit=3, output_fields=["image_name", "image_path"])
                        print(f"   - 示例数据: {len(result)} 条")
                        for j, item in enumerate(result[:2]):
                            print(f"     {j+1}. 图片: {item.get('image_name', 'N/A')}")
                            print(f"        路径: {item.get('image_path', 'N/A')}")
                        if len(result) > 2:
                            print(f"     ... 等{len(result)}条记录")
                        # 释放集合
                        collection.release()
                except Exception as e:
                    print(f"   - 无法获取示例数据: {e}")
                    
            except Exception as e:
                print(f"   - 获取集合信息失败: {e}")
        
        print("\n" + "=" * 60)
        print("✅ 检查完成")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("请确保Milvus服务正在运行，并且可以通过 localhost:19532 访问")
        print("\n" + "=" * 60)
        print("❌ 检查失败")
        print("=" * 60)
        return False

if __name__ == "__main__":
    check_milvus_collections()
