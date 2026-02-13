#!/usr/bin/env python3
"""
清空Milvus数据库中的resnet50集合
"""

from pymilvus import connections, Collection, utility

def clear_collection():
    """
    清空Milvus数据库中的image_features_resnet50集合
    """
    try:
        # 连接到Milvus服务器
        print("🔍 连接到Milvus服务器...")
        connections.connect(host='localhost', port=19531)
        print("✅ 连接成功")
        
        # 定义集合名称
        collection_name = "image_features_resnet50"
        
        # 检查集合是否存在
        if utility.has_collection(collection_name):
            print(f"📦 集合 {collection_name} 存在")
            
            # 获取集合对象
            collection = Collection(collection_name)
            
            # 获取集合中的实体数量
            collection.flush()
            entity_count = collection.num_entities
            print(f"📊 当前集合中的向量数量: {entity_count}")
            
            # 删除集合
            print(f"🗑️  正在删除集合 {collection_name}...")
            utility.drop_collection(collection_name)
            print(f"✅ 成功删除集合 {collection_name}")
            
            # 验证集合是否已删除
            if not utility.has_collection(collection_name):
                print(f"✅ 集合 {collection_name} 已成功删除")
            else:
                print(f"❌ 集合 {collection_name} 仍存在")
            
        else:
            print(f"❌ 集合 {collection_name} 不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 清空集合失败: {e}")
        return False
    
    finally:
        # 断开连接
        if connections.has_connection("default"):
            connections.disconnect("default")
            print("✅ 已断开与Milvus服务器的连接")

if __name__ == "__main__":
    print("=" * 60)
    print("清空Milvus数据库中的resnet50集合")
    print("=" * 60)
    
    success = clear_collection()
    
    if success:
        print("\n✅ 集合清空完成!")
    else:
        print("\n❌ 集合清空失败!")
