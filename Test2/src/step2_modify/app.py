import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from feature import FeatureExtractor

app = Flask(__name__)
CORS(app)

# 配置路径
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IMG_DIR = os.path.join(BASE_DIR, 'img')
MODEL_CACHE_DIR = os.path.join(BASE_DIR, 'model_cache')

# 确保目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 初始化特征提取器
fe = FeatureExtractor('openclip_vit_b_32', cache_dir=MODEL_CACHE_DIR)

# 导入Milvus客户端
try:
    from pymilvus import connections, Collection, utility
except ImportError:
    print("错误: 请安装pymilvus库")
    sys.exit(1)

# 连接Milvus
try:
    connections.connect(
        alias="default",
        host="localhost",
        port="19532"
    )
    print("成功连接到Milvus")
except Exception as e:
    print(f"错误: 无法连接到Milvus: {e}")
    sys.exit(1)

# Milvus集合名称
COLLECTION_NAME = "image_features_openclip_vit_b_32"

# 首页路由
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# 静态文件路由
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# 图片路由
@app.route('/api/images/<path:path>')
def send_image(path):
    image_path = os.path.join(IMG_DIR, path)
    print(f'image_path: {image_path}')
    if os.path.exists(image_path):
        return send_from_directory(IMG_DIR, path)
    else:
        return jsonify({"error": "图片未找到"}), 404

# 搜索API路由
@app.route('/api/search', methods=['POST'])
def search_similar_images():
    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return jsonify({"error": "请上传图片"}), 400
        
        # 获取上传的文件
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "请选择图片"}), 400
        
        # 读取图片
        image = Image.open(file.stream)
        
        # 保存临时图片
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_image_path = temp_file.name
            image.save(temp_image_path, format='JPEG')
        
        try:
            # 提取特征
            feature = fe.extract_features(temp_image_path)
            
            # 特征归一化
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
        finally:
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
        
        # 转换为列表格式
        feature_list = feature.tolist()
        
        # 连接到Milvus集合
        if not utility.has_collection(COLLECTION_NAME):
            return jsonify({"error": f"集合 {COLLECTION_NAME} 不存在"}), 404
        
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # 执行相似度搜索
        search_params = {
            "metric_type": "COSINE",  # 余弦相似度
            "params": {"nprobe": 10}
        }
        
        # 搜索前10个最相似的结果
        results = collection.search(
            data=[feature_list],
            anns_field="feature_vector",
            param=search_params,
            
            limit=10,
            expr=None,
            output_fields=["image_path"]
        )
        
        # 处理搜索结果
        search_results = []
        for hits in results:
            for hit in hits:
                # 获取图片路径
                try:
                    # 尝试直接获取字段值
                    image_path = hit.entity.image_path
                except Exception:
                    try:
                        # 尝试使用get方法（不传递默认值）
                        image_path = hit.entity.get("image_path")
                        # print(f'image_path: {image_path}')
                    except Exception:
                        # 如果都失败，使用空字符串
                        image_path = ""
                
                if image_path:
                    # 构建图片URL
                    try:
                        # 确保image_path是绝对路径
                        if not os.path.isabs(image_path):
                            # 如果是相对路径，尝试构建绝对路径
                            image_path = os.path.join(IMG_DIR, image_path)
                        
                        # 计算相对于IMG_DIR的路径
                        relative_path = os.path.relpath(image_path, IMG_DIR)
                        # 构建正确的URL路径
                        image_url = f"/api/images/{relative_path.replace(os.sep, '/')}"
                        
                        search_results.append({
                            "url": image_url
                        })
                    except Exception as e:
                        print(f"构建图片URL时出错: {e}")
                        # 如果构建URL失败，跳过该结果
                        pass
        
        # 返回搜索结果
        return jsonify({
            "results": search_results
        })
        
    except Exception as e:
        print(f"错误: {e}")
        return jsonify({"error": f"搜索失败: {str(e)}"}), 500

if __name__ == '__main__':
    print("============================================================")
    print("启动后端服务器")
    print("============================================================")
    print(f"服务器地址: http://localhost:5008")
    print("============================================================")
    
    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5008,
        debug=True
    )