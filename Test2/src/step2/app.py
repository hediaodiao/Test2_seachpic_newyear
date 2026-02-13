#!/usr/bin/env python3
"""
后端服务器
提供 RESTful API 接口，处理前端请求，返回评估结果
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path

# 创建 Flask 应用
app = Flask(__name__)

# 配置
app.config['DEBUG'] = True
app.config['JSON_AS_ASCII'] = False

# 获取当前文件的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 评估结果目录
EVALUATION_DIR = os.path.join(CURRENT_DIR, '../../querySet_step2_profile/evaluation_results')
EVALUATION_DIR = os.path.abspath(EVALUATION_DIR)

# 增强图片目录
AUGMENTED_DIR = os.path.join(CURRENT_DIR, '../../querySet_step2_profile')
AUGMENTED_DIR = os.path.abspath(AUGMENTED_DIR)

# 原始图片目录
ORIGINAL_DIR = os.path.join(CURRENT_DIR, '../../querySet_step2')
ORIGINAL_DIR = os.path.abspath(ORIGINAL_DIR)

# 新增：img 目录（top_K 结果图片存储位置）
IMG_DIR = os.path.join(CURRENT_DIR, '../../img')
IMG_DIR = os.path.abspath(IMG_DIR)

# 打印目录信息
print(f"评估结果目录: {EVALUATION_DIR}")
print(f"增强图片目录: {AUGMENTED_DIR}")
print(f"原始图片目录: {ORIGINAL_DIR}")
print(f"IMG目录: {IMG_DIR}")

# 确保目录存在
Path(EVALUATION_DIR).mkdir(parents=True, exist_ok=True)


@app.route('/')
def index():
    """
    首页路由
    """
    return send_from_directory('static', 'index.html')


@app.route('/api/models')
def get_models():
    """
    获取支持的模型列表
    """
    models = [
        'resnet50',
        'efficientnet_lite0',
        'mobilenet_v3_small',
        'convnext_tiny',
        'openclip_vit_b_32',
        'dinov2_vit_s'
    ]
    return jsonify({'models': models})


@app.route('/api/augmentation/types')
def get_augmentation_types():
    """
    获取数据增强类型列表
    """
    augmentation_types = [
        'rotation',
        'cutting',
        'rotation_cutting'
    ]
    return jsonify({'augmentation_types': augmentation_types})


@app.route('/api/augmentation/parameters')
def get_augmentation_parameters():
    """
    获取数据增强参数
    """
    # 获取类型参数
    type = request.args.get('type', '')
    # 新增：获取裁剪类型参数
    crop_type = request.args.get('crop_type', '')
    # 新增：获取裁剪比例参数
    crop_ratio = request.args.get('crop_ratio', 0.0, type=float)
    # 新增：获取旋转角度参数
    rotation_angle = request.args.get('rotation_angle', 0, type=int)
    
    # 打印调试日志
    print(f"接收到的参数: type={type}, crop_type={crop_type}, crop_ratio={crop_ratio}, rotation_angle={rotation_angle}")
    
    # 扫描目录获取实际的增强参数
    rotation_angles = []
    crop_types = []
    crop_ratios = []
    offset_x_percents = []
    offset_y_percents = []
    
    # 根据数据增强类型扫描对应目录
    if type == 'cutting':
        # 扫描裁剪参数
        cutting_dir = os.path.join(AUGMENTED_DIR, 'cutting')
        if os.path.exists(cutting_dir):
            for crop_dir in os.listdir(cutting_dir):
                if '_' in crop_dir:
                    # 处理新的目录命名规则
                    if 'offset' in crop_dir:
                        # 处理 offset{ratio}_center 格式
                        if '_center' in crop_dir:
                            current_crop_type = 'center'
                            # 只有当用户选择了 center 裁剪类型，或者没有选择裁剪类型时，才添加这些参数
                            if crop_type == '' or crop_type == current_crop_type:
                                try:
                                    # 提取比例：offset80_center -> 80
                                    parts = crop_dir.split('offset')
                                    if len(parts) > 1:
                                        ratio_part = parts[1].split('_center')[0]
                                        ratio_str = ratio_part.rstrip('_')
                                        current_crop_ratio = int(ratio_str) / 100.0
                                        
                                        # 只有当没有指定裁剪比例，或者指定的裁剪比例与当前目录匹配时，才添加
                                        if crop_ratio == 0.0 or abs(current_crop_ratio - crop_ratio) < 0.01:
                                            if current_crop_type not in crop_types:
                                                crop_types.append(current_crop_type)
                                            if current_crop_ratio not in crop_ratios:
                                                crop_ratios.append(current_crop_ratio)
                                except Exception as e:
                                    print(f"处理 center 裁剪目录 {crop_dir} 时出错: {str(e)}")
                        # 处理 offset{ratio}_offset_x{offset_x}_y{offset_y} 格式
                        elif '_offset_x' in crop_dir:
                            current_crop_type = 'offset'
                            # 只有当用户选择了 offset 裁剪类型，或者没有选择裁剪类型时，才添加这些参数
                            if crop_type == '' or crop_type == current_crop_type:
                                try:
                                    # 提取比例：offset80_offset_x10_y10 -> 80
                                    parts = crop_dir.split('offset')
                                    if len(parts) > 1:
                                        ratio_part = parts[1].split('_offset_x')[0]
                                        ratio_str = ratio_part.rstrip('_')
                                        current_crop_ratio = int(ratio_str) / 100.0
                                        
                                        # 提取偏移百分比
                                        offset_part = crop_dir.split('_offset_x')[1]
                                        if '_y' in offset_part:
                                            offset_x_str = offset_part.split('_y')[0]
                                            offset_y_str = offset_part.split('_y')[1]
                                            try:
                                                offset_x = int(offset_x_str)
                                                offset_y = int(offset_y_str)
                                                
                                                # 只有当没有指定裁剪比例，或者指定的裁剪比例与当前目录匹配时，才添加偏移百分比
                                                if crop_ratio == 0.0 or abs(current_crop_ratio - crop_ratio) < 0.01:
                                                    if current_crop_type not in crop_types:
                                                        crop_types.append(current_crop_type)
                                                    if current_crop_ratio not in crop_ratios:
                                                        crop_ratios.append(current_crop_ratio)
                                                    if offset_x not in offset_x_percents:
                                                        offset_x_percents.append(offset_x)
                                                    if offset_y not in offset_y_percents:
                                                        offset_y_percents.append(offset_y)
                                            except Exception as e:
                                                print(f"处理 offset 裁剪偏移参数时出错: {str(e)}")
                                except Exception as e:
                                    print(f"处理 offset 裁剪目录 {crop_dir} 时出错: {str(e)}")
                    # 保持对旧格式的兼容
                    else:
                        parts = crop_dir.split('_')
                        if len(parts) >= 2:
                            current_crop_type = parts[0]
                            # 只有当用户选择了对应的裁剪类型，或者没有选择裁剪类型时，才添加这些参数
                            if crop_type == '' or crop_type == current_crop_type:
                                try:
                                    current_crop_ratio = int(parts[1]) / 100.0
                                    # 只有当没有指定裁剪比例，或者指定的裁剪比例与当前目录匹配时，才添加
                                    if crop_ratio == 0.0 or abs(current_crop_ratio - crop_ratio) < 0.01:
                                        if current_crop_type not in crop_types:
                                            crop_types.append(current_crop_type)
                                        if current_crop_ratio not in crop_ratios:
                                            crop_ratios.append(current_crop_ratio)
                                except Exception as e:
                                    print(f"处理旧格式裁剪目录 {crop_dir} 时出错: {str(e)}")
    elif type == 'rotation_cutting':
        # 扫描旋转和裁剪组合参数
        rotation_cutting_dir = os.path.join(AUGMENTED_DIR, 'rotation_cutting')
        if os.path.exists(rotation_cutting_dir):
            for crop_dir in os.listdir(rotation_cutting_dir):
                if '_' in crop_dir:
                    # 打印当前处理的目录名称
                    print(f"处理目录: {crop_dir}")
                    # 处理新的目录命名规则
                    if 'offset' in crop_dir:
                        # 处理 offset{ratio}_center_angle{angle} 格式
                        if '_center_angle' in crop_dir:
                            # 只有当用户选择了 center 裁剪类型，或者没有选择裁剪类型时，才处理这些目录
                            if crop_type == '' or crop_type == 'center':
                                try:
                                    # 提取比例：offset80_center_angle30 -> 80
                                    parts = crop_dir.split('offset')
                                    if len(parts) > 1:
                                        ratio_part = parts[1].split('_center_angle')[0]
                                        ratio_str = ratio_part.rstrip('_')
                                        current_crop_ratio = int(ratio_str) / 100.0
                                        current_crop_type = 'center'
                                        
                                        # 提取角度：offset80_center_angle30 -> 30
                                        angle_part = crop_dir.split('_center_angle')[1]
                                        try:
                                            angle = int(angle_part)
                                            print(f"提取到的角度: {angle}, 当前旋转角度: {rotation_angle}")
                                            
                                            # 只有当没有指定旋转角度，或者指定的旋转角度与当前目录匹配时，才添加
                                            if rotation_angle == 0 or rotation_angle == angle:
                                                print(f"角度匹配: {angle}")
                                                # 只有当没有指定裁剪比例，或者指定的裁剪比例与当前目录匹配时，才添加
                                                if crop_ratio == 0.0 or abs(current_crop_ratio - crop_ratio) < 0.01:
                                                    print(f"添加 center 类型，比例: {current_crop_ratio}")
                                                    if current_crop_type not in crop_types:
                                                        crop_types.append(current_crop_type)
                                                    if current_crop_ratio not in crop_ratios:
                                                        crop_ratios.append(current_crop_ratio)
                                                    if angle not in rotation_angles:
                                                        rotation_angles.append(angle)
                                        except Exception as e:
                                            print(f"处理 center 裁剪角度时出错: {str(e)}")
                                except Exception as e:
                                    print(f"处理 center 裁剪目录 {crop_dir} 时出错: {str(e)}")
                        # 处理 offset{ratio}_offset_x{offset_x}_y{offset_y}_angle{angle} 格式
                        elif '_offset_x' in crop_dir and '_angle' in crop_dir:
                            # 只有当用户选择了 offset 裁剪类型，或者没有选择裁剪类型时，才处理这些目录
                            if crop_type == '' or crop_type == 'offset':
                                try:
                                    # 提取比例：offset80_offset_x10_y10_angle30 -> 80
                                    parts = crop_dir.split('offset')
                                    if len(parts) > 1:
                                        ratio_part = parts[1].split('_offset_x')[0]
                                        ratio_str = ratio_part.rstrip('_')
                                        current_crop_ratio = int(ratio_str) / 100.0
                                        current_crop_type = 'offset'
                                        
                                        # 提取偏移百分比和角度
                                        offset_part = crop_dir.split('_offset_x')[1]
                                        if '_y' in offset_part and '_angle' in offset_part:
                                            offset_x_str = offset_part.split('_y')[0]
                                            y_angle_part = offset_part.split('_y')[1]
                                            if '_angle' in y_angle_part:
                                                offset_y_str = y_angle_part.split('_angle')[0]
                                                angle_str = y_angle_part.split('_angle')[1]
                                                try:
                                                    offset_x = int(offset_x_str)
                                                    offset_y = int(offset_y_str)
                                                    angle = int(angle_str)
                                                    print(f"提取到的角度: {angle}, 当前旋转角度: {rotation_angle}")
                                                    
                                                    # 只有当没有指定旋转角度，或者指定的旋转角度与当前目录匹配时，才添加
                                                    if rotation_angle == 0 or rotation_angle == angle:
                                                        print(f"角度匹配: {angle}")
                                                        # 只有当没有指定裁剪比例，或者指定的裁剪比例与当前目录匹配时，才添加
                                                        if crop_ratio == 0.0 or abs(current_crop_ratio - crop_ratio) < 0.01:
                                                            print(f"添加 offset 类型，比例: {current_crop_ratio}")
                                                            if current_crop_type not in crop_types:
                                                                crop_types.append(current_crop_type)
                                                            if current_crop_ratio not in crop_ratios:
                                                                crop_ratios.append(current_crop_ratio)
                                                            if offset_x not in offset_x_percents:
                                                                offset_x_percents.append(offset_x)
                                                            if offset_y not in offset_y_percents:
                                                                offset_y_percents.append(offset_y)
                                                            if angle not in rotation_angles:
                                                                rotation_angles.append(angle)
                                                except Exception as e:
                                                    print(f"处理 offset 裁剪角度参数时出错: {str(e)}")
                                except Exception as e:
                                    print(f"处理 offset 裁剪目录 {crop_dir} 时出错: {str(e)}")
    elif type == 'rotation':
        # 扫描旋转角度
        rotation_dir = os.path.join(AUGMENTED_DIR, 'rotation')
        if os.path.exists(rotation_dir):
            for angle_dir in os.listdir(rotation_dir):
                if angle_dir.startswith('angle_'):
                    angle = angle_dir.split('angle_')[1]
                    try:
                        rotation_angles.append(int(angle))
                    except:
                        pass
    else:
        # 默认情况，扫描所有类型的参数
        # 扫描旋转角度
        rotation_dir = os.path.join(AUGMENTED_DIR, 'rotation')
        if os.path.exists(rotation_dir):
            for angle_dir in os.listdir(rotation_dir):
                if angle_dir.startswith('angle_'):
                    angle = angle_dir.split('angle_')[1]
                    try:
                        rotation_angles.append(int(angle))
                    except:
                        pass
        
        # 扫描裁剪参数
        cutting_dir = os.path.join(AUGMENTED_DIR, 'cutting')
        if os.path.exists(cutting_dir):
            for crop_dir in os.listdir(cutting_dir):
                if '_' in crop_dir:
                    # 处理新的目录命名规则
                    if 'offset' in crop_dir:
                        # 处理 offset{ratio}_center 格式
                        if '_center' in crop_dir:
                            try:
                                # 提取比例：offset80_center -> 80
                                parts = crop_dir.split('offset')
                                if len(parts) > 1:
                                    ratio_part = parts[1].split('_center')[0]
                                    ratio_str = ratio_part.rstrip('_')
                                    current_crop_ratio = int(ratio_str) / 100.0
                                    current_crop_type = 'center'
                                    if current_crop_type not in crop_types:
                                        crop_types.append(current_crop_type)
                                    if current_crop_ratio not in crop_ratios:
                                        crop_ratios.append(current_crop_ratio)
                            except:
                                pass
                        # 处理 offset{ratio}_offset_x{offset_x}_y{offset_y} 格式
                        elif '_offset_x' in crop_dir:
                            try:
                                # 提取比例：offset80_offset_x10_y10 -> 80
                                parts = crop_dir.split('offset')
                                if len(parts) > 1:
                                    ratio_part = parts[1].split('_offset_x')[0]
                                    ratio_str = ratio_part.rstrip('_')
                                    current_crop_ratio = int(ratio_str) / 100.0
                                    current_crop_type = 'offset'
                                    
                                    # 提取偏移百分比
                                    offset_part = crop_dir.split('_offset_x')[1]
                                    if '_y' in offset_part:
                                        offset_x_str = offset_part.split('_y')[0]
                                        offset_y_str = offset_part.split('_y')[1]
                                        try:
                                            offset_x = int(offset_x_str)
                                            offset_y = int(offset_y_str)
                                            if current_crop_type not in crop_types:
                                                crop_types.append(current_crop_type)
                                            if current_crop_ratio not in crop_ratios:
                                                crop_ratios.append(current_crop_ratio)
                                            if offset_x not in offset_x_percents:
                                                offset_x_percents.append(offset_x)
                                            if offset_y not in offset_y_percents:
                                                offset_y_percents.append(offset_y)
                                        except:
                                            pass
                            except:
                                pass
                    # 保持对旧格式的兼容
                    else:
                        parts = crop_dir.split('_')
                        if len(parts) >= 2:
                            current_crop_type = parts[0]
                            try:
                                current_crop_ratio = int(parts[1]) / 100.0
                                if current_crop_type not in crop_types:
                                    crop_types.append(current_crop_type)
                                if current_crop_ratio not in crop_ratios:
                                    crop_ratios.append(current_crop_ratio)
                            except:
                                pass
        
        # 扫描旋转和裁剪组合参数
        rotation_cutting_dir = os.path.join(AUGMENTED_DIR, 'rotation_cutting')
        if os.path.exists(rotation_cutting_dir):
            for crop_dir in os.listdir(rotation_cutting_dir):
                if '_' in crop_dir:
                    # 处理新的目录命名规则
                    if 'offset' in crop_dir:
                        # 处理 offset{ratio}_center_angle{angle} 格式
                        if '_center_angle' in crop_dir:
                            try:
                                # 提取比例：offset80_center_angle30 -> 80
                                parts = crop_dir.split('offset')
                                if len(parts) > 1:
                                    ratio_part = parts[1].split('_center_angle')[0]
                                    ratio_str = ratio_part.rstrip('_')
                                    current_crop_ratio = int(ratio_str) / 100.0
                                    current_crop_type = 'center'
                                    
                                    # 提取角度：offset80_center_angle30 -> 30
                                    angle_part = crop_dir.split('_center_angle')[1]
                                    try:
                                        angle = int(angle_part)
                                        if angle not in rotation_angles:
                                            rotation_angles.append(angle)
                                    except:
                                        pass
                                    
                                    if current_crop_type not in crop_types:
                                        crop_types.append(current_crop_type)
                                    if current_crop_ratio not in crop_ratios:
                                        crop_ratios.append(current_crop_ratio)
                            except:
                                pass
                        # 处理 offset{ratio}_offset_x{offset_x}_y{offset_y}_angle{angle} 格式
                        elif '_offset_x' in crop_dir and '_angle' in crop_dir:
                            try:
                                # 提取比例：offset80_offset_x10_y10_angle30 -> 80
                                parts = crop_dir.split('offset')
                                if len(parts) > 1:
                                    ratio_part = parts[1].split('_offset_x')[0]
                                    ratio_str = ratio_part.rstrip('_')
                                    current_crop_ratio = int(ratio_str) / 100.0
                                    current_crop_type = 'offset'
                                    
                                    # 提取偏移百分比和角度
                                    offset_part = crop_dir.split('_offset_x')[1]
                                    if '_y' in offset_part and '_angle' in offset_part:
                                        offset_x_str = offset_part.split('_y')[0]
                                        y_angle_part = offset_part.split('_y')[1]
                                        if '_angle' in y_angle_part:
                                            offset_y_str = y_angle_part.split('_angle')[0]
                                            angle_str = y_angle_part.split('_angle')[1]
                                            try:
                                                offset_x = int(offset_x_str)
                                                offset_y = int(offset_y_str)
                                                angle = int(angle_str)
                                                
                                                if current_crop_type not in crop_types:
                                                    crop_types.append(current_crop_type)
                                                if current_crop_ratio not in crop_ratios:
                                                    crop_ratios.append(current_crop_ratio)
                                                if offset_x not in offset_x_percents:
                                                    offset_x_percents.append(offset_x)
                                                if offset_y not in offset_y_percents:
                                                    offset_y_percents.append(offset_y)
                                                if angle not in rotation_angles:
                                                    rotation_angles.append(angle)
                                            except:
                                                pass
                            except:
                                pass
    
    return jsonify({
        'rotation_angles': sorted(rotation_angles),
        'crop_types': sorted(crop_types),
        'crop_ratios': sorted(crop_ratios),
        'offset_x_percents': sorted(offset_x_percents),
        'offset_y_percents': sorted(offset_y_percents)
    })


@app.route('/api/evaluation/results')
def get_evaluation_results():
    """
    获取评估结果
    支持分页和过滤
    """
    # 获取查询参数
    model = request.args.get('model', 'resnet50')
    augmentation_type = request.args.get('augmentation_type', '')
    rotation_angle = request.args.get('rotation_angle', '0', type=int)
    crop_type = request.args.get('crop_type', '')
    crop_ratio = request.args.get('crop_ratio', '0.0', type=float)
    page = request.args.get('page', '1', type=int)
    page_size = request.args.get('page_size', '10', type=int)
    
    # 打印原始参数值和类型
    print(f"原始参数值:")
    print(f"rotation_angle: {rotation_angle}, 类型: {type(rotation_angle)}")
    print(f"crop_ratio: {crop_ratio}, 类型: {type(crop_ratio)}")
    
    # 确保 rotation_angle 是整数类型
    try:
        rotation_angle = int(rotation_angle)
        print(f"转换后 rotation_angle: {rotation_angle}, 类型: {type(rotation_angle)}")
    except Exception as e:
        print(f"转换 rotation_angle 失败: {str(e)}")
        rotation_angle = 0
    
    # 确保 crop_ratio 是浮点数类型
    try:
        crop_ratio = float(crop_ratio)
        print(f"转换后 crop_ratio: {crop_ratio}, 类型: {type(crop_ratio)}")
    except Exception as e:
        print(f"转换 crop_ratio 失败: {str(e)}")
        crop_ratio = 0.0
    
    # 构建评估结果文件名
    filename = f"evaluation_{model}"
    print(f"初始 filename: {filename}")
    
    if augmentation_type:
        filename += f"_{augmentation_type}"
        print(f"添加 augmentation_type 后 filename: {filename}")
    
    if rotation_angle > 0:
        filename += f"_angle{rotation_angle}"
        print(f"添加 rotation_angle 后 filename: {filename}")
    
    if crop_type:
        if crop_type == 'center':
            filename += f"_offset{int(crop_ratio * 100)}_center"
        elif crop_type == 'offset':
            # 从请求中获取偏移百分比参数
            offset_x_percent = request.args.get('offset_x_percent', 0, type=int)
            offset_y_percent = request.args.get('offset_y_percent', 0, type=int)
            filename += f"_offset{int(crop_ratio * 100)}_x{offset_x_percent}_y{offset_y_percent}"
        else:
            filename += f"_{crop_type}_{int(crop_ratio * 100)}"
        print(f"添加 crop_type 后 filename: {filename}")
    
    filename += ".json"
    print(f"最终 filename: {filename}")
    
    # 打印文件名信息
    print(f"构建的评估结果文件名: {filename}")
    
    # 检查文件是否存在
    result_path = os.path.join(EVALUATION_DIR, filename)
    print(f"评估结果文件路径: {result_path}")
    print(f"检查文件是否存在: {os.path.exists(result_path)}")
    
    if not os.path.exists(result_path):
        print(f"评估结果文件不存在: {result_path}")
        # 尝试查找目录中的所有文件
        import glob
        files = glob.glob(os.path.join(EVALUATION_DIR, "*.json"))
        print(f"目录中的JSON文件: {files}")
        
        # 尝试构建可能的文件名变体
        possible_filenames = []
        # 变体1: 原始格式
        possible_filenames.append(filename)
        # 变体2: 检查是否有类似的文件名
        for file in files:
            file_basename = os.path.basename(file)
            if 'cutting' in file_basename and 'offset' in file_basename:
                possible_filenames.append(file_basename)
        
        print(f"可能的文件名变体: {possible_filenames}")
        
        return jsonify({
            'error': 'Evaluation results not found',
            'message': f'Please run validate_augmented.py first for model {model} with specified parameters',
            'available_files': files,
            'expected_filename': filename,
            'possible_filenames': possible_filenames
        }), 404
    
    # 读取评估结果文件
    
    try:
        print(f"尝试读取评估结果文件: {result_path}")
        
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取评估结果文件，包含 {len(data.get('results', []))} 个结果")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {str(e)}")
        return jsonify({
            'error': 'Failed to parse evaluation results',
            'message': str(e)
        }), 500
    except Exception as e:
        print(f"读取评估结果文件时出错: {str(e)}")
        return jsonify({
            'error': 'Failed to read evaluation results',
            'message': str(e)
        }), 500
    
    # 计算分页
    total_results = len(data.get('results', []))
    total_pages = (total_results + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    paginated_results = data.get('results', [])[start:end]
    
    # 转换路径为相对路径，方便前端访问
    for result in paginated_results:
        # 转换增强图片路径
        if 'augmented_path' in result:
            # 确保路径是字符串
            augmented_path_str = str(result['augmented_path'])
            
            # 直接从路径中提取相对路径
            # 例如：从 "D:\Project\SearchPic\Test2\querySet_step2_profile\rotation\angle_45\animal_model\P12345.jpg"
            # 提取出 "rotation\angle_45\animal_model\P12345.jpg"
            
            # 1. 首先尝试使用split方法提取相对路径
            try:
                # 分割路径，找到querySet_step2_profile后的部分
                if 'querySet_step2_profile' in augmented_path_str:
                    parts = augmented_path_str.split('querySet_step2_profile', 1)
                    if len(parts) > 1:
                        augmented_rel_path = parts[1]
                        # 移除前导斜杠
                        augmented_rel_path = augmented_rel_path.lstrip('\\/')
                        # 确保路径使用正斜杠
                        augmented_rel_path = augmented_rel_path.replace('\\', '/')
                        result['augmented_path'] = f'/api/images/augmented/{augmented_rel_path}'
                    else:
                        # 如果分割失败，使用文件名
                        import ntpath
                        filename = ntpath.basename(augmented_path_str)
                        result['augmented_path'] = f'/api/images/augmented/{filename}'
                else:
                    # 如果路径中没有querySet_step2_profile，使用文件名
                    import ntpath
                    filename = ntpath.basename(augmented_path_str)
                    result['augmented_path'] = f'/api/images/augmented/{filename}'
            except:
                # 如果所有方法都失败，使用文件名
                import ntpath
                filename = ntpath.basename(augmented_path_str)
                result['augmented_path'] = f'/api/images/augmented/{filename}'
            
            # 添加前端期望的字段
            result['query_image'] = result['augmented_path']
            result['query_image_url'] = result['augmented_path']
        
        # 转换原始图片路径
        if 'original_path' in result:
            # 获取相对路径
            original_rel_path = os.path.relpath(result['original_path'], ORIGINAL_DIR)
            result['original_path'] = f'/api/images/original/{original_rel_path}'
            # 添加前端期望的字段
            result['original_image'] = result['original_path']
        
        # 转换top_k_results为前端期望的格式
        if 'top_k_results' in result:
            result['top_k'] = []
            for item in result['top_k_results']:
                if 'image_path' in item:
                    # 确保路径是字符串
                    image_path_str = str(item['image_path'])
                    
                    # 构建正确的图片URL
                    # 结果图片存储在D:\Project\SearchPic\Test2\img文件夹中
                    try:
                        # 1. 检查路径是否已经包含img目录
                        if 'img' in image_path_str:
                            # 从img目录开始提取相对路径
                            parts = image_path_str.split('img', 1)
                            if len(parts) > 1:
                                img_rel_path = parts[1]
                                # 移除前导斜杠
                                img_rel_path = img_rel_path.lstrip('\\/')
                                # 确保路径使用正斜杠
                                img_rel_path = img_rel_path.replace('\\', '/')
                                image_url = f'/api/images/original/{img_rel_path}'
                            else:
                                # 作为后备，提取文件名
                                import ntpath
                                filename = ntpath.basename(image_path_str)
                                image_url = f'/api/images/original/{filename}'
                        else:
                            # 2. 检查是否是绝对路径
                            if os.path.isabs(image_path_str):
                                # 计算相对于IMG_DIR的路径
                                img_rel_path = os.path.relpath(image_path_str, IMG_DIR)
                                # 确保路径使用正斜杠
                                img_rel_path = img_rel_path.replace('\\', '/')
                                image_url = f'/api/images/original/{img_rel_path}'
                            else:
                                # 3. 直接使用路径
                                # 确保路径使用正斜杠
                                img_rel_path = image_path_str.replace('\\', '/')
                                image_url = f'/api/images/original/{img_rel_path}'
                    except Exception as e:
                        print(f"处理Top结果图片路径时出错: {str(e)}")
                        # 作为最后手段，使用文件名
                        import ntpath
                        filename = ntpath.basename(image_path_str)
                        image_url = f'/api/images/original/{filename}'
                    
                    result['top_k'].append({
                        'id': item.get('id', ''),
                        'distance': item.get('distance', 0),
                        'path': item['image_path'],
                        'url': image_url
                    })
            # 移除原始字段
            del result['top_k_results']
    
    return jsonify({
        'model': data.get('model', model),
        'augmentation_params': data.get('augmentation_params', {}),
        'metrics': data.get('metrics', {}),
        'results': paginated_results,
        'pagination': {
            'total_results': total_results,
            'total_pages': total_pages,
            'current_page': page,
            'page_size': page_size
        }
    })


@app.route('/api/images/augmented/<path:path>')
def get_augmented_image(path):
    """
    获取增强图片
    """
    image_path = os.path.join(AUGMENTED_DIR, path)
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    # 获取图片所在目录
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    
    return send_from_directory(directory, filename)


@app.route('/api/images/original/<path:path>')
def get_original_image(path):
    """
    获取原始图片
    """
    # 尝试从多个目录加载图片
    # 1. 首先尝试 ORIGINAL_DIR
    image_path = os.path.join(ORIGINAL_DIR, path)
    print(f"尝试从 ORIGINAL_DIR 读取原始图片: {image_path}")
    if os.path.exists(image_path):
        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        return send_from_directory(directory, filename)
    
    # 2. 然后尝试 IMG_DIR
    image_path = os.path.join(IMG_DIR, path)
    print(f"尝试从 IMG_DIR 读取原始图片: {image_path}")
    if os.path.exists(image_path):
        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        return send_from_directory(directory, filename)
    
    # 3. 最后尝试直接使用路径（处理可能的绝对路径）
    if os.path.exists(path):
        print(f"尝试直接读取原始图片: {path}")
        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        return send_from_directory(directory, filename)
    
    print(f"图片未找到: {path}")
    return jsonify({'error': 'Image not found'}), 404


@app.route('/api/run-evaluation', methods=['POST'])
def run_evaluation():
    """
    运行评估
    """
    # 获取请求数据
    data = request.json
    model = data.get('model', 'resnet50')
    augmentation_type = data.get('augmentation_type', '')
    rotation_angle = data.get('rotation_angle', 0)
    crop_type = data.get('crop_type', '')
    crop_ratio = data.get('crop_ratio', 0.0)
    offset_x_percent = data.get('offset_x_percent', 0)
    offset_y_percent = data.get('offset_y_percent', 0)
    device = data.get('device', 'auto')
    
    # 构建命令
    import subprocess
    import shlex
    
    cmd = [
        'python', 'validate_augmented.py',
        '--model', model,
        '--device', device
    ]
    
    if augmentation_type:
        cmd.extend(['--augmentation_type', augmentation_type])
    if rotation_angle > 0:
        cmd.extend(['--rotation_angle', str(rotation_angle)])
    if crop_type:
        cmd.extend(['--crop_type', crop_type])
    if crop_ratio > 0:
        cmd.extend(['--crop_ratio', str(crop_ratio)])
    if crop_type == 'offset':
        cmd.extend(['--offset_x_percent', str(offset_x_percent)])
        cmd.extend(['--offset_y_percent', str(offset_y_percent)])
    
    try:
        # 运行评估命令
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        # 检查命令是否成功执行
        if result.returncode != 0:
            return jsonify({
                'error': 'Evaluation failed',
                'stdout': result.stdout,
                'stderr': result.stderr
            }), 500
        
        # 构建评估结果文件名
        filename = f"evaluation_{model}"
        if augmentation_type:
            filename += f"_{augmentation_type}"
        if rotation_angle > 0:
            filename += f"_angle{rotation_angle}"
        if crop_type:
            if crop_type == 'center':
                filename += f"_offset{int(crop_ratio * 100)}_center"
            elif crop_type == 'offset':
                filename += f"_offset{int(crop_ratio * 100)}_x{offset_x_percent}_y{offset_y_percent}"
            else:
                filename += f"_{crop_type}_{int(crop_ratio * 100)}"
        filename += ".json"
        
        # 检查评估结果文件是否生成
        result_path = os.path.join(EVALUATION_DIR, filename)
        if not os.path.exists(result_path):
            return jsonify({
                'error': 'Evaluation results not generated',
                'stdout': result.stdout
            }), 500
        
        return jsonify({
            'success': True,
            'stdout': result.stdout,
            'result_file': filename
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({
            'error': 'Evaluation timed out after 1 hour'
        }), 500
    except Exception as e:
        return jsonify({
            'error': 'Failed to run evaluation',
            'message': str(e)
        }), 500


@app.route('/static/<path:path>')
def serve_static(path):
    """
    静态文件路由
    """
    return send_from_directory('static', path)


if __name__ == '__main__':
    """
    主函数
    """
    # 确保 static 目录存在
    Path('static').mkdir(parents=True, exist_ok=True)
    Path('static/css').mkdir(parents=True, exist_ok=True)
    Path('static/js').mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("启动后端服务器")
    print("=" * 60)
    print("服务器地址: http://localhost:5000")
    print("API 文档: http://localhost:5000/api/models")
    print("=" * 60)
    
    # 启动服务器
    app.run(host='0.0.0.0', port=5000, debug=True)