#!/usr/bin/env python3
"""
图片增强脚本
对queryset_step2中的图片进行旋转和裁剪，并生成对应的json标注文件
"""

import os
import json
import argparse
from PIL import Image
import numpy as np


def parse_args():
    """
    解析命令行参数
    python augment_queryset.py --crop_ratios 0.8 --crop_types offset --offset_x_percent 10 --offset_y_percent 10
    """
    parser = argparse.ArgumentParser(description='增强queryset图片并生成标注')
    parser.add_argument('--input_dir', type=str, default='../../querySet_step2',
                        help='输入图片目录')
    parser.add_argument('--output_dir', type=str, default=r"D:\Project\SearchPic\Test2\querySet_step2_profile",
                        help='输出目录')
    parser.add_argument('--rotation_angles', type=int, nargs='+',
                        help='旋转角度列表，例如：90 180 270')
    parser.add_argument('--crop_ratios', type=float, nargs='+',
                        help='裁剪比例列表，例如：0.8 0.6')
    parser.add_argument('--crop_types', type=str, nargs='+',
                        help='裁剪类型列表（center, offset），例如：center offset')
    parser.add_argument('--offset_x_percent', type=int, default=0,
                        help='X轴偏移百分比，例如：5 或 -10')
    parser.add_argument('--offset_y_percent', type=int, default=0,
                        help='Y轴偏移百分比，例如：5 或 -10')
    return parser.parse_args()


def ensure_dir(directory):
    """
    确保目录存在
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def rotate_image(image, angle):
    """
    旋转图片
    """
    return image.rotate(angle, expand=False, fillcolor='white')


def crop_center(image, crop_ratio):
    """
    中心裁剪
    """
    width, height = image.size
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image.crop((left, top, right, bottom))


def crop_offset(image, crop_ratio, offset_x_percent=0, offset_y_percent=0):
    """
    偏移裁剪
    
    :param image: 输入图像 (PIL.Image)
    :param crop_ratio: 裁剪比例 (保留多少主体区域，例如 0.85 表示保留 85% 的区域)
    :param offset_x_percent: X轴偏移百分比 (如 5% 或 -10%，表示相对于图像宽度的偏移)
    :param offset_y_percent: Y轴偏移百分比 (如 5% 或 -10%，表示相对于图像高度的偏移)
    :return: 裁剪后的图像 (PIL.Image)
    """
    width, height = image.size
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)
    
    # 计算偏移量 (基于百分比)
    offset_x = int(width * offset_x_percent / 100)
    offset_y = int(height * offset_y_percent / 100)
    
    # 确保裁剪区域在图像范围内
    left = max(0, (width - new_width) // 2 + offset_x)
    top = max(0, (height - new_height) // 2 + offset_y)
    right = min(width, left + new_width)
    bottom = min(height, top + new_height)
    
    # 调整以确保裁剪尺寸正确
    if right - left != new_width:
        left = (width - new_width) // 2
        right = left + new_width
    if bottom - top != new_height:
        top = (height - new_height) // 2
        bottom = top + new_height
    
    return image.crop((left, top, right, bottom))


def generate_variant_id(original_name, augmentation_type, params):
    """
    生成变体ID
    """
    if augmentation_type == 'rotation':
        return f"{original_name}_rot_{params['angle']}"
    elif augmentation_type == 'crop':
        return f"{original_name}_crop_{int(params['ratio']*100)}_{params['type']}"
    return original_name


def process_image(image_path, output_dir, input_dir, rotation_angles, crop_ratios, crop_types, offset_x_percent, offset_y_percent):
    """
    处理单个图片
    
    返回值:
        dict: 按增强类型和参数分组的变体
    """
    try:
        # 打开图片
        image = Image.open(image_path)
        
        # 获取文件名信息
        original_name = os.path.splitext(os.path.basename(image_path))[0]
        original_id = original_name
        
        # 获取相对路径（用于保持目录结构）
        relative_path = os.path.relpath(image_path, input_dir)
        category = os.path.dirname(relative_path)
        
        # 创建输出目录
        ensure_dir(output_dir)
        
        # 按增强类型和参数分组存储变体
        grouped_variants = {}
        
        # 处理旋转和裁剪
        if rotation_angles and crop_ratios and crop_types:
            # 旋转和裁剪参数都指定了，从rotation目录找旋转图片
            for angle in rotation_angles:
                # 构建旋转图片路径
                rotation_dir = os.path.join(output_dir, 'rotation', f"angle_{angle}", category)
                rotation_filename = f"{original_name}_rot_{angle}.jpg"
                rotation_path = os.path.join(rotation_dir, rotation_filename)
                
                # 检查旋转图片是否存在
                if not os.path.exists(rotation_path):
                    print(f"错误: 旋转图片不存在: {rotation_path}")
                    continue
                
                try:
                    # 打开旋转图片
                    rotated_image = Image.open(rotation_path)
                    
                    # 对旋转图片进行裁剪
                    for ratio in crop_ratios:
                        for crop_type in crop_types:
                            # 创建裁剪目录结构: rotation_cutting/{crop_subdir}/{category}
                            if crop_type == 'center':
                                crop_subdir = f"offset{int(ratio*100)}_center_angle{angle}"
                            elif crop_type == 'offset':
                                crop_subdir = f"offset{int(ratio*100)}_offset_x{offset_x_percent}_y{offset_y_percent}_angle{angle}"
                            else:
                                crop_subdir = f"{crop_type}_{int(ratio*100)}_angle{angle}"
                            crop_dir = os.path.join(output_dir, 'rotation_cutting', crop_subdir, category)
                            ensure_dir(crop_dir)
                            
                            # 裁剪旋转后的图片
                            if crop_type == 'center':
                                cropped_image = crop_center(rotated_image, ratio)
                            elif crop_type == 'offset':
                                cropped_image = crop_offset(rotated_image, ratio, offset_x_percent, offset_y_percent)
                            else:
                                continue
                            
                            # 保存裁剪后的图片
                            crop_filename = f"{original_name}_rot_{angle}_crop_{int(ratio*100)}_{crop_type}.jpg"
                            crop_path = os.path.join(crop_dir, crop_filename)
                            cropped_image.save(crop_path)
                            
                            # 构建裁剪组的键
                            crop_key = f"rotation_cutting_offset{int(ratio*100)}_{crop_type}_x{offset_x_percent}_y{offset_y_percent}_angle{angle}"
                            if crop_key not in grouped_variants:
                                grouped_variants[crop_key] = {
                                    "type": "rotation_cutting",
                                    "crop_type": crop_type,
                                    "crop_ratio": ratio,
                                    "offset_x_percent": offset_x_percent,
                                    "offset_y_percent": offset_y_percent,
                                    "angle": angle,
                                    "dir": os.path.join(output_dir, 'rotation_cutting', crop_subdir),
                                    "variants": []
                                }
                            
                            # 添加裁剪变体到列表
                            crop_variant_id = generate_variant_id(f"{original_name}_rot_{angle}", 'crop', {'ratio': ratio, 'type': crop_type})
                            crop_variant_path = crop_path.replace(output_dir, '').replace('\\', '/')
                            grouped_variants[crop_key]["variants"].append({
                                "original_id": original_id,
                                "original_path": image_path.replace('\\', '/'),
                                "category": category,
                                "variant_id": crop_variant_id,
                                "variant_path": crop_variant_path,
                                "augmentation_type": "center_crop" if crop_type == 'center' else "offset_crop",
                                "parameters": {
                                    "crop_ratio": ratio,
                                    "position": crop_type,
                                    "rotation_angle": angle,
                                    "offset_x_percent": offset_x_percent,
                                    "offset_y_percent": offset_y_percent
                                },
                                "ground_truth": original_id
                            })
                except Exception as e:
                    print(f"处理旋转图片 {rotation_path} 时出错: {e}")
                    continue
        elif rotation_angles:
            # 只旋转，不裁剪
            for angle in rotation_angles:
                # 创建旋转目录结构: rotation/angle_{angle}/{category}
                rotation_dir = os.path.join(output_dir, 'rotation', f"angle_{angle}", category)
                ensure_dir(rotation_dir)
                
                # 旋转图片
                rotated_image = rotate_image(image, angle)
                
                # 保存旋转后的图片
                rotation_filename = f"{original_name}_rot_{angle}.jpg"
                rotation_path = os.path.join(rotation_dir, rotation_filename)
                rotated_image.save(rotation_path)
                
                # 构建旋转组的键
                rotation_key = f"rotation_angle_{angle}"
                if rotation_key not in grouped_variants:
                    grouped_variants[rotation_key] = {
                        "type": "rotation",
                        "angle": angle,
                        "dir": os.path.join(output_dir, 'rotation', f"angle_{angle}"),
                        "variants": []
                    }
                
                # 添加旋转变体到列表
                rotation_variant_id = generate_variant_id(original_name, 'rotation', {'angle': angle})
                rotation_variant_path = rotation_path.replace(output_dir, '').replace('\\', '/')
                grouped_variants[rotation_key]["variants"].append({
                    "original_id": original_id,
                    "original_path": image_path.replace('\\', '/'),
                    "category": category,
                    "variant_id": rotation_variant_id,
                    "variant_path": rotation_variant_path,
                    "augmentation_type": "rotation",
                    "parameters": {
                        "angle": angle
                    },
                    "ground_truth": original_id
                })
        elif crop_ratios and crop_types:
            # 只裁剪，不旋转
            for ratio in crop_ratios:
                for crop_type in crop_types:
                    # 创建裁剪目录结构: cutting/{crop_type}_{ratio}/{category}
                    if crop_type == 'center':
                        crop_subdir = f"offset{int(ratio*100)}_center"
                    elif crop_type == 'offset':
                        crop_subdir = f"offset{int(ratio*100)}_offset_x{offset_x_percent}_y{offset_y_percent}"
                    else:
                        crop_subdir = f"{crop_type}_{int(ratio*100)}"
                    crop_dir = os.path.join(output_dir, 'cutting', crop_subdir, category)
                    ensure_dir(crop_dir)
                    
                    # 裁剪原始图片
                    if crop_type == 'center':
                        cropped_image = crop_center(image, ratio)
                    elif crop_type == 'offset':
                        cropped_image = crop_offset(image, ratio, offset_x_percent, offset_y_percent)
                    else:
                        continue
                    
                    # 保存裁剪后的图片
                    crop_filename = f"{original_name}_crop_{int(ratio*100)}_{crop_type}.jpg"
                    crop_path = os.path.join(crop_dir, crop_filename)
                    cropped_image.save(crop_path)
                    
                    # 构建裁剪组的键
                    if crop_type == 'center':
                        crop_key = f"cutting_offset{int(ratio*100)}_center"
                        crop_subdir = f"offset{int(ratio*100)}_center"
                    elif crop_type == 'offset':
                        crop_key = f"cutting_offset{int(ratio*100)}_offset_x{offset_x_percent}_y{offset_y_percent}"
                        crop_subdir = f"offset{int(ratio*100)}_offset_x{offset_x_percent}_y{offset_y_percent}"
                    else:
                        crop_key = f"cutting_{crop_type}_{int(ratio*100)}"
                        crop_subdir = f"{crop_type}_{int(ratio*100)}"
                    if crop_key not in grouped_variants:
                        grouped_variants[crop_key] = {
                            "type": "cutting",
                            "crop_type": crop_type,
                            "crop_ratio": ratio,
                            "offset_x_percent": offset_x_percent,
                            "offset_y_percent": offset_y_percent,
                            "dir": os.path.join(output_dir, 'cutting', crop_subdir),
                            "variants": []
                        }
                    
                    # 添加裁剪变体到列表
                    crop_variant_id = generate_variant_id(original_name, 'crop', {'ratio': ratio, 'type': crop_type})
                    crop_variant_path = crop_path.replace(output_dir, '').replace('\\', '/')
                    grouped_variants[crop_key]["variants"].append({
                        "original_id": original_id,
                        "original_path": image_path.replace('\\', '/'),
                        "category": category,
                        "variant_id": crop_variant_id,
                        "variant_path": crop_variant_path,
                        "augmentation_type": "center_crop" if crop_type == 'center' else "offset_crop",
                        "parameters": {
                            "crop_ratio": ratio,
                            "position": crop_type,
                            "offset_x_percent": offset_x_percent,
                            "offset_y_percent": offset_y_percent
                        },
                        "ground_truth": original_id
                    })
    
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return {}
    
    return grouped_variants


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 验证参数
    if not args.rotation_angles and not (args.crop_ratios and args.crop_types):
        print("错误：必须指定旋转角度或裁剪参数")
        return
    
    # 确保输入目录存在
    if not os.path.exists(args.input_dir):
        print(f"输入目录不存在: {args.input_dir}")
        return
    
    # 确保输出目录存在
    ensure_dir(args.output_dir)
    
    # 收集所有图片
    image_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 按增强类型和参数分组存储所有变体
    all_grouped_variants = {}
    total_variants = 0
    
    # 处理所有图片
    for i, image_path in enumerate(image_files, 1):
        print(f"处理图片 {i}/{len(image_files)}: {os.path.basename(image_path)}")
        result = process_image(
            image_path,
            args.output_dir,
            args.input_dir,
            args.rotation_angles,
            args.crop_ratios,
            args.crop_types,
            args.offset_x_percent,
            args.offset_y_percent
        )
        if result:
            # 合并分组变体
            for key, group in result.items():
                if key not in all_grouped_variants:
                    all_grouped_variants[key] = group.copy()
                else:
                    all_grouped_variants[key]["variants"].extend(group["variants"])
                total_variants += len(group["variants"])
    
    # 为每个分组生成JSON文件
    for key, group in all_grouped_variants.items():
        # 生成JSON数据
        total_queries = len(group["variants"])
        json_data = {
            "version": "1.0",
            "test_set": "phase2_augmentation",
            "total_queries": total_queries,
            "augmentation_type": group["type"],
            "parameters": {}
        }
        
        # 添加参数
        if group["type"] == "rotation":
            json_data["parameters"]["angle"] = group["angle"]
        elif group["type"] == "cutting":
            json_data["parameters"]["crop_type"] = group["crop_type"]
            json_data["parameters"]["crop_ratio"] = group["crop_ratio"]
            if "offset_x_percent" in group:
                json_data["parameters"]["offset_x_percent"] = group["offset_x_percent"]
            if "offset_y_percent" in group:
                json_data["parameters"]["offset_y_percent"] = group["offset_y_percent"]
        
        # 添加变体
        json_data["variants"] = group["variants"]
        
        # 生成JSON文件名
        if group["type"] == "rotation":
            json_filename = f"augmentation_annotations_rotation_angle_{group['angle']}.json"
        elif group["type"] == "cutting":
            if group["crop_type"] == 'center':
                json_filename = f"augmentation_annotations_cutting_offset{int(group['crop_ratio']*100)}_center.json"
            elif group["crop_type"] == 'offset':
                json_filename = f"augmentation_annotations_cutting_offset{int(group['crop_ratio']*100)}_offset_x{group.get('offset_x_percent', 0)}_y{group.get('offset_y_percent', 0)}.json"
            else:
                json_filename = f"augmentation_annotations_cutting_{group['crop_type']}_{int(group['crop_ratio']*100)}.json"
        elif group["type"] == "rotation_cutting":
            if group["crop_type"] == 'center':
                json_filename = f"augmentation_annotations_rotation_cutting_offset{int(group['crop_ratio']*100)}_center_angle{group['angle']}.json"
            elif group["crop_type"] == 'offset':
                json_filename = f"augmentation_annotations_rotation_cutting_offset{int(group['crop_ratio']*100)}_offset_x{group.get('offset_x_percent', 0)}_y{group.get('offset_y_percent', 0)}_angle{group['angle']}.json"
            else:
                json_filename = f"augmentation_annotations_rotation_cutting_{group['crop_type']}_{int(group['crop_ratio']*100)}_angle{group['angle']}.json"
        
        # 保存JSON文件到对应的目录
        json_path = os.path.join(group["dir"], json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"标注文件保存到: {json_path}")
    
    print(f"\n增强完成！")
    print(f"处理了 {len(image_files)} 张原图")
    print(f"生成了 {total_variants} 个变体")
    print(f"生成了 {len(all_grouped_variants)} 个标注文件")


if __name__ == "__main__":
    main()
