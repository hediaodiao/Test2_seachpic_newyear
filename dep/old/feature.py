#!/usr/bin/env python3
"""
图像特征提取与相似度分析 - 优化版
支持多种模型选择，本地缓存管理
"""

import os
import sys
import ssl
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========== 1. SSL证书修复 ==========
ssl._create_default_https_context = ssl._create_unverified_context

# ========== 2. 本地缓存管理器 ==========
class ModelCacheManager:
    """模型缓存管理器"""
    
    def __init__(self, cache_dir="./model_cache"):
        """
        初始化缓存管理器
        
        参数:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 设置PyTorch缓存环境变量
        os.environ['TORCH_HOME'] = str(self.cache_dir)
        
        # 模型文件映射
        self.model_files = {
            'resnet50': 'resnet50-11ad3fa6.pth',
            'efficientnet_lite0': 'efficientnet_lite0-0aa5c2b1.pth',
            'mobilenet_v3_small': 'mobilenet_v3_small-047dcff4.pth',
            'convnext_tiny': 'convnext_tiny-983f1584.pth',
            'openclip_vit_b_32': 'open_clip_model.safetensors',
            'openclip_vit_l_14': 'open_clip_model_vit_l_14.safetensors',
            'dinov2_vit_s': 'dinov2_vit_small.pth',
        }
        
        # 模型加载函数映射
        self.model_loaders = {
            'resnet50': models.resnet50,
            'efficientnet_lite0': self._load_efficientnet_lite0,
            'mobilenet_v3_small': models.mobilenet_v3_small,
            'convnext_tiny': models.convnext_tiny,
            'openclip_vit_b_32': self._load_openclip_vit_b_32,
            'openclip_vit_l_14': self._load_openclip_vit_l_14,
            'dinov2_vit_s': self._load_dinov2_vit_s,
        }
        
        print(f"📁 模型缓存目录: {self.cache_dir.absolute()}")
    
    def get_model_path(self, model_name):
        """获取模型文件路径"""
        if model_name not in self.model_files:
            raise ValueError(f"不支持的模型: {model_name}")
        return self.cache_dir / self.model_files[model_name]
    
    def is_model_cached(self, model_name):
        """检查模型是否已缓存"""
        model_path = self.get_model_path(model_name)
        return model_path.exists()
    
    def load_model_from_cache(self, model_name, weights='IMAGENET1K_V1'):
        """
        从缓存加载模型，如果没有则下载
        
        参数:
            model_name: 模型名称
            weights: 权重类型
            
        返回:
            model: 加载的模型
        """
        if model_name not in self.model_loaders:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 检查缓存
        if self.is_model_cached(model_name):
            print(f"✅ 从缓存加载: {model_name}")
            try:
                # 尝试从缓存加载
                return self.model_loaders[model_name](weights=weights)
            except:
                # 如果缓存文件损坏，删除并重新下载
                model_path = self.get_model_path(model_name)
                print(f"⚠ 缓存文件损坏，删除: {model_path}")
                model_path.unlink(missing_ok=True)
        
        # 下载模型
        print(f"📥 下载模型: {model_name}")
        try:
            model = self.model_loaders[model_name](weights=weights)
            print(f"✅ 模型下载完成: {model_name}")
            return model
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            raise
    
    def _load_efficientnet_lite0(self, weights=None):
        """
        加载EfficientNet-B0模型（替代EfficientNet-Lite0）
        
        参数:
            weights: 权重参数（不使用）
            
        返回:
            model: EfficientNet-B0模型
        """
        try:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b0')
            return model
        except ImportError:
            print("❌ 需要安装 efficientnet-pytorch 包")
            print("请运行: pip install efficientnet-pytorch")
            raise
    
    def _load_openclip_vit_b_32(self, weights=None):
        """
        加载OpenCLIP ViT-B/32模型
        
        参数:
            weights: 权重参数（不使用）
            
        返回:
            model: OpenCLIP模型
        """
        try:
            import open_clip
            import os
            from pathlib import Path
            
            # 设置离线模式环境变量
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            # 构建本地模型文件路径
            local_model_path = Path(self.cache_dir) / "timm" / "vit_base_patch32_clip_224.laion2b_e16" / "open_clip_model.safetensors"
            
            if local_model_path.exists():
                # 如果本地有模型文件，直接加载
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', 
                    pretrained=str(local_model_path)
                )
                print(f"✅ 从本地文件加载 OpenCLIP 模型: {local_model_path}")
            else:
                # 否则从网络下载（这可能会失败，如果没有网络连接）
                print("⚠️ 本地模型文件不存在，尝试从网络加载...")
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
            
            # 更新实例的预处理方法，因为CLIP有自己的预处理
            self.clip_preprocess = preprocess
            return model
        except ImportError:
            print("❌ 需要安装 open_clip 包")
            print("请运行: pip install open_clip_torch")
            raise
        except Exception as e:
            print(f"❌ OpenCLIP模型加载失败: {e}")
            print("请确保已安装 open_clip_torch 包并确保模型文件存在")
            raise
    
    def _load_openclip_vit_l_14(self, weights=None):
        """
        加载OpenCLIP ViT-L/14模型并转换为INT8精度
        
        参数:
            weights: 权重参数（不使用）
            
        返回:
            model: OpenCLIP模型（INT8精度）
        """
        try:
            import open_clip
            import os
            import torch
            from pathlib import Path
            
            # 构建本地模型文件路径
            # 1. 检查model_cache目录中现有的OpenCLIP模型文件
            openclip_model_paths = [
                Path(self.cache_dir) / "open_clip_pytorch_model.safetensors",
                Path(self.cache_dir) / "open_clip_pytorch_model.bin",
                Path(self.cache_dir) / "model.safetensors",
                Path(self.cache_dir) / "pytorch_model.bin",
                Path(self.cache_dir) / "open_clip_model_vit_l_14.safetensors"
            ]
            
            # 2. 检查用户提供的模型文件路径
            user_model_path = Path(self.cache_dir) / "vit_l_14-laion2b_s32b_b82k.bin"
            
            # 3. 同时检查子目录中的模型文件（兼容旧路径）
            local_model_dir = Path(self.cache_dir) / "timm" / "vit_large_patch14_clip_224.laion2b_e16"
            local_model_path_subdir = local_model_dir / "open_clip_model_vit_l_14.safetensors"
            
            # 确保本地模型目录存在
            local_model_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查模型文件是否存在（按优先级顺序）
            model_path_to_use = None
            # 检查用户提供的路径
            if user_model_path.exists():
                model_path_to_use = user_model_path
                print(f"✅ 找到用户提供的模型文件: {model_path_to_use}")
            # 检查model_cache目录中的OpenCLIP模型文件
            else:
                for path in openclip_model_paths:
                    if path.exists():
                        model_path_to_use = path
                        print(f"✅ 找到本地OpenCLIP模型文件: {model_path_to_use}")
                        break
            # 检查子目录中的模型文件
            if not model_path_to_use and local_model_path_subdir.exists():
                model_path_to_use = local_model_path_subdir
                print(f"✅ 找到模型文件: {model_path_to_use}")
            
            if model_path_to_use:
                # 如果本地有模型文件，直接加载
                try:
                    # 设置离线模式环境变量
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
                    
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        'ViT-L-14', 
                        pretrained=str(model_path_to_use)
                    )
                    print(f"✅ 从本地文件加载 OpenCLIP ViT-L/14 模型: {model_path_to_use}")
                except Exception as e:
                    print(f"⚠️ 从本地文件加载失败: {e}")
                    print("⚠️ 尝试使用默认的CLIP模型...")
                    # 尝试使用默认的CLIP模型，不指定pretrained
                    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
            else:
                # 否则从网络下载（这可能会失败，如果没有网络连接）
                print("⚠️ 本地模型文件不存在，尝试从网络加载...")
                try:
                    # 尝试使用laion2b_s32b_b82k权重
                    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
                except Exception as e:
                    print(f"⚠️ 从网络加载失败: {e}")
                    print("⚠️ 尝试使用默认的CLIP模型...")
                    # 尝试使用默认的CLIP模型，不指定pretrained
                    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
            
            # 将模型转换为INT8精度
            print("🔄 正在将模型转换为INT8精度...")
            try:
                # 使用动态量化，适用于CPU推理
                model_int8 = torch.quantization.quantize_dynamic(
                    model,  # 要量化的模型
                    {torch.nn.Linear, torch.nn.Conv2d},  # 要量化的层类型
                    dtype=torch.qint8  # 量化目标类型
                )
                print("✅ 模型已成功转换为INT8精度")
            except Exception as e:
                print(f"⚠️ 模型量化失败: {e}")
                print("⚠️ 使用原始精度模型")
                model_int8 = model
            
            # 更新实例的预处理方法，因为CLIP有自己的预处理
            self.clip_preprocess = preprocess
            return model_int8
        except ImportError:
            print("❌ 需要安装 open_clip 包")
            print("请运行: pip install open_clip_torch")
            raise
        except Exception as e:
            print(f"❌ OpenCLIP ViT-L/14 模型加载失败: {e}")
            print("请确保已安装 open_clip_torch 包并确保网络连接正常")
            print("\n📁 手动下载模型文件到以下目录:")
            print(f"{Path(self.cache_dir) / 'timm' / 'vit_large_patch14_clip_224.laion2b_e16'}")
            print("\n🔗 模型下载链接:")
            print("https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_model.safetensors")
            print("\n📄 下载后重命名为:")
            print("open_clip_model_vit_l_14.safetensors")
            raise
    
    def _load_dinov2_vit_s(self, weights=None):
        """
        加载DINOv2 ViT-S模型
        
        参数:
            weights: 权重参数（不使用）
            
        返回:
            model: DINOv2 ViT-S模型
        """
        try:
            import torch
            import torchvision.models as models
            
            # 检查是否安装了torchvision>=0.16.0（支持DINOv2）
            from torchvision import __version__
            version = tuple(map(int, __version__.split('.')[:2]))
            if version < (0, 16):
                print("⚠️ torchvision版本过低，可能不支持DINOv2")
                print("建议安装 torchvision>=0.16.0")
            
            # 尝试加载DINOv2 ViT-S模型
            # 注意：DINOv2的正确模型名称是'dinov2_vits14'，其中'vits'表示ViT-Small，'14'表示patch size
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            print("✅ DINOv2 ViT-S模型加载成功")
            return model
        except ImportError as e:
            print("❌ 导入错误，可能需要安装相关依赖")
            print(f"错误: {e}")
            print("请确保已安装 torch 和 torchvision")
            raise
        except Exception as e:
            print(f"❌ DINOv2模型加载失败: {e}")
            print("请确保网络连接正常，或已在本地缓存模型")
            raise
    
    def get_cached_models(self):
        """获取已缓存的模型列表"""
        cached_models = []
        for model_name, filename in self.model_files.items():
            if (self.cache_dir / filename).exists():
                cached_models.append(model_name)
        return cached_models
    
    def clear_cache(self, model_name=None):
        """清理缓存"""
        if model_name:
            # 清理指定模型
            model_path = self.get_model_path(model_name)
            if model_path.exists():
                model_path.unlink()
                print(f"🗑 已删除缓存: {model_name}")
        else:
            # 清理所有缓存
            for file in self.cache_dir.glob("*.pth"):
                file.unlink()
            print(f"🗑 已清理所有缓存")

# ========== 3. 通用特征提取器 ==========
class FeatureExtractor:
    """通用特征提取器，支持多种模型"""
    
    SUPPORTED_MODELS = ['resnet50', 'efficientnet_lite0', 'mobilenet_v3_small', 'convnext_tiny', 'openclip_vit_b_32', 'openclip_vit_l_14', 'dinov2_vit_s']
    
    def __init__(self, model_name='resnet50', device='auto', cache_dir="./model_cache"):
        """
        初始化特征提取器
        
        参数:
            model_name: 模型名称
            device: 设备类型
            cache_dir: 模型缓存目录
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}。支持: {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_manager = ModelCacheManager(cache_dir)
        self.model = None
        self.preprocess = None
        
        # 添加线程锁，确保模型在多线程环境下的安全性
        import threading
        self.model_lock = threading.Lock()
        
        self._initialize_model()
        self._initialize_preprocess()
    
    def _get_device(self, device):
        """获取可用的设备"""
        if device == 'auto':
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif device == 'mps' and torch.backends.mps.is_available():
            return torch.device("mps")
        elif device == 'cuda' and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _initialize_model(self):
        """初始化模型"""
        print(f"正在加载 {self.model_name} 模型...")
        
        # 从缓存加载或下载模型
        model = self.cache_manager.load_model_from_cache(self.model_name)
        
        # 修改模型结构，提取特征
        if self.model_name == 'resnet50':
            # ResNet50：移除最后的全连接层
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
            
        elif self.model_name == 'efficientnet_lite0':
            # EfficientNet-Lite0：移除最后的分类层
            # EfficientNet 模型有 _fc 层作为分类器
            if hasattr(model, '_fc'):
                model._fc = nn.Identity()
            elif hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
            elif hasattr(model, 'fc'):
                model.fc = nn.Identity()
            else:
                # 如果以上都没有，尝试移除最后一层
                model = nn.Sequential(*list(model.children())[:-1])
            
            self.model = model
            self.feature_dim = 1280
                
        elif self.model_name == 'mobilenet_v3_small':
            # MobileNetV3-Small：移除最后的分类层
            if hasattr(model, 'classifier'):
                # MobileNetV3 的分类器通常是 [GlobalAveragePool, Dropout, Linear]
                features = list(model.children())[:-1]  # 移除 classifier
                # 添加全局平均池化层以获得固定大小的特征
                features.append(nn.AdaptiveAvgPool2d((1, 1)))
                self.model = nn.Sequential(*features)
                self.feature_dim = 576  # MobileNetV3-Small 的特征维度
            else:
                # 如果没有classifier，直接移除最后几层
                self.model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 576
                
        elif self.model_name == 'convnext_tiny':
            # ConvNeXt-Tiny：移除最后的分类层
            if hasattr(model, 'classifier'):
                # ConvNeXt 的分类器通常是 LayerNorm + AdaptiveAvgPool + Linear
                features = list(model.children())[:-1]  # 移除 classifier
                self.model = nn.Sequential(*features)
                self.feature_dim = 768  # ConvNeXt-Tiny 的特征维度
            else:
                self.model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 768
                
        elif self.model_name == 'openclip_vit_b_32':
            # OpenCLIP ViT-B/32：移除最后的投影层
            # OpenCLIP 模型结构不同，通常有transformer和projection两部分
            if hasattr(model, 'visual'):
                # 使用视觉编码器部分
                visual_model = model.visual
                # 保留transformer部分，但移除最终的投影层
                self.model = visual_model
                self.feature_dim = 512  # ViT-B/32 视觉编码器的特征维度
            else:
                # 如果没有单独的visual组件，则使用整个模型
                self.model = model
                self.feature_dim = 512
        elif self.model_name == 'openclip_vit_l_14':
            # OpenCLIP ViT-L/14：移除最后的投影层
            # OpenCLIP 模型结构不同，通常有transformer和projection两部分
            # 注意：量化后的模型可能没有visual属性，直接使用整个模型
            try:
                if hasattr(model, 'visual'):
                    # 使用视觉编码器部分
                    visual_model = model.visual
                    # 保留transformer部分，但移除最终的投影层
                    self.model = visual_model
                    self.feature_dim = 768  # ViT-L/14 视觉编码器的特征维度
                else:
                    # 如果没有单独的visual组件，则使用整个模型
                    self.model = model
                    self.feature_dim = 768
            except Exception as e:
                print(f"⚠️ 处理openclip_vit_l_14模型时出错: {e}")
                print("⚠️ 直接使用整个模型")
                # 直接使用整个模型
                self.model = model
                self.feature_dim = 768
        
        elif self.model_name == 'dinov2_vit_s':
            # DINOv2 ViT-S：使用整个模型，提取CLS token的特征
            # DINOv2模型输出包含多个部分，我们使用最后一层的特征
            self.model = model
            self.feature_dim = 384  # ViT-S 的特征维度
                
        # 将模型移到设备
        # 注意：量化后的模型只能在CPU上运行
        if self.model_name == 'openclip_vit_l_14':
            # 检查模型是否被量化
            is_quantized = any(hasattr(module, 'qconfig') for module in self.model.modules())
            if is_quantized:
                print("⚠️ 量化模型只能在CPU上运行，将设备设置为CPU")
                self.device = torch.device("cpu")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ {self.model_name} 加载完成")
        print(f"  设备: {self.device}")
        print(f"  特征维度: {self.feature_dim}")
    
    def _initialize_preprocess(self):
        """初始化预处理管道"""
        # 根据模型类型选择预处理参数
        if self.model_name in ['resnet50', 'efficientnet_lite0', 'mobilenet_v3_small']:
            # 使用ImageNet预处理参数
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        elif self.model_name == 'convnext_tiny':
            # ConvNeXt模型使用相同的预处理参数
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.456],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        elif self.model_name in ['openclip_vit_b_32', 'openclip_vit_l_14']:
            # OpenCLIP使用特定的预处理，已在模型加载时设置
            # 如果没有预处理函数，则使用默认的
            if hasattr(self.cache_manager, 'clip_preprocess'):
                self.preprocess = self.cache_manager.clip_preprocess
            else:
                # 默认预处理
                self.preprocess = transforms.Compose([
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]
                    ),
                ])
        
        elif self.model_name == 'dinov2_vit_s':
            # DINOv2使用特定的预处理参数
            self.preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    # ========== 对比学习相关代码已注释 ==========
    # def _initialize_contrastive_augmentation(self):
    #     """初始化对比学习的数据增强管道"""
    #     self.contrastive_augmentation = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225]
    #         ),
    #     ])
    
    def extract_features(self, image_path):
        """
        从单张图片提取特征
        
        参数:
            image_path: 图片路径
            
        返回:
            features: 特征向量
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        
        # 加载图片
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"无法加载图片: {e}")
        
        # 预处理
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(input_batch)
            
            # 根据模型类型处理输出
            if self.model_name == 'resnet50':
                features = features.squeeze()  # [1, 2048, 1, 1] -> [2048]
            elif self.model_name == 'efficientnet_lite0':
                # EfficientNet 移除 _fc 层后，输出已经是 [batch_size, 1280]
                if len(features.shape) == 4:
                    features = features.mean([2, 3]).squeeze()  # 全局平均池化
                else:
                    features = features.squeeze()
            elif self.model_name == 'mobilenet_v3_small':
                # MobileNetV3-Small 经过自定义结构后，输出 [batch_size, 576, 1, 1]
                features = features.squeeze()  # [1, 576, 1, 1] -> [576]
            elif self.model_name == 'convnext_tiny':
                # ConvNeXt-Tiny 输出 [batch_size, 768, 1, 1]，需要全局平均池化
                if len(features.shape) == 4:
                    features = features.mean([2, 3]).squeeze()  # 全局平均池化
                else:
                    features = features.squeeze()
            elif self.model_name in ['openclip_vit_b_32', 'openclip_vit_l_14']:
                # OpenCLIP 模型输出特征向量
                if len(features.shape) == 4:
                    features = features.mean([2, 3]).squeeze()  # 全局平均池化
                else:
                    features = features.squeeze()
            elif self.model_name == 'dinov2_vit_s':
                # DINOv2 ViT-S 输出包含CLS token和patch tokens
                # 我们使用CLS token的特征，它是输出的第一个元素
                if isinstance(features, dict):
                    # 如果输出是字典，获取最后一层的特征
                    if 'last_hidden_state' in features:
                        features = features['last_hidden_state'][:, 0].squeeze()  # 获取CLS token
                    else:
                        # 尝试获取其他可能的特征键
                        for key in features:
                            if isinstance(features[key], torch.Tensor):
                                features = features[key]
                                if len(features.shape) > 1:
                                    features = features[:, 0].squeeze()
                                break
                elif len(features.shape) == 3:
                    # 如果输出是 [batch_size, seq_len, hidden_dim]，获取CLS token
                    features = features[:, 0].squeeze()  # 获取第一个token (CLS)
                else:
                    features = features.squeeze()
        
        features = features.cpu().numpy()
        
        return features
    
    def extract_batch_features(self, image_paths, show_progress=True, base_dir=None, batch_size=32):
        """
        批量提取特征
        
        参数:
            image_paths: 图片路径列表
            show_progress: 是否显示进度
            base_dir: 基础目录路径，用于计算相对路径
            batch_size: 批量大小
            
        返回:
            features_dict: 字典 {相对路径: 特征向量}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        features_dict = {}
        total_images = len(image_paths)
        
        # 生成所有批次的索引
        batch_indices = [(i, min(i + batch_size, total_images)) for i in range(0, total_images, batch_size)]
        
        # 获取CPU核心数，设置线程池大小
        max_workers = min(8, os.cpu_count() or 4)
        
        # 使用线程池并行处理批次
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次处理任务
            future_to_batch = {}
            for batch_start, batch_end in batch_indices:
                batch_paths = image_paths[batch_start:batch_end]
                future = executor.submit(self._extract_batch_features, batch_paths)
                future_to_batch[future] = (batch_paths, batch_start, batch_end)
            
            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_batch):
                batch_paths, batch_start, batch_end = future_to_batch[future]
                try:
                    batch_features = future.result()
                    
                    # 保存到字典
                    for img_path, features in zip(batch_paths, batch_features):
                        if base_dir:
                            rel_path = os.path.relpath(img_path, base_dir)
                        else:
                            rel_path = os.path.basename(img_path)
                        features_dict[rel_path] = features
                    
                    completed += len(batch_paths)
                    if show_progress:
                        print(f"  [{completed}/{total_images}] 处理完成")
                except Exception as e:
                    print(f"  ⚠ 处理批次失败 {batch_start}-{batch_end}: {e}")
        
        return features_dict
    
    def _extract_batch_features(self, image_paths):
        """
        内部批量特征提取方法
        
        参数:
            image_paths: 图片路径列表
            
        返回:
            batch_features: 批量特征向量列表
        """
        batch_features = []
        valid_images = []
        valid_indices = []
        
        # 加载和预处理图片
        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"⚠ 图片不存在: {image_path}")
                batch_features.append(None)  # 占位符
                continue
            
            # 加载图片
            try:
                image = Image.open(image_path).convert('RGB')
                valid_images.append(image)
                valid_indices.append(i)
                batch_features.append(None)  # 占位符
            except Exception as e:
                print(f"⚠ 无法加载图片 {image_path}: {e}")
                batch_features.append(None)  # 占位符
        
        if not valid_images:
            return batch_features
        
        # 预处理图片
        input_tensors = []
        for image in valid_images:
            input_tensor = self.preprocess(image)
            input_tensors.append(input_tensor)
        
        # 转换为批次张量
        input_batch = torch.stack(input_tensors).to(self.device)
        
        # 提取特征，使用线程锁确保线程安全
        try:
            with torch.no_grad():
                with self.model_lock:
                    features = self.model(input_batch)
                    
                    # 根据模型类型处理输出
                    if self.model_name == 'resnet50':
                        # [batch_size, 2048, 1, 1] -> [batch_size, 2048]
                        features = features.squeeze()
                        if len(features.shape) == 1:  # 处理批次大小为1的情况
                            features = features.unsqueeze(0)
                    elif self.model_name == 'efficientnet_lite0':
                        # EfficientNet 移除 _fc 层后，输出已经是 [batch_size, 1280]
                        if len(features.shape) == 4:
                            features = features.mean([2, 3])  # 全局平均池化
                    elif self.model_name == 'mobilenet_v3_small':
                        # MobileNetV3-Small 经过自定义结构后，输出 [batch_size, 576, 1, 1]
                        features = features.squeeze()
                        if len(features.shape) == 1:  # 处理批次大小为1的情况
                            features = features.unsqueeze(0)
                    elif self.model_name == 'convnext_tiny':
                        # ConvNeXt-Tiny 输出 [batch_size, 768, 1, 1]，需要全局平均池化
                        if len(features.shape) == 4:
                            features = features.mean([2, 3])  # 全局平均池化
                    elif self.model_name in ['openclip_vit_b_32', 'openclip_vit_l_14']:
                        # OpenCLIP 模型输出特征向量
                        if isinstance(features, tuple):
                            # 对于某些CLIP模型，输出可能是元组
                            features = features[0]  # 取第一个元素作为特征
                        
                        if len(features.shape) == 4:
                            features = features.mean([2, 3])  # 全局平均池化
                        elif len(features.shape) == 3:
                            # 对于某些CLIP模型，输出可能是 [batch_size, seq_len, hidden_dim]
                            # 取CLS token的特征（第一个token）
                            features = features[:, 0]  # 取第一个token
        
            # 转换为numpy数组
            features_np = features.cpu().numpy()
            
            # 填充结果
            for i, idx in enumerate(valid_indices):
                feature = features_np[i]
                batch_features[idx] = feature
        except Exception as e:
            print(f"❌ 批量特征提取失败: {e}")
            print(f"  批次大小: {len(image_paths)}")
            print(f"  有效图片数: {len(valid_images)}")
            print(f"  模型类型: {self.model_name}")
            print(f"  设备: {self.device}")
            
            # 打印前几张图片的路径，以便定位问题图片
            if image_paths:
                print(f"  前5张图片路径:")
                for path in image_paths[:5]:
                    print(f"    - {path}")
            
            # 为失败的批次返回None
            for i in range(len(batch_features)):
                batch_features[i] = None
        
        return batch_features
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': self.model_name,
            'device': str(self.device),
            'feature_dim': self.feature_dim,
            'cached': self.cache_manager.is_model_cached(self.model_name)
        }
    
    # def generate_contrastive_pairs(self, image_paths):
    #     """
    #     生成对比学习的正样本对
        
    #     参数:
    #         image_paths: 图片路径列表
            
    #     返回:
    #         pairs: 元组列表 [(aug1, aug2) ...]，每个元素是同一图片的两个增强版本
    #     """
    #     self._initialize_contrastive_augmentation()
    #     pairs = []
        
    #     for img_path in image_paths:
    #         try:
    #             # 加载原始图片
    #             img = Image.open(img_path).convert('RGB')
                
    #             # 生成两个增强版本
    #             aug1 = self.contrastive_augmentation(img)
    #             aug2 = self.contrastive_augmentation(img)
                
    #             pairs.append((aug1, aug2))
    #         except Exception as e:
    #             print(f"⚠ 生成对比样本对失败 {img_path}: {e}")
        
    #     return pairs
    
    # def train_contrastive(self, image_paths, epochs=10, learning_rate=1e-4, batch_size=32, temperature=0.5):
    #     """
    #     使用对比学习训练模型
        
    #     参数:
    #         image_paths: 训练图片路径列表
    #         epochs: 训练轮数
    #         learning_rate: 学习率
    #         batch_size: 批次大小
    #         temperature: 对比损失的温度参数
            
    #     返回:
    #         history: 训练历史，包含每轮的损失值
    #     """
    #     import torch.optim as optim
    #     import torch.utils.data as data
        
    #     print(f"\n{'='*60}")
    #     print("🔧 开始对比学习训练")
    #     print(f"{'='*60}")
        
    #     # 将模型切换到训练模式
    #     self.model.train()
        
    #     # 生成对比样本对
    #     pairs = self.generate_contrastive_pairs(image_paths)
        
    #     if not pairs:
    #         print("❌ 没有生成任何对比样本对，训练失败")
    #         return None
        
    #     # 创建数据加载器
    #     dataset = data.TensorDataset(
    #         torch.stack([pair[0] for pair in pairs]),
    #         torch.stack([pair[1] for pair in pairs])
    #     )
    #     dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    #     # 初始化优化器
    #     optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    #     # 初始化ContrastiveLearner
    #     from torch.nn.functional import normalize
        
    #     history = []
    #     pytor
        
    #     for epoch in range(epochs):
    #         running_loss = 0.0
            
    #         for i, (aug1, aug2) in enumerate(dataloader):
    #             # 将数据移到设备
    #             aug1 = aug1.to(self.device)
    #             aug2 = aug2.to(self.device)
                
    #             # 零梯度
    #             optimizer.zero_grad()
                
    #             # 提取特征
    #             features1 = self.model(aug1)
    #             features2 = self.model(aug2)
                
    #             # 处理特征格式
    #             if self.model_name.startswith('vgg'):
    #                 features1 = features1.view(features1.size(0), -1)
    #                 features2 = features2.view(features2.size(0), -1)
    #             elif self.model_name.startswith('resnet'):
    #                 features1 = features1.squeeze()
    #                 features2 = features2.squeeze()
    #             elif self.model_name == 'mobilenet_v2':
    #                 features1 = features1.mean([2, 3])
    #                 features2 = features2.mean([2, 3])
                
    #             # 拼接特征
    #             features = torch.cat([features1, features2], dim=0)
                
    #             # L2归一化
    #             features = normalize(features, dim=1)
                
    #             # 计算相似度矩阵
    #             similarity_matrix = torch.mm(features, features.t()) / temperature
                
    #             # 创建标签
    #             batch_size = features1.size(0)
    #             labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
                
    #             # 创建掩码矩阵
    #             mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    #             mask = mask.fill_diagonal_(0)
                
    #             # 计算正样本
    #             positive_pairs = similarity_matrix[mask]
                
    #             # 计算负样本
    #             exp_similarity = torch.exp(similarity_matrix)
    #             sum_exp = torch.sum(exp_similarity, dim=1) - torch.exp(similarity_matrix.diag())
                
    #             # 获取正样本对的索引
    #             positive_indices = torch.nonzero(mask, as_tuple=True)
    #             sum_exp_positive = sum_exp[positive_indices[0]]
                
    #             # 计算损失
    #             loss = -torch.log(positive_pairs / sum_exp_positive)
    #             loss = loss.mean()
                
    #             # 反向传播
    #             loss.backward()
    #             optimizer.step()
                
    #             running_loss += loss.item()
                
    #         # 计算平均损失
    #         avg_loss = running_loss / len(dataloader)
    #         history.append(avg_loss)
            
    #         print(f"Epoch {epoch+1:3d}/{epochs:3d} | Loss: {avg_loss:.6f}")
        
    #     # 将模型切回评估模式
    #     self.model.eval()
        
    #     print(f"\n✅ 对比学习训练完成")
    #     print(f"最后的损失值: {history[-1]:.6f}")
        
    #     return history
    
    def save_trained_model(self, save_path):
        """
        保存训练后的模型
        
        参数:
            save_path: 保存路径
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"✓ 训练后的模型已保存: {save_path}")
    
    def load_trained_model(self, load_path):
        """
        加载训练后的模型
        
        参数:
            load_path: 加载路径
        """
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"✓ 训练后的模型已加载: {load_path}")

# ========== 4. 相似度分析器 (保持不变) ==========
class SimilarityAnalyzer:
    """相似度分析器"""
    
    def __init__(self):
        self.similarity_methods = {
            'cosine': self.cosine_similarity,
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance,
        }
    
    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)
    
    def manhattan_distance(self, vec1, vec2):
        return np.sum(np.abs(vec1 - vec2))
    
    def compute_similarity_matrix(self, features_dict, method='cosine'):
        if method not in self.similarity_methods:
            raise ValueError(f"不支持的相似度方法: {method}")
        
        image_names = list(features_dict.keys())
        n_images = len(image_names)
        similarity_func = self.similarity_methods[method]
        
        similarity_matrix = np.zeros((n_images, n_images))
        
        for i in range(n_images):
            for j in range(n_images):
                if i == j:
                    if method == 'cosine':
                        similarity_matrix[i, j] = 1.0
                    else:
                        similarity_matrix[i, j] = 0.0
                else:
                    sim = similarity_func(
                        features_dict[image_names[i]], 
                        features_dict[image_names[j]]
                    )
                    similarity_matrix[i, j] = sim
        
        return similarity_matrix, image_names
    
    def print_similarity_analysis(self, features_dict, method='cosine'):
        print(f"\n{'='*60}")
        print(f"相似度分析 (方法: {method})")
        print(f"{'='*60}")
        
        image_names = list(features_dict.keys())
        similarity_func = self.similarity_methods[method]
        
        dog_images = [name for name in image_names if 'dog' in name.lower()]
        cat_images = [name for name in image_names if 'cat' in name.lower()]
        
        print(f"\n图片分类:")
        print(f"  狗类图片 ({len(dog_images)}张): {', '.join(dog_images)}")
        print(f"  猫类图片 ({len(cat_images)}张): {', '.join(cat_images)}")
        
        print(f"\n类内相似度 (同类图片之间的相似度):")
        
        if len(dog_images) > 1:
            dog_similarities = []
            for i in range(len(dog_images)):
                for j in range(i+1, len(dog_images)):
                    sim = similarity_func(
                        features_dict[dog_images[i]], 
                        features_dict[dog_images[j]]
                    )
                    dog_similarities.append(sim)
            avg_dog_sim = np.mean(dog_similarities)
            print(f"  狗类平均相似度: {avg_dog_sim:.4f} (范围: {np.min(dog_similarities):.4f} - {np.max(dog_similarities):.4f})")
        
        if len(cat_images) > 1:
            cat_similarities = []
            for i in range(len(cat_images)):
                for j in range(i+1, len(cat_images)):
                    sim = similarity_func(
                        features_dict[cat_images[i]], 
                        features_dict[cat_images[j]]
                    )
                    cat_similarities.append(sim)
            avg_cat_sim = np.mean(cat_similarities)
            print(f"  猫类平均相似度: {avg_cat_sim:.4f} (范围: {np.min(cat_similarities):.4f} - {np.max(cat_similarities):.4f})")
        
        print(f"\n类间相似度 (不同类图片之间的相似度):")
        cross_similarities = []
        for dog_img in dog_images:
            for cat_img in cat_images:
                sim = similarity_func(
                    features_dict[dog_img], 
                    features_dict[cat_img]
                )
                cross_similarities.append(sim)
        
        if cross_similarities:
            avg_cross_sim = np.mean(cross_similarities)
            print(f"  狗猫平均相似度: {avg_cross_sim:.4f} (范围: {np.min(cross_similarities):.4f} - {np.max(cross_similarities):.4f})")
            
            if len(dog_images) > 1 and len(cat_images) > 1:
                avg_within = (avg_dog_sim + avg_cat_sim) / 2
                # contrast_score = avg_within - avg_cross_sim  # 对比学习相关，已注释
                # 对比学习指标部分已注释
                # print(f"\n对比学习指标:")
                # print(f"  类内平均相似度: {avg_within:.4f}")
                # print(f"  类间平均相似度: {avg_cross_sim:.4f}")
                # print(f"  对比得分 (类内-类间): {contrast_score:.4f}")
                # if contrast_score > 0:
                #     print(f"  ✓ 模型能够区分猫狗 (对比得分为正)")
                # else:
                #     print(f"  ⚠ 模型难以区分猫狗 (对比得分为负或零)")

# ========== 5.1 对比学习实现 (已注释) ==========
# class ContrastiveLearner:
#     """简单的对比学习实现"""
    
#     def __init__(self):
#         self.temperature = 0.5  # 温度参数，用于控制相似度分布的平滑程度
    
#     def contrastive_loss(self, features, labels):
#         """
#         计算对比损失
        
#         参数:
#             features: 特征向量列表，形状为 [batch_size, feature_dim]
#             labels: 样本标签列表，形状为 [batch_size]
            
#         返回:
#             loss: 对比损失值
#         """
#         import torch
        
#         # 将特征转换为张量
#         features = torch.tensor(features)
#         labels = torch.tensor(labels)
        
#         # 计算特征向量的L2范数-归一化
#         features = features / torch.norm(features, dim=1, keepdim=True)
        
#         # 计算相似性矩阵
#         similarity_matrix = torch.mm(features, features.t()) / self.temperature
        
#         # 创建掩码矩阵，区分正样本和负样本
#         mask = labels.unsqueeze(0) == labels.unsqueeze(1)
#         mask = mask.fill_diagonal_(0)  # 排除自身
        
#         # 计算正样本损失
#         positive_pairs = similarity_matrix[mask]
#         if len(positive_pairs) == 0:
#             return 0.0  # 没有正样本对时，损失为0
        
#         # 计算负样本损失
#         exp_similarity = torch.exp(similarity_matrix)
#         sum_exp = torch.sum(exp_similarity, dim=1) - torch.exp(similarity_matrix.diag())
        
#         # 获取正样本对的索引
#         positive_indices = torch.nonzero(mask, as_tuple=True)
        
#         # 为每个正样本对获取对应的sum_exp值
#         sum_exp_positive = sum_exp[positive_indices[0]]
        
#         # 计算每个样本的损失
#         loss = -torch.log(positive_pairs / sum_exp_positive)
        
#         return torch.mean(loss).item()
    
#     def create_contrastive_pairs(self, features_dict):
#         """
#         创建对比学习的数据对
        
#         参数:
#             features_dict: 特征字典 {图片名: 特征向量}
            
#         返回:
#             features: 特征向量数组
#             labels: 样本标签数组 (0=cat, 1=dog)
#         """
#         features = []
#         labels = []
        
#         for img_name, feature in features_dict.items():
#             features.append(feature)
#             # 根据文件名判断标签
#             if 'cat' in img_name.lower():
#                 labels.append(0)
#             elif 'dog' in img_name.lower():
#                 labels.append(1)
#             else:
#                 labels.append(2)  # 其他类别
        
#         return np.array(features), np.array(labels)
    
#     def run_contrastive_learning_example(self, features_dict):
#         """
#         运行对比学习示例
        
#         参数:
#             features_dict: 特征字典 {图片名: 特征向量}
#         """
#         print(f"\n{'='*60}")
#         print("🔬 对比学习示例")
#         print(f"{'='*60}")
        
#         # 创建对比学习数据
#         features, labels = self.create_contrastive_pairs(features_dict)
        
#         print(f"对比学习数据:")
#         print(f"  - 总样本数: {len(features)}")
#         print(f"  - 特征维度: {features.shape[1]}")
#         print(f"  - 猫类样本 (标签0): {np.sum(labels == 0)}个")
#         print(f"  - 狗类样本 (标签1): {np.sum(labels == 1)}个")
        
#         # 计算对比损失
#         if len(features) < 2:
#             print("⚠ 样本数不足，无法进行对比学习")
#             return
        
#         loss = self.contrastive_loss(features, labels)
#         print(f"\n对比损失值: {loss:.4f}")
        
#         # 解释结果
#         print(f"\n对比学习结果分析:")
#         if loss < 1.0:
#             print("  ✓ 特征向量具有较好的区分能力")
#         elif loss < 2.0:
#             print("  ⚠ 特征向量的区分能力一般")
#         else:
#             print("  ❌ 特征向量的区分能力较弱")
        
#         print(f"\n📝 对比学习说明:")
#         print(f"  - 对比学习目标是让同类样本的特征更相似，不同类样本的特征更不同")
#         print(f"  - 损失值越小，表示特征的区分能力越强")
#         print(f"  - 温度参数({self.temperature})控制相似度分布的平滑程度")

# ========== 5.2 可视化工具 (优化) ==========
class Visualizer:
    """可视化工具"""
    
    def __init__(self):
        self.colors = {
            'dog': '#FF6B6B',
            'cat': '#4ECDC4',
        }
    
    def create_similarity_heatmap(self, similarity_matrix, image_names, model_name, save_dir='./output'):
        """创建相似度热图"""
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f'similarity_heatmap_{model_name}.png')
        
        # plt.figure(figsize=(10, 8))
        # im = plt.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        
        # short_names = [n.split('.')[0] for n in image_names]
        # plt.xticks(range(len(image_names)), short_names, rotation=45, ha='right')
        # plt.yticks(range(len(image_names)), short_names)
        
        # for i in range(len(image_names)):
        #     for j in range(len(image_names)):
        #         color = 'black' if similarity_matrix[i, j] < 0.7 else 'white'
        #         plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
        #                 ha='center', va='center', color=color, fontsize=9)
        
        # plt.colorbar(im, fraction=0.046, pad=0.04)
        # plt.title(f'图片相似度矩阵 - {model_name}', fontsize=14, fontweight='bold')
        # plt.tight_layout()
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"✓ 相似度热图已保存: {save_path}")
        print(f"⚠ 相似度热图功能已禁用")
    
    def create_feature_scatter(self, features_dict, model_name, save_dir='./output'):
        """创建特征散点分布图"""
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f'feature_scatter_{model_name}.png')
        
        # image_names = list(features_dict.keys())
        # features = np.array([features_dict[name] for name in image_names])
        
        # pca = PCA(n_components=2)
        # features_2d = pca.fit_transform(features)
        
        # plt.figure(figsize=(12, 10))
        
        # for i, img_name in enumerate(image_names):
        #     if 'dog' in img_name.lower():
        #         color = self.colors['dog']
        #         marker = 'o'
        #     else:
        #         color = self.colors['cat']
        #         marker = 's'
            
        #     plt.scatter(features_2d[i, 0], features_2d[i, 1], 
        #                color=color, s=200, marker=marker, 
        #                edgecolor='black', linewidth=1.5, alpha=0.8)
            
        #     plt.annotate(img_name.split('.')[0], 
        #                 xy=(features_2d[i, 0], features_2d[i, 1]),
        #                 xytext=(5, 5), textcoords='offset points',
        #                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
        #                                       facecolor='white', 
        #                                       edgecolor='gray', alpha=0.8))
        
        # from matplotlib.patches import Patch 
        # legend_elements = [
        #     Patch(facecolor=self.colors['dog'], edgecolor='black', label='狗'),
        #     Patch(facecolor=self.colors['cat'], edgecolor='black', label='猫'),
        # ]
        # plt.legend(handles=legend_elements, loc='upper right')
        
        # explained_var = pca.explained_variance_ratio_
        # plt.title(f'特征空间分布 - {model_name}', fontsize=16, fontweight='bold')
        # plt.xlabel(f'主成分 1 ({explained_var[0]:.2%})', fontsize=12)
        # plt.ylabel(f'主成分 2 ({explained_var[1]:.2%})', fontsize=12)
        # plt.grid(True, alpha=0.3, linestyle='--')
        
        # plt.tight_layout()
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"✓ 特征散点图已保存: {save_path}")
        print(f"⚠ 特征散点图功能已禁用")
    
    def create_tsne_visualization(self, features_dict, model_name, save_dir='./output'):
        """创建t-SNE可视化"""
        # if len(features_dict) < 3:
        #     print("⚠ 样本数不足，跳过t-SNE可视化")
        #     return
        
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f'tsne_visualization_{model_name}.png')
        
        # image_names = list(features_dict.keys())
        # features = np.array([features_dict[name] for name in image_names])
        
        # tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(features)-1))
        # features_2d = tsne.fit_transform(features)
        
        # plt.figure(figsize=(12, 10))
        
        # for i, img_name in enumerate(image_names):
        #     if 'dog' in img_name.lower():
        #         color = self.colors['dog']
        #         marker = 'o'
        #     else:
        #         color = self.colors['cat']
        #         marker = 's'
            
        #     plt.scatter(features_2d[i, 0], features_2d[i, 1], 
        #                color=color, s=200, marker=marker,
        #                edgecolor='black', linewidth=1.5, alpha=0.8)
            
        #     plt.annotate(img_name.split('.')[0], 
        #                 xy=(features_2d[i, 0], features_2d[i, 1]),
        #                 xytext=(5, 5), textcoords='offset points',
        #                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
        #                                       facecolor='white', 
        #                                       edgecolor='gray', alpha=0.8))
        
        # plt.title(f'特征空间分布 (t-SNE) - {model_name}', fontsize=16, fontweight='bold')
        # plt.xlabel('t-SNE 1', fontsize=12)
        # plt.ylabel('t-SNE 2', fontsize=12)
        # plt.grid(True, alpha=0.3, linestyle='--')
        
        # plt.tight_layout()
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"✓ t-SNE可视化已保存: {save_path}")
        print(f"⚠ t-SNE可视化功能已禁用")

# ========== 6. 主程序 (优化) ==========
def main():
    """主函数"""
    print("=" * 60)
    print("🐱🐶 猫狗图片特征提取与相似度分析")
    print("=" * 60)
    
    # 创建缓存管理器
    cache_manager = ModelCacheManager()
    
    # 显示可用模型
    print(f"\n📊 支持的模型:")
    supported_models = FeatureExtractor.SUPPORTED_MODELS
    for i, model in enumerate(supported_models, 1):
        cached = "✓" if cache_manager.is_model_cached(model) else " "
        print(f"  {cached} {i:2d}. {model:12}", end="")
        if i % 3 == 0:
            print()
    
    # 选择模型
    print(f"\n\n🔧 请选择模型 (1-{len(supported_models)}):")
    for i, model in enumerate(supported_models, 1):
        print(f"  {i:2d}. {model}")
    
    try:
        choice = int(input(f"\n请输入选择 (默认1): ") or "1")
        if 1 <= choice <= len(supported_models):
            model_name = supported_models[choice-1]
        else:
            print(f"⚠ 输入无效，使用默认: resnet18")
            model_name = "resnet18"
    except:
        print(f"⚠ 输入无效，使用默认: resnet18")
        model_name = "resnet18"
    
    # 选择设备
    device_options = []
    if torch.backends.mps.is_available():
        device_options.append(("1", "mps", "Apple Silicon"))
    if torch.cuda.is_available():
        device_options.append(("2", "cuda", "NVIDIA GPU"))
    device_options.append(("3", "cpu", "CPU"))
    
    print(f"\n💻 请选择设备:")
    for idx, device_code, description in device_options:
        print(f"  {idx}. {description} ({device_code})")
    
    device_choice = input(f"\n请输入选择 (默认1): ") or "1"
    device_map = {idx: device_code for idx, device_code, _ in device_options}
    device = device_map.get(device_choice, "auto")
    
    # 检查图片目录
    img_dir = "img"
    if not os.path.exists(img_dir):
        print(f"\n❌ 错误: 图片目录 '{img_dir}' 不存在")
        print("请创建 img/ 目录并放入图片")
        return
    
    # 查找图片
    import glob
    image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(img_dir, pattern)))
    
    if not image_paths:
        print(f"\n❌ 在 {img_dir} 中未找到图片")
        return
    
    # 按名称排序
    image_paths.sort()
    print(f"\n📸 找到 {len(image_paths)} 张图片:")
    for i, path in enumerate(image_paths[:10], 1):  # 只显示前10个
        print(f"  {i:2d}. {os.path.basename(path)}")
    if len(image_paths) > 10:
        print(f"  ... 和 {len(image_paths)-10} 张更多图片")
    
    # 确认是否继续
    confirm = input(f"\n是否继续处理这 {len(image_paths)} 张图片? (y/n, 默认y): ") or "y"
    if confirm.lower() != 'y':
        print("👋 已取消")
        return
    
    # 创建特征提取器
    print(f"\n{'='*30}")
    print(f"初始化 {model_name} 特征提取器")
    print(f"{'='*30}")
    
    try:
        extractor = FeatureExtractor(
            model_name=model_name,
            device=device,
            cache_dir="./model_cache"
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 提取特征
    print(f"\n{'='*30}")
    print("提取特征")
    print(f"{'='*30}")
    
    features_dict = extractor.extract_batch_features(image_paths, show_progress=True)
    
    if not features_dict:
        print("❌ 未能提取任何特征")
        return
    
    print(f"\n✅ 成功提取 {len(features_dict)} 张图片的特征")
    
    # 显示特征信息
    first_key = list(features_dict.keys())[0]
    print(f"特征维度: {features_dict[first_key].shape}")
    
    # 询问是否进行对比学习训练 (已注释)
    # print(f"\n{'='*60}")
    # print("🤔 是否进行对比学习训练?")
    # print(f"{'='*60}")
    
    # train_choice = input("输入 'y' 进行训练，其他任意键跳过 (默认: n): ") or "n"
    
    # if train_choice.lower() == 'y':
    #     # 训练配置选项
    #     try:
    #         epochs = int(input("输入训练轮数 (默认: 10): ") or "10")
    #         learning_rate = float(input("输入学习率 (默认: 1e-4): ") or "1e-4")
    #         batch_size = int(input("输入批次大小 (默认: 32): ") or "32")
    #         temperature = float(input("输入温度参数 (默认: 0.5): ") or "0.5")
    #     except ValueError:
    #         print("⚠ 输入无效，使用默认参数")
    #         epochs = 10
    #         learning_rate = 1e-4
    #         batch_size = 32
    #         temperature = 0.5
        
    #     # 进行对比学习训练
    #     history = extractor.train_contrastive(
    #         image_paths,
    #         epochs=epochs,
    #         learning_rate=learning_rate,
    #         batch_size=batch_size,
    #         temperature=temperature
    #     )
        
    #     # 询问是否保存训练后的模型
    #     if history:
    #         save_choice = input("\n是否保存训练后的模型? (y/n, 默认: y): ") or "y"
    #         if save_choice.lower() == 'y':
    #             save_path = f"./output/trained_{model_name}.pt"
    #             extractor.save_trained_model(save_path)
        
    #     # 重新提取特征，使用训练后的模型
    #     print(f"\n{'='*60}")
    #     print("🔄 使用训练后的模型重新提取特征")
    #     print(f"{'='*60}")
    #     features_dict = extractor.extract_batch_features(image_paths, show_progress=True)
    
    # 保存特征
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    features_file = os.path.join(output_dir, f'features_{model_name}.npz')
    np.savez(features_file, **features_dict)
    print(f"✓ 特征已保存: {features_file}")
    
    # 相似度分析
    analyzer = SimilarityAnalyzer()
    
    print(f"\n{'='*30}")
    print("相似度分析")
    print(f"{'='*30}")
    
    # 使用不同方法分析
    methods = ['cosine', 'euclidean', 'manhattan']
    
    for method in methods:
        try:
            analyzer.print_similarity_analysis(features_dict, method=method)
            
            # 计算并保存相似度矩阵
            similarity_matrix, image_names = analyzer.compute_similarity_matrix(
                features_dict, method=method
            )
            
            matrix_file = os.path.join(output_dir, f'similarity_matrix_{method}_{model_name}.npy')
            np.save(matrix_file, similarity_matrix)
            print(f"✓ 相似度矩阵已保存: {matrix_file}")
            
        except Exception as e:
            print(f"警告: {method} 方法分析失败: {e}")
    
    # 可视化
    print(f"\n{'='*30}")
    print("生成可视化")
    print(f"{'='*30}")
    
    visualizer = Visualizer()
    
    # 获取相似度矩阵用于可视化
    similarity_matrix, image_names = analyzer.compute_similarity_matrix(
        features_dict, method='cosine'
    )
    
    # 1. 相似度热图
    visualizer.create_similarity_heatmap(
        similarity_matrix, image_names, model_name, save_dir=output_dir
    )
    
    # 2. PCA散点图
    visualizer.create_feature_scatter(features_dict, model_name, save_dir=output_dir)
    
    # 3. t-SNE可视化
    visualizer.create_tsne_visualization(features_dict, model_name, save_dir=output_dir)
    
    # 4. 对比学习示例 (已注释)
    # print(f"\n{'='*30}")
    # print("对比学习示例")
    # print(f"{'='*30}")
    
    # contrastive_learner = ContrastiveLearner()
    # contrastive_learner.run_contrastive_learning_example(features_dict)
    
    # 打印详细的相似度表格
    print(f"\n{'='*60}")
    print("详细相似度表格 (余弦相似度)")
    print(f"{'='*60}")
    
    print("\n" + " " * 12, end="")
    for name in image_names:
        print(f"{name[:8]:>8}", end="")
    print()
    
    for i, name1 in enumerate(image_names):
        print(f"{name1[:8]:8}", end="")
        for j, name2 in enumerate(image_names):
            similarity = analyzer.cosine_similarity(
                features_dict[name1], 
                features_dict[name2]
            )
            print(f"{similarity:8.3f}", end="")
        print()
    
    # 生成分析报告
    print(f"\n{'='*60}")
    print("📊 分析报告")
    print(f"{'='*60}")
    
    print(f"\n模型信息:")
    model_info = extractor.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 缓存信息
    cached_models = cache_manager.get_cached_models()
    print(f"\n缓存信息:")
    print(f"  已缓存模型: {', '.join(cached_models) if cached_models else '无'}")
    print(f"  缓存目录: {cache_manager.cache_dir.absolute()}")
    
    print(f"\n文件输出:")
    print(f"  - {features_file}: 特征向量")
    for method in methods:
        matrix_file = os.path.join(output_dir, f'similarity_matrix_{method}_{model_name}.npy')
        if os.path.exists(matrix_file):
            print(f"  - {matrix_file}: 相似度矩阵 ({method})")
    
    viz_files = [
        f'similarity_heatmap_{model_name}.png',
        f'feature_scatter_{model_name}.png',
        f'tsne_visualization_{model_name}.png',
    ]
    
    for viz_file in viz_files:
        viz_path = os.path.join(output_dir, viz_file)
        if os.path.exists(viz_path):
            print(f"  - {viz_path}: 可视化图表")
    
    print(f"\n{'='*60}")
    print("🎉 分析完成！")
    print(f"{'='*60}")
    
    # 清理提示
    print(f"\n💡 提示:")
    print(f"  下次运行可以使用已缓存的模型，无需重新下载")
    print(f"  如需清理缓存: python {sys.argv[0]} --clear-cache")

if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--clear-cache":
        cache_manager = ModelCacheManager()
        cache_manager.clear_cache()
        print("缓存已清理")
    else:
        main()