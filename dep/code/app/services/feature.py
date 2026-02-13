# 特征提取服务
import logging
from typing import Dict, List, Optional
from app.services.model_cache import ModelCacheManager
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

class FeatureExtractor:
    """通用特征提取器，支持多种模型"""
    
    SUPPORTED_MODELS = ['resnet50', 'efficientnet_lite0', 'mobilenet_v3_small', 'convnext_tiny', 'openclip_vit_b_32', 'openclip_vit_l_14', 'dinov2_vit_s']
    
    def __init__(self, model_name='resnet50', device='auto', cache_manager=None):
        """
        初始化特征提取器
        
        参数:
            model_name: 模型名称
            device: 设备类型
            cache_manager: 模型缓存管理器实例
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}。支持: {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_manager = cache_manager or ModelCacheManager()
        self.model = None
        self.preprocess = None
        self.feature_dim = None
        
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
            if hasattr(model, '_fc'):
                model._fc = nn.Identity()
            elif hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
            elif hasattr(model, 'fc'):
                model.fc = nn.Identity()
            else:
                model = nn.Sequential(*list(model.children())[:-1])
            
            self.model = model
            self.feature_dim = 1280
                
        elif self.model_name == 'mobilenet_v3_small':
            # MobileNetV3-Small：移除最后的分类层
            if hasattr(model, 'classifier'):
                features = list(model.children())[:-1]
                features.append(nn.AdaptiveAvgPool2d((1, 1)))
                self.model = nn.Sequential(*features)
                self.feature_dim = 576
            else:
                self.model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 576
                
        elif self.model_name == 'convnext_tiny':
            # ConvNeXt-Tiny：移除最后的分类层
            if hasattr(model, 'classifier'):
                features = list(model.children())[:-1]
                self.model = nn.Sequential(*features)
                self.feature_dim = 768
            else:
                self.model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 768
                
        elif self.model_name == 'openclip_vit_b_32':
            # OpenCLIP ViT-B/32：移除最后的投影层
            if hasattr(model, 'visual'):
                visual_model = model.visual
                self.model = visual_model
                self.feature_dim = 512
            else:
                self.model = model
                self.feature_dim = 512
        elif self.model_name == 'openclip_vit_l_14':
            # OpenCLIP ViT-L/14：移除最后的投影层
            try:
                if hasattr(model, 'visual'):
                    visual_model = model.visual
                    self.model = visual_model
                    self.feature_dim = 768
                else:
                    self.model = model
                    self.feature_dim = 768
            except Exception as e:
                print(f"⚠️ 处理openclip_vit_l_14模型时出错: {e}")
                print("⚠️ 直接使用整个模型")
                self.model = model
                self.feature_dim = 768
        
        elif self.model_name == 'dinov2_vit_s':
            # DINOv2 ViT-S：使用整个模型，提取CLS token的特征
            self.model = model
            self.feature_dim = 384
                
        # 将模型移到设备
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
                if len(features.shape) == 4:
                    features = features.mean([2, 3]).squeeze()  # 全局平均池化
                else:
                    features = features.squeeze()
            elif self.model_name == 'mobilenet_v3_small':
                features = features.squeeze()  # [1, 576, 1, 1] -> [576]
            elif self.model_name == 'convnext_tiny':
                if len(features.shape) == 4:
                    features = features.mean([2, 3]).squeeze()  # 全局平均池化
                else:
                    features = features.squeeze()
            elif self.model_name in ['openclip_vit_b_32', 'openclip_vit_l_14']:
                if len(features.shape) == 4:
                    features = features.mean([2, 3]).squeeze()  # 全局平均池化
                else:
                    features = features.squeeze()
            elif self.model_name == 'dinov2_vit_s':
                if isinstance(features, dict):
                    if 'last_hidden_state' in features:
                        features = features['last_hidden_state'][:, 0].squeeze()  # 获取CLS token
                    else:
                        for key in features:
                            if isinstance(features[key], torch.Tensor):
                                features = features[key]
                                if len(features.shape) > 1:
                                    features = features[:, 0].squeeze()
                                break
                elif len(features.shape) == 3:
                    features = features[:, 0].squeeze()  # 获取第一个token (CLS)
                else:
                    features = features.squeeze()
        
        features = features.cpu().numpy()
        
        return features

class FeatureService:
    """特征提取服务"""
    
    def __init__(self, model_names: List[str], cache_dir: str):
        """
        初始化特征提取服务
        
        参数:
            model_names: 模型名称列表
            cache_dir: 模型缓存目录
        """
        self.model_names = model_names
        self.cache_dir = cache_dir
        self.extractors = {}
        self.models = model_names
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """初始化特征提取器"""
        for model_name in self.model_names:
            logging.info(f"加载模型: {model_name}")
            cache_manager = ModelCacheManager(cache_dir=self.cache_dir)
            extractor = FeatureExtractor(model_name=model_name, cache_manager=cache_manager)
            self.extractors[model_name] = extractor
            logging.info(f"✓ 模型加载成功: {model_name}")
    
    def extract_features(self, image_path: str, model_name: str):
        """提取特征"""
        if model_name not in self.extractors:
            raise ValueError(f"模型 {model_name} 未加载")
        
        return self.extractors[model_name].extract_features(image_path)
    
    async def search_similar_images(self, image_path: str, top_k: int, vector_db_service):
        """搜索相似图片"""
        results = {}
        
        for model_name in self.model_names:
            # 提取特征
            features = self.extract_features(image_path, model_name)
            
            # 搜索相似图片
            search_results = vector_db_service.search(
                model_name=model_name,
                query_vector=features,
                top_k=top_k
            )
            
            results[model_name] = {
                "model_name": model_name,
                "similar_images": search_results
            }
        
        return results
    
    def get_models_status(self):
        """获取模型状态"""
        status = {}
        for model_name in self.model_names:
            status[model_name] = model_name in self.extractors
        return status

# 导入os模块
import os
