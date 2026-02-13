# 模型缓存管理器
import os
import ssl
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')

# ========== SSL证书修复 ==========
ssl._create_default_https_context = ssl._create_unverified_context

class ModelCacheManager:
    """模型缓存管理器"""
    
    def __init__(self, cache_dir="./models/cache"):
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
            'resnet50': self._load_resnet50,
            'efficientnet_lite0': self._load_efficientnet_lite0,
            'mobilenet_v3_small': self._load_mobilenet_v3_small,
            'convnext_tiny': self._load_convnext_tiny,
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
    
    def load_model_from_cache(self, model_name):
        """
        从缓存加载模型，如果没有则下载
        
        参数:
            model_name: 模型名称
            
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
                return self.model_loaders[model_name]()
            except:
                # 如果缓存文件损坏，删除并重新下载
                model_path = self.get_model_path(model_name)
                print(f"⚠ 缓存文件损坏，删除: {model_path}")
                model_path.unlink(missing_ok=True)
        
        # 下载模型
        print(f"📥 下载模型: {model_name}")
        try:
            model = self.model_loaders[model_name]()
            print(f"✅ 模型下载完成: {model_name}")
            return model
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            raise
    
    def _load_resnet50(self):
        """加载ResNet50模型"""
        from torchvision import models
        return models.resnet50(pretrained=True)
    
    def _load_efficientnet_lite0(self):
        """
        加载EfficientNet-B0模型（替代EfficientNet-Lite0）
        """
        try:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b0')
            return model
        except ImportError:
            print("❌ 需要安装 efficientnet-pytorch 包")
            print("请运行: pip install efficientnet-pytorch")
            raise
    
    def _load_mobilenet_v3_small(self):
        """加载MobileNetV3-Small模型"""
        from torchvision import models
        return models.mobilenet_v3_small(pretrained=True)
    
    def _load_convnext_tiny(self):
        """加载ConvNeXt-Tiny模型"""
        from torchvision import models
        return models.convnext_tiny(pretrained=True)
    
    def _load_openclip_vit_b_32(self):
        """
        加载OpenCLIP ViT-B/32模型
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
                # 否则从网络下载
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
    
    def _load_openclip_vit_l_14(self):
        """
        加载OpenCLIP ViT-L/14模型
        """
        try:
            import open_clip
            import os
            import torch
            from pathlib import Path
            
            # 构建本地模型文件路径
            openclip_model_paths = [
                Path(self.cache_dir) / "open_clip_pytorch_model.safetensors",
                Path(self.cache_dir) / "open_clip_pytorch_model.bin",
                Path(self.cache_dir) / "model.safetensors",
                Path(self.cache_dir) / "pytorch_model.bin",
                Path(self.cache_dir) / "open_clip_model_vit_l_14.safetensors"
            ]
            
            # 检查用户提供的模型文件路径
            user_model_path = Path(self.cache_dir) / "vit_l_14-laion2b_s32b_b82k.bin"
            
            # 同时检查子目录中的模型文件（兼容旧路径）
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
                # 否则从网络下载
                print("⚠️ 本地模型文件不存在，尝试从网络加载...")
                try:
                    # 尝试使用laion2b_s32b_b82k权重
                    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
                except Exception as e:
                    print(f"⚠️ 从网络加载失败: {e}")
                    print("⚠️ 尝试使用默认的CLIP模型...")
                    # 尝试使用默认的CLIP模型，不指定pretrained
                    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
            
            # 更新实例的预处理方法，因为CLIP有自己的预处理
            self.clip_preprocess = preprocess
            return model
        except ImportError:
            print("❌ 需要安装 open_clip 包")
            print("请运行: pip install open_clip_torch")
            raise
        except Exception as e:
            print(f"❌ OpenCLIP ViT-L/14 模型加载失败: {e}")
            print("请确保已安装 open_clip_torch 包并确保网络连接正常")
            raise
    
    def _load_dinov2_vit_s(self):
        """
        加载DINOv2 ViT-S模型
        """
        try:
            import torch
            import torchvision.models as models
            
            # 尝试加载DINOv2 ViT-S模型
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
