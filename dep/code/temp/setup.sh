#!/bin/bash
# setup.sh - PyTorch 3.10 环境安装脚本

echo "开始安装 PyTorch 3.10 环境..."

# 1. 检查 Python 版本
echo "1. 检查 Python 版本..."
python3 --version

# 2. 创建虚拟环境
echo -e "\n2. 创建虚拟环境..."
python3 -m venv pytorch-env
if [ $? -eq 0 ]; then
    echo "✓ 虚拟环境创建成功"
else
    echo "✗ 虚拟环境创建失败，尝试安装 venv"
    python3 -m ensurepip
    python3 -m pip install --upgrade pip
    python3 -m pip install virtualenv
    python3 -m venv pytorch-env
fi

# 3. 激活虚拟环境
echo -e "\n3. 激活虚拟环境..."
source pytorch-env/bin/activate
echo "当前环境: $(which python)"

# 4. 升级 pip
echo -e "\n4. 升级 pip..."
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 安装 PyTorch
echo -e "\n5. 安装 PyTorch 和 torchvision..."
pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 安装其他依赖
echo -e "\n6. 安装其他依赖包..."
pip install numpy pandas matplotlib jupyter pillow scikit-learn scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple

# 7. 验证安装
echo -e "\n7. 验证安装..."
python -c "
import sys
import torch
import torchvision
print('='*50)
print('安装验证报告')
print('='*50)
print(f'Python 版本: {sys.version.split()[0]}')
print(f'PyTorch 版本: {torch.__version__}')
print(f'Torchvision 版本: {torchvision.__version__}')
print(f'MPS 支持: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
print('='*50)
"

echo -e "\n安装完成！"
echo "使用以下命令激活环境:"
echo "source pytorch-env/bin/activate"