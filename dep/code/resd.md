searchpic/
├── app/                  # 应用核心代码
│   ├── api/              # API路由和端点
│   │   ├── __init__.py
│   │   ├── routes.py     # 路由定义
│   │   └── health.py     # 健康检查端点
│   ├── services/         # 业务逻辑服务
│   │   ├── __init__.py
│   │   ├── feature.py    # 特征提取服务
│   │   └── vector_db.py  # 向量数据库服务
│   ├── models/           # 数据模型和DTO
│   │   ├── __init__.py
│   │   └── schemas.py    # 请求/响应模型
│   ├── utils/            # 工具函数
│   │   ├── __init__.py
│   │   └── helpers.py    # 辅助函数
│   ├── core/             # 核心配置和初始化
│   │   ├── __init__.py
│   │   ├── config.py     # 配置管理
│   │   └── startup.py    # 服务启动逻辑
│   └── __init__.py
├── config/               # 配置文件
│   ├── __init__.py
│   └── settings.py       # 全局配置
├── data/                 # 数据目录
│   └── vector_db/        # 向量数据库存储
├── models/               # 模型文件
│   └── cache/            # 模型缓存
├── static/               # 静态文件
│   └── img/              # 图片存储
├── templates/            # HTML模板
├── tests/                # 测试代码
│   ├── __init__.py
│   ├── test_api.py
│   └── test_services.py
├── deploy/               # 部署相关文件
│   ├── docker/           # Docker配置
│   │   ├── Dockerfile
│   │   └── .dockerignore
│   └── k8s/              # Kubernetes配置（可选）
├── scripts/              # 脚本文件
│   ├── init_db.py        # 数据库初始化
│   └── evaluate_model.py # 模型评估
├── requirements.txt      # 依赖管理
├── docker-compose.yml    # Docker Compose配置
├── .env.example          # 环境变量示例
├── README.md             # 项目文档
└── main.py               # 应用入口