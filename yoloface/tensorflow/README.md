# YOLO Face - TensorFlow 实现

本目录包含基于 TensorFlow 的 YOLO Face 人脸检测模型的完整实现，包括训练、验证和测试代码。该实现遵循 YOLOv3 架构，并针对人脸检测任务进行了优化。

## 目录结构

```
tensorflow/
├── yoloface_tf.py        # YOLO Face 网络结构实现
├── train_tf.py           # 训练和验证脚本
├── yoloface_test.py      # 测试和推理脚本
├── README.md             # 使用说明（本文档）
├── checkpoints/          # 模型检查点保存目录
│   └── logs/             # TensorBoard 日志
├── evaluation/           # 评估结果目录
└── visualization/        # 可视化结果目录
```

## 依赖项

要运行本项目的代码，您需要安装以下依赖项：

```bash
# 基本依赖
pip install tensorflow==2.10.0
pip install opencv-python==4.6.0
pip install numpy==1.23.4
pip install matplotlib==3.6.2
pip install scipy==1.9.3

# 可选依赖（用于数据增强和可视化）
pip install albumentations==1.2.1
pip install tqdm==4.64.1
```

## 快速开始

### 1. 准备数据集

确保您的数据集按以下格式组织：

```
dataset/
├── train/
│   ├── images/           # 训练图像
│   └── labels/           # 训练标签（YOLO格式）
└── val/
    ├── images/           # 验证图像
    └── labels/           # 验证标签（YOLO格式）
```

标签格式：YOLO 格式，每行表示一个人脸，格式为 `0 x_center y_center width height`，其中坐标是相对于图像宽高的归一化值。

### 2. 训练模型

修改 `train_tf.py` 中的 `Config` 类以配置训练参数，然后运行训练脚本：

```bash
python train_tf.py
```

主要配置参数说明：

- `seed`: 随机种子，用于复现结果
- `epochs`: 训练轮次
- `batch_size`: 批次大小
- `input_shape`: 模型输入尺寸 [width, height]
- `train_images_dir`/`val_images_dir`: 训练/验证图像目录
- `train_labels_dir`/`val_labels_dir`: 训练/验证标签目录
- `checkpoint_dir`: 检查点保存目录

### 3. 测试模型

使用 `yoloface_test.py` 脚本测试训练好的模型：

#### 单图像测试

```bash
python yoloface_test.py --model checkpoints/best_model --input path/to/image.jpg --mode image
```

#### 视频测试

```bash
python yoloface_test.py --model checkpoints/best_model --input path/to/video.mp4 --output output_video.mp4 --mode video --display
```

#### 批量测试

```bash
python yoloface_test.py --model checkpoints/best_model --input path/to/images_dir --output detection_results --mode batch
```

## 模型说明

### YOLO Face 网络结构

YOLO Face 模型基于 YOLOv3 架构，针对人脸检测任务进行了优化：

- **主干网络**: 使用深度可分离卷积降低计算量
- **特征融合**: 采用特征金字塔网络(FPN)融合多尺度特征
- **检测头**: 针对人脸检测优化的输出层

### 训练特性

- **数据增强**: 实现了YOLOv3标准数据增强策略（随机翻转、亮度/对比度调整等）
- **学习率调度**: 使用余弦退火学习率调度器
- **模型检查点**: 自动保存最佳模型和定期检查点
- **TensorBoard**: 实时监控训练进度和性能指标
- **多尺度训练**: 支持多尺度输入训练

## 使用示例

### 自定义训练

```python
# 导入必要的模块
from train_tf import Config, set_seed, create_dataset, initialize_model, train_model
import tensorflow as tf
import os

# 自定义配置
config = Config()
config.epochs = 100
config.batch_size = 16
config.input_shape = [416, 416]

# 设置随机种子
set_seed(config.seed)

# 创建数据集
train_dataset = create_dataset(
    config.train_images_dir, 
    config.train_labels_dir, 
    config.batch_size, 
    config.input_shape, 
    train=True
)
val_dataset = create_dataset(
    config.val_images_dir, 
    config.val_labels_dir, 
    config.batch_size, 
    config.input_shape, 
    train=False
)

# 初始化模型
model, loss_function, optimizer, lr_callback = initialize_model()

# 开始训练
train_model(model, loss_function, optimizer, lr_callback, train_dataset, val_dataset)
```

### 自定义推理

```python
# 导入必要的模块
from yoloface_test import YoloFaceDetector
import cv2

# 创建检测器实例
detector = YoloFaceDetector(
    model_path="checkpoints/best_model",
    confidence_threshold=0.5,
    iou_threshold=0.4
)

# 读取图像
image = cv2.imread("test.jpg")

# 执行检测
detections = detector.detect(image)

# 可视化结果
viz_image = detector.visualize_detections(image, detections, "result.jpg")
```

## 评估结果

训练完成后，评估结果将保存在以下位置：

- **损失曲线**: `visualization/loss_curves.png`
- **TensorBoard 日志**: `checkpoints/logs/`
- **最终评估报告**: `evaluation/final_evaluation_report.txt`
- **最佳模型**: `checkpoints/best_model/`

## 常见问题

### 1. 如何调整模型性能？

- 调整 `input_shape` 可以影响精度和速度平衡
- 增加训练轮次或调整学习率可能提高精度
- 尝试不同的 `batch_size` 和数据增强策略

### 2. 如何处理训练中断？

训练脚本支持自动恢复功能，中断后重新运行脚本会从最近的检查点继续训练。

### 3. 如何优化推理速度？

- 减小输入尺寸可以显著提高速度
- 使用量化技术（如INT8量化）
- 考虑导出为TensorRT模型以获得最佳性能

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件

## 联系方式

如有问题或建议，请联系项目维护者。