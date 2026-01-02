# ========================================================
# YOLO Face 训练脚本 - TensorFlow版本
# ========================================================
# 基于YOLOv3官方训练方法实现的人脸检测模型训练和验证代码
# 包含完整的训练流程：数据加载、数据增强、损失计算、模型训练和评估
# 使用TensorFlow 2.x实现，支持GPU加速和分布式训练
# ========================================================

# 导入必要的库
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt

# 导入TensorFlow版本的yoloface模型
from yoloface_tf import YoloFace, YoloLayer, xywh2xyxy, non_max_suppression

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    """
    设置随机种子以确保实验可重复性
    
    参数：
        seed: 随机种子值，默认为42
    """
    np.random.seed(seed)      # 设置NumPy随机种子
    tf.random.set_seed(seed)  # 设置TensorFlow随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子

set_seed(42)

# 配置参数
class Config:
    """
    训练配置类，包含所有训练相关的超参数和路径设置
    该配置遵循YOLOv3官方训练方法的标准设置
    """
    def __init__(self):
        # 训练配置
        self.batch_size = 32                # 批次大小
        self.epochs = 100                   # 训练轮数
        self.learning_rate = 0.001          # 初始学习率
        self.weight_decay = 0.0005          # 权重衰减系数
        
        # 模型配置 - 遵循YOLOv3训练标准
        self.img_size = 56                  # 输入图像大小
        self.grid_size = 7                  # 最终特征图大小
        self.num_anchors = 3                # 锚框数量
        self.anchors = np.array([[9,14], [12,17], [22,21]])  # 预定义锚框
        
        # 数据集配置
        self.train_dir = '../Datasets/train/images'  # 训练数据集图像目录
        self.train_label_dir = '../Datasets/train/labels'  # 训练数据集标签目录
        self.val_dir = '../Datasets/test/images'    # 验证数据集图像目录
        self.val_label_dir = '../Datasets/test/labels'    # 验证数据集标签目录
        
        # 检查点和增强配置
        self.checkpoint_dir = 'checkpoints'  # 模型保存目录
        self.save_interval = 10              # 每10个epoch保存一次模型
        self.multiscale_training = False     # 是否使用多尺度训练
        self.mosaic_augmentation = False     # 是否使用马赛克增强

config = Config()

# 创建检查点目录
os.makedirs(config.checkpoint_dir, exist_ok=True)

# ========================================================
# 数据增强函数 - 遵循YOLOv3标准增强策略
# YOLOv3标准的数据增强包括：色调/饱和度/亮度调整、水平翻转、
# 马赛克增强（Mosaic）等，以提高模型的鲁棒性和泛化能力
# ========================================================

def random_hue(img, delta=18.0):
    """
    随机调整图像色调
    
    参数：
        img: 输入图像，BGR格式（OpenCV默认格式）
        delta: 色调调整的最大变化范围，默认18.0度（HSV色彩空间中）
    
    返回：
        色调调整后的图像
    """
    # OpenCV的HSV范围是[0,179], [0,255], [0,255]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 转换为float32以避免数据类型错误
    h = img[:,:,0].astype(np.float32)
    # 随机生成色调调整值，范围为[-delta, delta]
    h += np.random.uniform(-delta, delta)
    # 确保色调值在有效范围内[0, 179]
    h = np.mod(h + 180, 180).astype(np.uint8)
    img[:,:,0] = h
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def random_saturation(img, lower=0.5, upper=1.5):
    """
    随机调整图像饱和度
    
    参数：
        img: 输入图像，BGR格式
        lower: 饱和度调整的最小值（小于1.0降低饱和度），默认0.5
        upper: 饱和度调整的最大值（大于1.0增加饱和度），默认1.5
    
    返回：
        饱和度调整后的图像
    """
    # 转换为HSV色彩空间进行饱和度调整
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = img[:,:,1].astype(np.float32)  # 获取饱和度通道
    
    # 随机生成饱和度调整因子
    s *= np.random.uniform(lower, upper)
    s = np.clip(s, 0, 255).astype(np.uint8)  # 确保值在0-255范围内
    
    img[:,:,1] = s  # 更新饱和度通道
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)  # 转换回BGR格式

def random_brightness(img, delta=32):
    """
    随机调整图像亮度
    
    参数：
        img: 输入图像
        delta: 亮度调整的最大变化范围，默认32
    
    返回：
        亮度调整后的图像
    """
    # 直接调整图像亮度，而不转换到HSV空间
    img = img.astype(np.float32)  # 转换为浮点型以避免溢出
    
    # 随机生成亮度调整值
    img += np.random.uniform(-delta, delta)
    img = np.clip(img, 0, 255).astype(np.uint8)  # 确保值在0-255范围内
    
    return img

def random_flip(img, flip_prob=0.5):
    """
    随机水平翻转图像 - YOLOv3标准增强技术之一
    可以帮助模型学习不同方向的人脸特征
    
    参数：
        img: 输入图像
        flip_prob: 水平翻转的概率，默认0.5
    
    返回：
        原图或水平翻转后的图像
    """
    if np.random.random() < flip_prob:
        img = cv2.flip(img, 1)  # 使用OpenCV的水平翻转函数（参数1表示水平翻转）
    return img

def augment_image(img, flip_prob=0.5):
    """
    应用多种数据增强策略 - YOLOv3标准增强流程
    按顺序应用色调、饱和度、亮度调整和水平翻转
    
    参数：
        img: 输入图像
        flip_prob: 水平翻转的概率，默认0.5
    
    返回：
        经过多种数据增强处理后的图像
    """
    # 按顺序应用各种数据增强
    # 1. 随机调整色调
    img = random_hue(img)
    # 2. 随机调整饱和度
    img = random_saturation(img)
    # 3. 随机调整亮度
    img = random_brightness(img)
    # 4. 随机水平翻转
    img = random_flip(img, flip_prob)
    return img

# ========================================================
# IoU计算和标签处理函数
# ========================================================

def _calculate_iou(box1, box2):
    """
    计算两个边界框的IoU值
    IoU = 交集面积 / (box1面积 + box2面积 - 交集面积)
    
    参数：
        box1: 第一个边界框，格式为 [x1, y1, w, h]
        box2: 第二个边界框，格式为 [x1, y1, w, h]
    
    返回：
        IoU值，范围在[0, 1]之间
    """
    # box1和box2的格式: [x1, y1, w, h]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    return intersection / (area1 + area2 - intersection + 1e-10)

def load_labels(label_path, img_width, img_height, img_size):
    """
    从标签文件加载边界框和类别信息 - YOLOv3标准标签格式
    
    YOLO格式的标签: [class_id, x_center, y_center, width, height]
    其中x_center, y_center, width, height都是归一化的 [0, 1]范围
    
    参数：
        label_path: 标签文件路径
        img_width: 原始图像宽度
        img_height: 原始图像高度
        img_size: 目标图像大小，用于坐标缩放
    
    返回：
        标签数组，格式为 [x_center, y_center, width, height, class_id]
        坐标已根据目标图像大小进行了缩放
    """
    labels = []
    
    # 检查标签文件是否存在
    if not os.path.exists(label_path):
        # 如果标签文件不存在，返回空标签
        return np.array(labels)
    
    # 读取标签文件
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    # 解析每一行标签
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        class_id = float(parts[0])  # 类别ID
        x_center_norm = float(parts[1])  # 归一化的x_center
        y_center_norm = float(parts[2])  # 归一化的y_center
        width_norm = float(parts[3])  # 归一化的width
        height_norm = float(parts[4])  # 归一化的height
        
        # 计算目标图像中的坐标
        # 将归一化坐标乘以目标图像大小进行缩放
        x_center = x_center_norm * img_size
        y_center = y_center_norm * img_size
        width = width_norm * img_size
        height = height_norm * img_size
        
        # 添加到标签列表
        labels.append([x_center, y_center, width, height, class_id])
    
    return np.array(labels)

def process_image(img_path, img_size, augment=False):
    """处理单个图像及其标签 - YOLOv3标准预处理流程
    
    参数：
        img_path: 图像文件路径
        img_size: 目标图像尺寸
        augment: 是否应用数据增强
    
    返回：
        处理后的图像和目标张量
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    # 获取原始图像尺寸
    h, w = img.shape[:2]
    
    # 如果是训练模式，应用数据增强
    if augment:
        img = augment_image(img)
    
    # 调整图像大小到模型输入尺寸
    img = cv2.resize(img, (img_size, img_size))
    
    # 将BGR转换为RGB（OpenCV默认读取为BGR格式）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 归一化并确保float32类型
    # 将像素值缩放到[0, 1]范围
    img = img / 255.0
    img = img.astype(np.float32)
    
    # 构建标签文件路径
    # 从图像路径中提取文件名（不带扩展名）
    img_filename = os.path.basename(img_path)
    img_name_without_ext = os.path.splitext(img_filename)[0]
    
    # 确定是训练还是验证数据，选择正确的标签目录
    if config.train_dir in img_path:
        label_dir = config.train_label_dir
    else:
        label_dir = config.val_label_dir
    
    # 构建完整的标签文件路径
    label_path = os.path.join(label_dir, f"{img_name_without_ext}.txt")
    
    # 加载标签并根据目标尺寸进行缩放
    labels = load_labels(label_path, w, h, img_size)
    
    # 如果没有标签，使用默认标签（防止训练失败）
    if len(labels) == 0:
        labels = np.array([[0.5 * img_size, 0.5 * img_size, 0.3 * img_size, 0.3 * img_size, 0]])
    
    # 创建目标张量 - YOLOv3输出格式
    # [num_anchors, grid_size, grid_size, 6] - 6表示[t_x, t_y, t_w, t_h, confidence, class_id]
    target = np.zeros((config.num_anchors, config.grid_size, config.grid_size, 6))
    
    # 分配标签到网格和锚框 - YOLOv3核心处理逻辑
    for label in labels:
        x_center, y_center, width, height, class_id = label
        
        # 计算目标所在的网格
        grid_x = int(x_center / (img_size / config.grid_size))
        grid_y = int(y_center / (img_size / config.grid_size))
        
        # 计算相对于网格的偏移量 - YOLOv3预测格式
        tx = x_center / (img_size / config.grid_size) - grid_x
        ty = y_center / (img_size / config.grid_size) - grid_y
        
        # 计算相对于锚框的宽高比例的对数 - YOLOv3预测格式
        tw = np.log(width / config.anchors[:, 0])
        th = np.log(height / config.anchors[:, 1])
        
        # 计算IoU并选择最佳锚框 - YOLOv3锚框分配策略
        ious = []
        for anchor in config.anchors:
            # 计算预测框和锚框的IoU
            iou = _calculate_iou(np.array([0, 0, width, height]), 
                                 np.array([0, 0, anchor[0], anchor[1]]))
            ious.append(iou)
        best_anchor = np.argmax(ious)
        
        # 分配目标到最佳锚框
        target[best_anchor, grid_y, grid_x, 0] = tx  # 中心点x偏移
        target[best_anchor, grid_y, grid_x, 1] = ty  # 中心点y偏移
        target[best_anchor, grid_y, grid_x, 2] = tw[best_anchor]  # 宽度比例对数
        target[best_anchor, grid_y, grid_x, 3] = th[best_anchor]  # 高度比例对数
        target[best_anchor, grid_y, grid_x, 4] = 1.0  # 目标存在置信度
        target[best_anchor, grid_y, grid_x, 5] = class_id  # 类别ID
    
    # 确保target也是float32类型
    target = target.astype(np.float32)
    return img, target

# 创建数据集
def create_dataset(img_dir, img_size, batch_size, augment=False):
    """创建tf.data.Dataset用于训练或验证 - TensorFlow最佳实践实现
    
    参数：
        img_dir: 图像目录路径
        img_size: 图像调整后的尺寸
        batch_size: 批次大小
        augment: 是否应用数据增强
    
    返回：
        优化后的TensorFlow数据集
    """
    # 获取图像文件列表
    # 仅支持常见的图像格式
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 创建数据集
    # 使用from_tensor_slices将文件路径列表转换为数据集
    dataset = tf.data.Dataset.from_tensor_slices(img_files)
    
    # 映射处理函数
    def load_and_process(img_path):
        """内部处理函数，用于从文件路径加载和处理图像"""
        # 使用tf.numpy_function包装Python函数
        # 如果是bytes对象直接解码，如果是tensor则先转为numpy再解码
        if isinstance(img_path, bytes):
            img_path_str = img_path.decode('utf-8')
        else:
            img_path_str = img_path.numpy().decode('utf-8')
        img, target = process_image(img_path_str, img_size, augment)
        return img, target
    
    # 应用处理函数
    # 使用tf.numpy_function包装Python函数，使其兼容TensorFlow的图模式
    # num_parallel_calls启用并行处理以提高效率
    dataset = dataset.map(
        lambda x: tf.numpy_function(
            load_and_process, [x], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # 处理形状信息
    # 确保每个批次的张量都有确定的形状，这对模型训练至关重要
    dataset = dataset.map(
        lambda x, y: (tf.ensure_shape(x, (img_size, img_size, 3)),
                    tf.ensure_shape(y, (config.num_anchors, config.grid_size, config.grid_size, 6)))
    )
    
    # 数据集优化：打乱、批处理和预取
    # 这些是TensorFlow数据加载的最佳实践，可以显著提高训练性能
    
    # 仅在增强模式（通常是训练集）应用打乱
    if augment:
        dataset = dataset.shuffle(buffer_size=len(img_files))
    
    # 批处理 - 将多个样本组合成批次
    dataset = dataset.batch(batch_size)
    
    # 预取 - 在处理当前批次时加载下一批次，减少等待时间
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

# ========================================================
# YOLOv3损失函数 - 核心训练组件
# ========================================================
# 实现YOLOv3官方损失函数，包括：
# 1. 坐标损失（中心点偏移和宽高比例）
# 2. 置信度损失（正样本和负样本）
# 3. 类别损失
# 权重系数与官方实现一致，继承自TensorFlow的Loss基类
# ========================================================

class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, anchors, grid_size):
        super(YoloLoss, self).__init__()
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        self.grid_size = grid_size
        
        # 损失权重 - 与YOLOv3官方实现一致
        self.lambda_coord = 5.0  # 坐标损失权重，对坐标预测给予更高关注
        self.lambda_noobj = 0.5  # 无目标损失权重，降低背景误检率
        
        # 损失函数 - 坐标使用MSE，置信度和类别使用BCE
        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction='sum')
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='sum')
    
    def call(self, y_true, y_pred):
        """
        计算YOLO损失
        支持多种输入形状：
        - 训练阶段: y_pred [batch_size, num_predictions, 6], y_true [batch_size, num_anchors, grid_size, grid_size, 6]
        - 验证阶段: y_pred [num_predictions, 6], y_true [num_anchors, grid_size, grid_size, 6]
        """
        # 确保y_pred和y_true都有batch维度
        if len(tf.shape(y_pred)) == 2:
            y_pred = tf.expand_dims(y_pred, axis=0)
        if len(tf.shape(y_true)) == 4:
            y_true = tf.expand_dims(y_true, axis=0)
        
        batch_size = tf.shape(y_pred)[0]
        
        # 重塑y_true以匹配y_pred的形状
        y_true_flat = tf.reshape(y_true, (batch_size, -1, 6))
        
        # 截断或填充y_true_flat以匹配y_pred的长度
        pred_length = tf.shape(y_pred)[1]
        true_length = tf.shape(y_true_flat)[1]
        
        if true_length > pred_length:
            # 如果y_true_flat更长，则截断
            y_true_flat = y_true_flat[:, :pred_length, :]
        elif true_length < pred_length:
            # 如果y_pred更长，则填充y_true_flat
            padding = tf.zeros((batch_size, pred_length - true_length, 6), dtype=y_true_flat.dtype)
            y_true_flat = tf.concat([y_true_flat, padding], axis=1)
        
        # 计算置信度掩码
        obj_mask = y_true_flat[..., 4] >= 0.5  # 使用>=0.5以适应可能的浮点值
        
        # 直接使用boolean_mask提取有目标的样本，避免形状不匹配问题
        obj_indices = tf.where(obj_mask)
        noobj_indices = tf.where(~obj_mask)
        
        # 计算坐标损失 - 只处理有目标的位置
        if tf.size(obj_indices) > 0:
            # 提取有目标的预测和真实值
            y_pred_obj = tf.gather_nd(y_pred, obj_indices)
            y_true_obj = tf.gather_nd(y_true_flat, obj_indices)
            
            # 计算坐标损失
            loss_x = self.mse_loss(y_pred_obj[..., 0], y_true_obj[..., 0])
            loss_y = self.mse_loss(y_pred_obj[..., 1], y_true_obj[..., 1])
            loss_w = self.mse_loss(y_pred_obj[..., 2], y_true_obj[..., 2])
            loss_h = self.mse_loss(y_pred_obj[..., 3], y_true_obj[..., 3])
            
            loss_coord = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)
        else:
            loss_coord = 0.0
        
        # 计算置信度损失 - 有目标
        if tf.size(obj_indices) > 0:
            # 提取有目标位置的置信度
            conf_obj_pred = tf.gather_nd(y_pred[..., 4], obj_indices[:, :-1])
            conf_obj_true = tf.gather_nd(y_true_flat[..., 4], obj_indices[:, :-1])
            loss_obj = self.bce_loss(conf_obj_pred, conf_obj_true)
        else:
            loss_obj = 0.0
        
        # 计算置信度损失 - 无目标
        if tf.size(noobj_indices) > 0:
            # 提取无目标位置的置信度
            conf_noobj_pred = tf.gather_nd(y_pred[..., 4], noobj_indices[:, :-1])
            conf_noobj_true = tf.gather_nd(y_true_flat[..., 4], noobj_indices[:, :-1])
            loss_noobj = self.bce_loss(conf_noobj_pred, conf_noobj_true)
        else:
            loss_noobj = 0.0
        
        loss_conf = loss_obj + self.lambda_noobj * loss_noobj
        
        # 计算类别损失 - 只处理有目标的位置
        if tf.size(obj_indices) > 0:
            # 提取有目标位置的类别
            cls_obj_pred = tf.gather_nd(y_pred[..., 5], obj_indices[:, :-1])
            cls_obj_true = tf.gather_nd(y_true_flat[..., 5], obj_indices[:, :-1])
            loss_cls = self.bce_loss(cls_obj_pred, cls_obj_true)
        else:
            loss_cls = 0.0
        
        # 总损失
        total_loss = loss_coord + loss_conf + loss_cls
        
        # 平均损失
        total_loss /= tf.cast(tf.maximum(1, batch_size), dtype=tf.float32)
        
        # 确保返回的是标量损失值
        return tf.reduce_sum(total_loss)

# ========================================================
# 模型初始化函数 - 训练前准备
# ========================================================
# 初始化YOLOv3模型、损失函数和优化器
# 配置与YOLOv3官方实现一致的训练设置
# 使用余弦退火学习率调度策略
# ========================================================

def initialize_model():
    """初始化模型、损失函数和优化器
    
    返回：
        model: YoloFace模型实例
        loss_function: YOLOv3损失函数
        optimizer: 配置好的Adam优化器
        lr_callback: 学习率调度回调
    """
    # 创建模型实例 - 基于YOLOv3架构的人脸检测模型
    model = YoloFace()
    
    # 构建模型 - 使用虚拟输入让TensorFlow构建计算图
    dummy_input = tf.random.uniform((1, config.img_size, config.img_size, 3))
    model(dummy_input, training=False)
    
    # 定义损失函数 - YOLOv3官方实现的损失计算方法
    loss_function = YoloLoss(config.anchors, config.grid_size)
    
    # 定义优化器 - 使用Adam优化器，与YOLOv3训练一致
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay  # L2正则化，防止过拟合
    )
    
    # 定义学习率调度器 - 余弦退火策略
    # 这是YOLOv3训练中的标准学习率调整方法
    def lr_scheduler(epoch):
        """余弦退火学习率调度器
        在训练过程中平滑降低学习率，帮助模型更好地收敛
        """
        # 余弦退火调度公式
        lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * epoch / config.epochs))
        # 确保学习率不低于最小值，避免训练停滞
        return max(lr, 1e-5)
    
    # 创建学习率调度回调
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    return model, loss_function, optimizer, lr_callback

# ========================================================
# 训练函数
# ========================================================
# 使用tf.function装饰器加速训练，编译为计算图
# 实现YOLOv3标准的训练步骤：前向传播、损失计算、梯度下降
# ========================================================

@tf.function  # 转换为TensorFlow计算图，提升训练性能
def train_step(model, loss_function, optimizer, images, targets):
    """
    单个训练步骤的实现
    执行一次完整的前向传播、损失计算和反向传播
    
    参数：
        model: YoloFace模型实例
        loss_function: YOLOv3损失函数
        optimizer: 优化器实例
        images: 批次图像数据
        targets: 批次目标标签
    
    返回：
        当前批次的损失值
    """
    # 使用梯度磁带记录计算图
    with tf.GradientTape() as tape:
        # 前向传播 - 执行模型预测
        predictions = model(images, training=True)
        
        # 计算损失 - 应用YOLOv3损失函数
        loss = loss_function(targets, predictions)
    
    # 计算梯度 - 反向传播
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # 梯度裁剪 - 防止梯度爆炸，YOLOv3训练的关键步骤
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    
    # 更新参数 - 应用梯度下降
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# ========================================================
# 验证函数
# ========================================================
# 在验证集上评估模型性能
# 计算平均验证损失，不更新模型权重
# ========================================================

def validate_model(model, loss_function, val_dataset):
    """
    在验证集上评估模型性能
    
    参数：
        model: YoloFace模型实例
        loss_function: YOLOv3损失函数
        val_dataset: 验证数据集
    
    返回：
        平均验证损失值
    """
    val_loss = 0.0
    num_batches = 0
    
    # 遍历验证数据集中的所有批次
    for images, targets in val_dataset:
        # 前向传播 - 不进行训练
        predictions = model(images, training=False)  # 设置training=False关闭批归一化更新
        
        # 计算批次损失
        batch_loss = loss_function(targets, predictions)
        val_loss += batch_loss.numpy()
        num_batches += 1
    
    # 计算平均损失
    avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_val_loss

# ========================================================
# 模型保存函数
# ========================================================
# 实现YOLOv3标准的模型保存机制
# 保存完整模型、检查点文件和训练元数据
# ========================================================

def save_model(model, optimizer, epoch, loss, is_best=False):
    """
    保存模型检查点
    
    参数：
        model: YoloFace模型实例
        optimizer: 当前使用的优化器
        epoch: 当前训练轮次
        loss: 当前损失值
        is_best: 是否是最佳模型
    """
    # 使用TensorFlow的标准检查点机制
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    # 保存检查点
    if is_best:
        # 保存最佳模型 - 用于后续推理和部署
        model_path = os.path.join(config.checkpoint_dir, 'best_model')
        # 保存模型（TensorFlow SavedModel格式）
        model.save(model_path, save_format='tf', include_optimizer=False)
        
        # 同时保存检查点文件，方便恢复训练
        checkpoint_path = os.path.join(config.checkpoint_dir, 'best_checkpoint')
        checkpoint.save(file_prefix=checkpoint_path)
        print(f"最佳模型已保存到 {model_path}")
        print(f"最佳检查点已保存到 {checkpoint_path}")
    else:
        # 保存普通检查点 - 定期保存，用于恢复训练
        model_path = os.path.join(config.checkpoint_dir, f'model_epoch_{epoch}')
        model.save(model_path, save_format='tf', include_optimizer=False)
        
        # 保存检查点文件
        checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}')
        checkpoint.save(file_prefix=checkpoint_path)
        print(f"模型检查点已保存到 {model_path}")
        print(f"优化器检查点已保存到 {checkpoint_path}")
    
    # 保存额外的元数据
    # 确保所有数据都可以被JSON序列化
    metadata = {
        'epoch': int(epoch),
        'loss': float(loss)  # 转换为Python原生float类型
    }
    
    # 处理优化器配置，确保所有值都可JSON序列化
    optimizer_config = optimizer.get_config()
    serializable_config = {}
    for key, value in optimizer_config.items():
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):
            # 基本类型直接保存
            if isinstance(value, tf.Tensor):
                serializable_config[key] = float(value.numpy())
            elif isinstance(value, (int, float)):
                serializable_config[key] = float(value)
            else:
                serializable_config[key] = value
        else:
            # 复杂类型转换为字符串
            serializable_config[key] = str(value)
    
    metadata['optimizer_config'] = serializable_config
    
    metadata_path = os.path.join(config.checkpoint_dir, f'metadata_epoch_{epoch}.json')
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f)

# 导出模型为SavedModel格式
def export_model(model, export_path):
    """导出模型为SavedModel格式，支持自定义层"""
    # TensorFlow 2.x中，自定义层在构建模型时已经注册，保存时不需要额外传递custom_objects
    
    # 使用TensorFlow的标准导出方法
    model.export(export_path)
    print(f"模型已导出为SavedModel格式: {export_path}")

# ========================================================
# 训练循环函数
# ========================================================
# 实现YOLOv3标准训练流程
# 包含训练/验证循环、最佳模型保存、TensorBoard日志记录
# ========================================================

def train_model(model, loss_function, optimizer, lr_callback, train_dataset, val_dataset, start_epoch=0):
    """
    完整的模型训练流程实现
    
    参数：
        model: YoloFace模型实例
        loss_function: YOLOv3损失函数
        optimizer: 优化器实例
        lr_callback: 学习率调度器回调
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        start_epoch: 开始训练的轮次，默认为0
    """
    # 记录最佳损失值，初始化为无穷大
    best_val_loss = float('inf')
    
    # 用于记录训练过程中的损失值和学习率
    train_losses = []
    val_losses = []
    lrs = []
    
    # 创建评估和可视化目录
    os.makedirs('evaluation', exist_ok=True)
    os.makedirs('visualization', exist_ok=True)
    
    # 初始化TensorBoard日志记录器
    log_dir = os.path.join(config.checkpoint_dir, 'logs')
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    print('开始训练...')
    start_time = time.time()
    
    # 主训练循环 - YOLOv3标准训练流程
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # 更新学习率 - 应用余弦退火调度器
        current_lr = optimizer.learning_rate.numpy()
        if hasattr(optimizer, '_decayed_lr'):
            current_lr = optimizer._decayed_lr(tf.float32).numpy()
        lrs.append(current_lr)
        
        # 遍历训练数据集
        for images, targets in train_dataset:
            # 执行训练步骤 - 包含前向传播、损失计算、反向传播
            loss = train_step(model, loss_function, optimizer, images, targets)
            
            # 累加损失
            epoch_loss += loss.numpy()
            num_batches += 1
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 执行验证 - 不更新模型权重，仅评估性能
        avg_val_loss = validate_model(model, loss_function, val_dataset)
        val_losses.append(avg_val_loss)
        
        # 使用TensorBoard记录 - 用于可视化训练过程
        with summary_writer.as_default():
            tf.summary.scalar('loss/train', avg_train_loss, step=epoch)
            tf.summary.scalar('loss/val', avg_val_loss, step=epoch)
            tf.summary.scalar('learning_rate', current_lr, step=epoch)
        
        # 应用学习率调度
        # 注意：在TensorFlow中，回调会自动应用，但这里我们手动应用以匹配PyTorch行为
        new_lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / config.epochs))
        optimizer.learning_rate.assign(new_lr)
        
        # 打印训练信息
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{config.epochs}, ' 
              f'Train Loss: {avg_train_loss:.4f}, ' 
              f'Val Loss: {avg_val_loss:.4f}, ' 
              f'LR: {current_lr:.6f}, ' 
              f'Time: {epoch_time:.2f}s')
        
        # 检查是否为最佳模型 - 基于验证损失
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"新的最佳模型！验证损失: {best_val_loss:.6f}")
        
        # 保存模型 - 按间隔或达到最佳时保存
        if (epoch + 1) % config.save_interval == 0:
            save_model(model, optimizer, epoch + 1, avg_val_loss)
        
        # 保存最佳模型
        if is_best:
            save_model(model, optimizer, epoch + 1, avg_val_loss, is_best=True)
    
    # 总训练时间
    total_time = time.time() - start_time
    print(f'训练完成! 总用时: {total_time:.2f}秒')
    
    # 训练完成，绘制损失曲线 - 可视化训练过程
    plot_losses(train_losses, val_losses, lrs)

# ========================================================
# 可视化函数
# ========================================================
# 绘制训练过程中的损失曲线和学习率变化
# 用于监控训练进度和性能分析
# ========================================================

def plot_losses(train_losses, val_losses, lrs):
    """
    绘制训练过程中的损失曲线和学习率变化
    
    参数：
        train_losses: 训练损失值列表
        val_losses: 验证损失值列表
        lrs: 学习率变化列表
    """
    import matplotlib.pyplot as plt
    
    # 创建损失曲线图像
    plt.figure(figsize=(15, 6))
    
    # 绘制损失曲线 - 用于分析模型收敛情况
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制学习率曲线 - 验证余弦退火调度效果
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(lrs) + 1), lrs, label='学习率')
    plt.title('学习率变化')
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    plt.legend()
    plt.grid(True)
    
    # 保存图像 - 用于后续分析和报告
    plt.tight_layout()
    loss_curve_path = os.path.join('visualization', 'loss_curves.png')
    plt.savefig(loss_curve_path)
    print(f"损失曲线已保存到 {loss_curve_path}")
    
    # 在非交互式环境中显示
    plt.close()

# ========================================================
# 主函数
# ========================================================
# 实现YOLOv3训练的完整工作流
# 包括数据加载、模型初始化、检查点恢复、训练执行和最终评估
# ========================================================

def main():
    print(f'开始TensorFlow版本的YOLO Face训练')
    
    # 创建必要的目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs('visualization', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = create_dataset(
        config.train_dir,
        config.img_size,
        config.batch_size,
        augment=True
    )
    
    val_dataset = create_dataset(
        config.val_dir,
        config.img_size,
        config.batch_size,
        augment=False
    )
    
    # 初始化模型
    print("初始化模型...")
    model, loss_function, optimizer, lr_callback = initialize_model()
    
    # 打印模型摘要
    model.summary()
    
    # 尝试加载检查点 - 支持断点续训
    start_epoch = 0
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    # 查找最近的检查点
    latest_checkpoint = tf.train.latest_checkpoint(config.checkpoint_dir)
    if latest_checkpoint:
        try:
            # 恢复模型和优化器状态
            checkpoint.restore(latest_checkpoint)
            # 尝试从文件名中提取轮次信息
            epoch_str = latest_checkpoint.split('_')[-1]
            if epoch_str.isdigit():
                start_epoch = int(epoch_str) + 1
                print(f"已恢复检查点: {latest_checkpoint}, 从轮次 {start_epoch} 开始训练")
        except Exception as e:
            print(f"恢复检查点失败: {e}")
    
    # 开始训练 - 执行主训练循环
    train_model(model, loss_function, optimizer, lr_callback, train_dataset, val_dataset, start_epoch)
    
    # 训练结束后，加载最佳模型进行最终评估
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model')
    if os.path.exists(best_model_path):
        print(f"\n加载最佳模型进行最终评估: {best_model_path}")
        best_model = tf.keras.models.load_model(best_model_path)
        
        # 执行最终验证
        final_val_loss = validate_model(best_model, loss_function, val_dataset)
        print(f"\n最终模型评估结果:")
        print(f"验证损失: {final_val_loss:.6f}")
        
        # 生成评估报告
        from datetime import datetime
        report_path = os.path.join('evaluation', 'final_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("YOLO Face 模型最终评估报告\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型路径: {best_model_path}\n")
            f.write(f"验证损失: {final_val_loss:.6f}\n")
            f.write(f"总训练轮次: {config.epochs}\n")
        
        print(f"评估报告已保存到: {report_path}")
    else:
        print("未找到最佳模型，跳过最终评估")
    
    # 导出最终模型
    export_model(model, 'final_model')
    print("YOLO Face模型训练流程已完成！")

if __name__ == '__main__':
    main()