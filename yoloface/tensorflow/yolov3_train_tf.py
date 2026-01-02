import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np
import os
import math
import time
import random
import json
from yoloface_tf import YoloFace, YoloLayer, xywh2xyxy, non_max_suppression

# 设置随机种子，确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# 配置类 - YOLOv3风格的配置
class Config:
    def __init__(self):
        # 训练配置
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 0.0005
        
        # 模型配置
        self.img_size = 416
        self.grid_size = 13
        self.num_anchors = 3
        # YOLOv3风格的锚框
        self.anchors = np.array([[10, 13], [16, 30], [33, 23],
                                [30, 61], [62, 45], [59, 119],
                                [116, 90], [156, 198], [373, 326]])
        # 根据模型输出选择适当的锚框
        self.selected_anchors = self.anchors[:self.num_anchors]
        
        # 数据路径
        self.train_dir = '../data/images/train'
        self.val_dir = '../data/images/val'
        self.label_dir = '../data/labels'
        
        # 检查点和日志
        self.checkpoint_dir = 'checkpoints_yolov3_style'
        self.save_interval = 10
        
        # YOLOv3特有的配置
        self.multiscale_training = True  # 启用多尺度训练
        self.multiscale_min = 320
        self.multiscale_max = 608
        self.mosaic_augmentation = True  # 启用马赛克增强
        self.cosine_scheduler = True  # 余弦退火学习率调度
        self.warmup_epochs = 3  # 预热轮数

config = Config()

# 创建检查点目录
os.makedirs(config.checkpoint_dir, exist_ok=True)

# YOLOv3风格的数据增强函数
# 颜色增强
def random_hue(img, delta=18.0):
    # YOLOv3风格的色调调整
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0]
    h = h.astype(np.float32)
    delta_h = np.random.uniform(-delta, delta)
    h = (h + delta_h) % 180
    img_hsv[:, :, 0] = h.astype(np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_bgr

def random_saturation(img, lower=0.5, upper=1.5):
    # YOLOv3风格的饱和度调整
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = img_hsv[:, :, 1]
    s = s.astype(np.float32)
    scale = np.random.uniform(lower, upper)
    s = np.clip(s * scale, 0, 255)
    img_hsv[:, :, 1] = s.astype(np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_bgr

def random_brightness(img, delta=32):
    # YOLOv3风格的亮度调整
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = img_hsv[:, :, 2]
    v = v.astype(np.float32)
    delta_v = np.random.uniform(-delta, delta)
    v = np.clip(v + delta_v, 0, 255)
    img_hsv[:, :, 2] = v.astype(np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_bgr

# 空间增强
def random_flip(img, labels, flip_prob=0.5):
    # YOLOv3风格的水平翻转
    if random.random() < flip_prob:
        img = cv2.flip(img, 1)
        # 调整标签坐标
        labels[:, 1] = 1.0 - labels[:, 1]
    return img, labels

# YOLOv3风格的马赛克增强
def mosaic_augmentation(images, labels, img_size=416):
    # 实现YOLOv3风格的马赛克增强
    # 创建马赛克画布
    mosaic = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # 随机生成马赛克中心点
    xc, yc = [random.randint(img_size // 4, img_size * 3 // 4) for _ in range(2)]
    
    # 计算四个子图像的大小
    w1, h1 = xc, yc
    w2, h2 = img_size - xc, yc
    w3, h3 = xc, img_size - yc
    w4, h4 = img_size - xc, img_size - yc
    
    # 处理每张图像
    for i, (img, l) in enumerate(zip(images, labels)):
        # 调整图像大小
        h, w = img.shape[:2]
        
        if i == 0:  # 左上
            img_resized = cv2.resize(img, (w1, h1))
            mosaic[:h1, :w1] = img_resized
            if len(l) > 0:
                l[:, 1:] = l[:, 1:] * np.array([w1/w, h1/h, w1/w, h1/h])
                l[:, 1:3] = l[:, 1:3] + np.array([0, 0])
        elif i == 1:  # 右上
            img_resized = cv2.resize(img, (w2, h2))
            mosaic[:h2, xc:] = img_resized
            if len(l) > 0:
                l[:, 1:] = l[:, 1:] * np.array([w2/w, h2/h, w2/w, h2/h])
                l[:, 1:3] = l[:, 1:3] + np.array([xc, 0])
        elif i == 2:  # 左下
            img_resized = cv2.resize(img, (w3, h3))
            mosaic[yc:, :w3] = img_resized
            if len(l) > 0:
                l[:, 1:] = l[:, 1:] * np.array([w3/w, h3/h, w3/w, h3/h])
                l[:, 1:3] = l[:, 1:3] + np.array([0, yc])
        elif i == 3:  # 右下
            img_resized = cv2.resize(img, (w4, h4))
            mosaic[yc:, xc:] = img_resized
            if len(l) > 0:
                l[:, 1:] = l[:, 1:] * np.array([w4/w, h4/h, w4/w, h4/h])
                l[:, 1:3] = l[:, 1:3] + np.array([xc, yc])
    
    # 合并标签
    final_labels = np.zeros((0, 5), dtype=np.float32)
    for l in labels:
        if len(l) > 0:
            # 过滤超出边界的标签
            l = l[(l[:, 1] > 0) & (l[:, 1] < 1) & (l[:, 2] > 0) & (l[:, 2] < 1)]
            final_labels = np.vstack((final_labels, l))
    
    return mosaic, final_labels

# 加载标签
def load_labels(label_path, img_width, img_height, img_size):
    # 加载YOLO格式的标签并进行坐标转换
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)
    
    labels = np.loadtxt(label_path, delimiter=' ', dtype=np.float32)
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=0)
    
    # 归一化坐标
    labels[:, 1] /= img_width / img_size
    labels[:, 2] /= img_height / img_size
    labels[:, 3] /= img_width / img_size
    labels[:, 4] /= img_height / img_size
    
    return labels

# 处理单个图像和标签
def process_image(img_path, img_size, augment=False, current_img_size=None):
    # 处理单个图像和标签，支持YOLOv3风格的预处理和增强
    # 如果提供了当前图像大小，则使用它（用于多尺度训练）
    if current_img_size is None:
        current_img_size = img_size
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    
    # 获取图像尺寸
    h, w = img.shape[:2]
    
    # 构建标签路径
    img_name = os.path.basename(img_path)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(config.label_dir, label_name)
    
    # 加载标签
    labels = load_labels(label_path, w, h, current_img_size)
    
    # 数据增强 - YOLOv3风格的完整增强流程
    if augment:
        # 1. 随机调整亮度、对比度、饱和度和色调
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        
        # 2. 随机裁剪
        if random.random() < 0.4:
            img, labels = random_crop(img, labels)
        
        # 3. 随机旋转
        if random.random() < 0.4 and len(labels) > 0:
            img, labels = random_rotate(img, labels)
        
        # 4. 随机翻转 - YOLOv3论文中强调的重要增强方法
        img, labels = random_flip(img, labels)
    
    # 调整图像大小 - YOLOv3要求输入尺寸为32的倍数
    img = cv2.resize(img, (current_img_size, current_img_size), interpolation=cv2.INTER_LINEAR)
    
    # 归一化图像 - YOLOv3风格的标准化
    img = img.astype(np.float32) / 255.0
    img = img[..., ::-1]  # BGR到RGB
    
    # 额外的归一化：减去均值，除以标准差 - 类似YOLOv3的预处理
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # 创建目标张量（YOLO格式）
    target = np.zeros((config.num_anchors, config.grid_size, config.grid_size, 6), dtype=np.float32)
    
    if len(labels) > 0:
        # 处理每个标签
        for label in labels:
            class_id, x_center, y_center, width, height = label
            
            # 计算网格坐标
            grid_x = int(x_center * config.grid_size)
            grid_y = int(y_center * config.grid_size)
            
            # 确保在有效范围内
            grid_x = min(max(grid_x, 0), config.grid_size - 1)
            grid_y = min(max(grid_y, 0), config.grid_size - 1)
            
            # 计算相对于网格的坐标偏移
            tx = x_center * config.grid_size - grid_x
            ty = y_center * config.grid_size - grid_y
            
            # 计算相对于锚框的宽高比例的对数
            tw = np.log(width * current_img_size / config.selected_anchors[:, 0])
            th = np.log(height * current_img_size / config.selected_anchors[:, 1])
            
            # 计算IoU并选择最佳锚框
            ious = []
            for anchor in config.selected_anchors:
                # 计算预测框和锚框的IoU
                iou = _calculate_iou(np.array([0, 0, width * current_img_size, height * current_img_size]), 
                                    np.array([0, 0, anchor[0], anchor[1]]))
                ious.append(iou)
            best_anchor = np.argmax(ious)
            
            # 分配目标到最佳锚框
            target[best_anchor, grid_y, grid_x, 0] = tx
            target[best_anchor, grid_y, grid_x, 1] = ty
            target[best_anchor, grid_y, grid_x, 2] = tw[best_anchor]
            target[best_anchor, grid_y, grid_x, 3] = th[best_anchor]
            target[best_anchor, grid_y, grid_x, 4] = 1.0  # 置信度
            target[best_anchor, grid_y, grid_x, 5] = class_id
    
    return img, target

# IoU计算
def _calculate_iou(box1, box2):
    # 计算两个边界框的IoU
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

# 创建数据集 - YOLOv3风格
def create_dataset(img_dir, img_size, batch_size, augment=False):
    # 创建tf.data.Dataset用于训练或验证，支持YOLOv3风格的数据加载
    # 获取图像文件列表
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 为多尺度训练创建图像大小生成器
    def get_current_img_size():
        if augment and config.multiscale_training:
            # YOLOv3风格的多尺度训练：32的倍数
            sizes = list(range(config.multiscale_min, config.multiscale_max + 1, 32))
            return random.choice(sizes)
        return img_size
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices(img_files)
    
    # 映射处理函数
    def load_and_process(img_path):
        # 使用tf.numpy_function包装Python函数
        def py_func(img_path_str):
            # 对于训练，每次都随机选择图像大小
            current_size = get_current_img_size() if augment else img_size
            img, target = process_image(img_path_str.decode('utf-8'), img_size, augment, current_size)
            return img.astype(np.float32), target.astype(np.float32)
        
        return tf.numpy_function(py_func, [img_path], [tf.float32, tf.float32])
    
    # 应用处理函数
    dataset = dataset.map(
        load_and_process,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # 处理形状信息
    dataset = dataset.map(
        lambda x, y: (tf.ensure_shape(x, (None, None, 3)),
                    tf.ensure_shape(y, (config.num_anchors, config.grid_size, config.grid_size, 6)))
    )
    
    # 打乱、批处理和预取
    if augment:
        dataset = dataset.shuffle(buffer_size=min(1000, len(img_files)), reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

# YOLOv3风格的损失函数 - 官方实现风格
class YoloV3Loss(tf.keras.losses.Loss):
    def __init__(self, anchors, grid_size):
        super(YoloV3Loss, self).__init__()
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        self.grid_size = grid_size
        
        # YOLOv3官方的损失权重
        self.lambda_coord = 5.0  # 坐标损失权重 - 官方值
        self.lambda_noobj = 0.5  # 无目标损失权重 - 官方值
        self.lambda_class = 1.0  # 类别损失权重
        
        # 构建网格坐标
        self.grid = self._build_grid()
    
    def _build_grid(self):
        # 构建YOLOv3的网格坐标
        grid_x = tf.range(self.grid_size, dtype=tf.float32)
        grid_y = tf.range(self.grid_size, dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        
        grid = tf.stack([grid_x, grid_y], axis=-1)
        grid = tf.reshape(grid, [1, self.grid_size, self.grid_size, 1, 2])
        
        return grid
    
    def call(self, y_true, y_pred):
        # 实现YOLOv3官方风格的损失函数计算
        # 获取批次大小
        batch_size = tf.shape(y_true)[0]
        
        # 重塑预测和目标
        # 预测形状: [batch, grid, grid, anchors, 6] (x, y, w, h, conf, class)
        y_pred = tf.reshape(y_pred, 
                          (batch_size, self.grid_size, self.grid_size, 
                           self.anchors.shape[0], 6))
        
        # 应用sigmoid激活到x, y和置信度
        # YOLOv3官方实现中对坐标和置信度使用sigmoid激活
        pred_xy = tf.sigmoid(y_pred[..., :2])  # x, y (相对于网格)
        pred_wh = y_pred[..., 2:4]  # w, h (对数空间)
        pred_conf = tf.sigmoid(y_pred[..., 4:5])  # 置信度
        pred_class = tf.sigmoid(y_pred[..., 5:6])  # 类别
        
        # 获取目标掩码
        obj_mask = y_true[..., 4:5]  # 有目标的位置为1
        noobj_mask = 1 - obj_mask     # 无目标的位置为1
        
        # 计算坐标损失
        # 只对有目标的位置计算
        coord_loss = obj_mask * tf.square(pred_xy - y_true[..., :2])
        
        # 计算宽高损失 - YOLOv3使用根号来平衡大小目标
        # 注意：tf.sqrt在反向传播时对于接近0的值可能不稳定，这里使用平滑处理
        pred_wh_sqrt = tf.sign(pred_wh) * tf.sqrt(tf.abs(pred_wh) + 1e-10)
        true_wh_sqrt = tf.sqrt(y_true[..., 2:4] + 1e-10)  # 避免除以0
        coord_loss += obj_mask * tf.square(pred_wh_sqrt - true_wh_sqrt)
        coord_loss = self.lambda_coord * tf.reduce_sum(coord_loss)
        
        # 计算IoU用于置信度损失的加权 - YOLOv3风格
        # 解码预测框
        pred_boxes = self._decode_boxes(pred_xy, pred_wh)
        true_boxes = self._decode_boxes(y_true[..., :2], y_true[..., 2:4])
        
        # 计算预测框和真实框之间的IoU
        iou = self._calculate_iou(pred_boxes, true_boxes)
        
        # 有目标的置信度损失 - YOLOv3的一个关键特点是使用IoU作为置信度的目标值
        obj_conf_loss = obj_mask * tf.square(pred_conf - iou)
        obj_conf_loss = tf.reduce_sum(obj_conf_loss)
        
        # 无目标的置信度损失 - 应用Hard Negative Mining
        # 只对IoU低于阈值的位置计算无目标损失
        iou_threshold = 0.5
        noobj_mask = noobj_mask * tf.cast(iou < iou_threshold, tf.float32)
        noobj_conf_loss = noobj_mask * tf.square(pred_conf)
        noobj_conf_loss = self.lambda_noobj * tf.reduce_sum(noobj_conf_loss)
        
        # 计算类别损失 - 使用二元交叉熵
        class_loss = obj_mask * tf.square(pred_class - y_true[..., 5:6])
        class_loss = self.lambda_class * tf.reduce_sum(class_loss)
        
        # 总损失
        total_loss = coord_loss + obj_conf_loss + noobj_conf_loss + class_loss
        
        # 计算有目标的数量，用于归一化
        num_objects = tf.maximum(1.0, tf.reduce_sum(obj_mask))
        
        return total_loss / num_objects
    
    def _decode_boxes(self, xy, wh):
        # 将网络输出的坐标解码为实际的边界框坐标
        # 将相对坐标转换为绝对坐标
        xy = (xy + self.grid) / self.grid_size
        
        # 将对数空间的宽高转换为实际宽高
        wh = tf.exp(wh) * self.anchors / self.grid_size
        
        # 组合为边界框
        x1 = xy[..., 0:1] - wh[..., 0:1] / 2
        y1 = xy[..., 1:2] - wh[..., 1:2] / 2
        x2 = xy[..., 0:1] + wh[..., 0:1] / 2
        y2 = xy[..., 1:2] + wh[..., 1:2] / 2
        
        return tf.concat([x1, y1, x2, y2], axis=-1)
    
    def _calculate_iou(self, boxes1, boxes2):
        # 计算两个边界框集合之间的IoU
        # 计算交集
        intersection_x1 = tf.maximum(boxes1[..., 0:1], boxes2[..., 0:1])
        intersection_y1 = tf.maximum(boxes1[..., 1:2], boxes2[..., 1:2])
        intersection_x2 = tf.minimum(boxes1[..., 2:3], boxes2[..., 2:3])
        intersection_y2 = tf.minimum(boxes1[..., 3:4], boxes2[..., 3:4])
        
        # 计算交集面积
        intersection_area = tf.maximum(0.0, intersection_x2 - intersection_x1) * \
                          tf.maximum(0.0, intersection_y2 - intersection_y1)
        
        # 计算并集面积
        area1 = (boxes1[..., 2:3] - boxes1[..., 0:1]) * \
               (boxes1[..., 3:4] - boxes1[..., 1:2])
        area2 = (boxes2[..., 2:3] - boxes2[..., 0:1]) * \
               (boxes2[..., 3:4] - boxes2[..., 1:2])
        union_area = area1 + area2 - intersection_area
        
        # 计算IoU
        iou = intersection_area / (union_area + 1e-10)  # 避免除以0
        
        return iou

# 预热学习率调度
def get_warmup_lr(epoch, base_lr, warmup_epochs):
    # 实现学习率预热
    if epoch < warmup_epochs:
        # 线性预热
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

# 余弦退火学习率调度
def get_cosine_lr(epoch, base_lr, epochs, warmup_epochs):
    # 实现YOLOv3风格的余弦退火学习率调度
    if epoch < warmup_epochs:
        return get_warmup_lr(epoch, base_lr, warmup_epochs)
    
    # 余弦退火
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    return lr

# 初始化模型
def initialize_model():
    # 初始化模型、损失函数和优化器，实现YOLOv3风格的配置
    # 创建模型实例
    model = YoloFace()
    
    # 构建模型
    dummy_input = tf.random.uniform((1, config.img_size, config.img_size, 3))
    model(dummy_input, training=False)
    
    # 定义损失函数
    loss_function = YoloV3Loss(config.selected_anchors, config.grid_size)
    
    # 定义优化器 - YOLOv3风格：带权重衰减的Adam
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        weight_decay=config.weight_decay
    )
    
    return model, loss_function, optimizer

# 额外的数据增强功能
def random_rotate(img, labels, angle_range=(-10, 10)):
    # YOLOv3风格的随机旋转增强
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 执行旋转
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    
    # 调整标签坐标
    if len(labels) > 0:
        # 转换标签中心坐标
        labels_centers = np.zeros((len(labels), 2), dtype=np.float32)
        labels_centers[:, 0] = labels[:, 1] * w
        labels_centers[:, 1] = labels[:, 2] * h
        
        # 应用旋转变换
        rotated_centers = cv2.transform(np.expand_dims(labels_centers, axis=0), M)[0]
        
        # 转换回归一化坐标
        labels[:, 1] = rotated_centers[:, 0] / w
        labels[:, 2] = rotated_centers[:, 1] / h
    
    return rotated_img, labels

def random_crop(img, labels, min_size=0.3, max_size=1.0):
    # YOLOv3风格的随机裁剪增强
    h, w = img.shape[:2]
    
    # 随机选择裁剪区域的大小
    crop_size = random.uniform(min_size, max_size)
    crop_h = int(h * crop_size)
    crop_w = int(w * crop_size)
    
    # 随机选择裁剪区域的位置
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    # 执行裁剪
    cropped_img = img[y1:y2, x1:x2]
    
    # 调整标签坐标
    if len(labels) > 0:
        # 转换标签坐标
        labels[:, 1] = (labels[:, 1] * w - x1) / crop_w
        labels[:, 2] = (labels[:, 2] * h - y1) / crop_h
        labels[:, 3] = labels[:, 3] * w / crop_w
        labels[:, 4] = labels[:, 4] * h / crop_h
        
        # 过滤掉在裁剪区域外的标签
        valid_mask = ((labels[:, 1] > 0) & (labels[:, 1] < 1) & 
                     (labels[:, 2] > 0) & (labels[:, 2] < 1) &
                     (labels[:, 3] > 0) & (labels[:, 4] > 0))
        labels = labels[valid_mask]
    
    return cropped_img, labels

def apply_mosaic_augmentation(images_batch, targets_batch, img_size=416):
    # 将马赛克增强集成到批量处理中
    batch_size = len(images_batch)
    
    # 对于每个批次，随机决定是否应用马赛克增强
    if random.random() < 0.5 and batch_size >= 4:
        # 选择4张图像来创建马赛克
        indices = random.sample(range(batch_size), 4)
        selected_images = [images_batch[i] for i in indices]
        selected_targets = [targets_batch[i] for i in indices]
        
        # 应用马赛克增强
        mosaic_img, mosaic_target = mosaic_augmentation(selected_images, selected_targets, img_size)
        
        # 更新批次中的第一张图像
        images_batch[0] = mosaic_img
        targets_batch[0] = mosaic_target
    
    return images_batch, targets_batch

# 训练步骤
def train_step(model, loss_function, optimizer, images, targets, current_epoch):
    # 单个训练步骤，支持YOLOv3风格的训练过程
    # 动态调整学习率
    if config.cosine_scheduler:
        new_lr = get_cosine_lr(current_epoch, config.learning_rate, config.epochs, config.warmup_epochs)
        optimizer.learning_rate.assign(new_lr)
    
    with tf.GradientTape() as tape:
        # 前向传播
        predictions = model(images, training=True)
        
        # 计算损失
        loss = loss_function(targets, predictions)
    
    # 计算梯度
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # YOLOv3风格的梯度裁剪
    gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
    
    # 更新参数
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# 验证函数
def validate_model(model, loss_function, val_dataset):
    # 验证模型性能，实现YOLOv3风格的评估
    val_loss = 0.0
    num_batches = 0
    num_objects = 0
    
    for images, targets in val_dataset:
        # 前向传播
        predictions = model(images, training=False)
        
        # 计算损失
        batch_loss = loss_function(targets, predictions)
        
        # 计算批次中的目标数量
        batch_objects = tf.reduce_sum(targets[..., 4])
        
        # 累计损失
        val_loss += batch_loss.numpy() * batch_objects.numpy()
        num_objects += batch_objects.numpy()
        num_batches += 1
    
    # 计算平均损失
    avg_val_loss = val_loss / max(1.0, num_objects)
    
    return avg_val_loss

# 计算IoU
def calculate_iou(box1, box2):
    # 计算两个边界框的IoU
    # box1, box2: [x1, y1, x2, y2]
    
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算两个边界框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou

# 计算平均精度（AP）
def calculate_ap(recall, precision):
    # 计算平均精度（AP）
    # 确保精度是单调递减的
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # 计算PR曲线下的面积
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    
    return ap

# 计算平均精度均值（mAP）
def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    # 计算平均精度均值（mAP）
    # 按置信度排序所有预测
    all_detections = []
    all_gt_boxes = []
    all_gt_classes = []
    image_ids = []
    
    # 收集所有检测和真值
    for i, (preds, gts) in enumerate(zip(predictions, ground_truths)):
        all_detections.extend([(i, pred[0], pred[1], pred[2], pred[3], pred[4]) for pred in preds])
        all_gt_boxes.extend([(i, gt[0], gt[1], gt[2], gt[3]) for gt in gts])
        all_gt_classes.extend([(i, gt[5]) for gt in gts])
        image_ids.append(i)
    
    # 按置信度降序排序检测结果
    all_detections.sort(key=lambda x: x[4], reverse=True)
    
    # 初始化变量
    true_positives = np.zeros(len(all_detections))
    false_positives = np.zeros(len(all_detections))
    detected_gt = set()
    num_gt = len(all_gt_boxes)
    
    # 处理每个检测结果
    for i, (img_id, x1, y1, x2, y2, conf) in enumerate(all_detections):
        # 查找当前图像的所有真值边界框
        img_gt_boxes = [gt for j, gt in enumerate(all_gt_boxes) if gt[0] == img_id]
        img_gt_classes = [gt for j, gt in enumerate(all_gt_classes) if gt[0] == img_id]
        
        best_iou = 0
        best_gt_idx = -1
        
        # 计算与每个真值边界框的IoU
        for j, (_, gt_x1, gt_y1, gt_x2, gt_y2) in enumerate(img_gt_boxes):
            iou = calculate_iou([x1, y1, x2, y2], [gt_x1, gt_y1, gt_x2, gt_y2])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # 判断是否为真阳性
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_key = (img_id, best_gt_idx)
            if gt_key not in detected_gt:
                detected_gt.add(gt_key)
                true_positives[i] = 1
            else:
                false_positives[i] = 1
        else:
            false_positives[i] = 1
    
    # 计算累积的TP和FP
    cumulative_true_positives = np.cumsum(true_positives)
    cumulative_false_positives = np.cumsum(false_positives)
    
    # 计算精确率和召回率
    precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives + 1e-16)
    recall = cumulative_true_positives / max(1, num_gt)
    
    # 计算AP
    ap = calculate_ap(recall, precision)
    
    return ap

# 可视化检测结果
def visualize_detection(image, detections, save_path=None):
    # 可视化检测结果
    # image: 输入图像 (tensor)
    # detections: 检测结果列表
    # save_path: 保存路径
    
    # 反归一化图像
    img = image.numpy()
    
    # 还原图像（如果有归一化）
    if np.max(img) <= 1.0:
        img = img * 255.0
    
    # 转换为BGR格式（如果是RGB）
    if img.shape[-1] == 3:
        img = img[..., ::-1]  # RGB到BGR
    
    # 转换为uint8
    img = img.astype(np.uint8)
    
    # 绘制边界框
    for detection in detections:
        x1, y1, x2, y2, conf = detection[:5]
        
        # 确保坐标在有效范围内
        h, w = img.shape[:2]
        x1 = max(0, min(int(x1 * w), w - 1))
        y1 = max(0, min(int(y1 * h), h - 1))
        x2 = max(0, min(int(x2 * w), w - 1))
        y2 = max(0, min(int(y2 * h), h - 1))
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制置信度
        label = f"Face: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        cv2.imwrite(save_path, img)
    
    return img

# 模型评估
def evaluate_model(model, val_dataset, conf_threshold=0.5, iou_threshold=0.5, visualize=False, save_dir='visualizations'):
    # 评估模型性能，计算mAP等指标
    all_predictions = []
    all_ground_truths = []
    
    # 创建保存目录
    if visualize:
        os.makedirs(save_dir, exist_ok=True)
    
    # 限制评估样本数量，避免过长的评估时间
    num_batches = min(20, len(val_dataset))
    
    for batch_idx, (images, targets) in enumerate(val_dataset.take(num_batches)):
        # 前向传播
        raw_preds = model(images, training=False)
        
        # 对每个样本进行后处理
        for i in range(len(images)):
            # 提取预测结果
            pred = tf.reshape(raw_preds[0], (-1, 6))  # 注意这里的索引可能需要根据实际输出调整
            
            # 应用非极大值抑制
            boxes = pred[:, :4]
            confs = pred[:, 4:5]
            clss = pred[:, 5:6]
            
            # 转换为检测格式
            detections = tf.concat([xywh2xyxy(boxes), confs, clss], axis=-1)
            
            # 应用非极大值抑制
            nms_detections = non_max_suppression(
                detections.numpy(), 
                conf_threshold=conf_threshold, 
                iou_threshold=iou_threshold
            )
            
            all_predictions.append(nms_detections)
            
            # 提取真实标签
            gt = targets[i].numpy()
            gt = np.reshape(gt, (-1, 6))
            gt = gt[gt[:, 4] > 0.5]  # 只保留有目标的位置
            all_ground_truths.append(gt)
            
            # 可视化检测结果
            if visualize and i < 3 and batch_idx < 2:  # 每个批次只可视化前3张图像，最多2个批次
                img_idx = batch_idx * len(images) + i
                save_path = os.path.join(save_dir, f'detection_{img_idx}.jpg')
                visualize_detection(images[i], nms_detections, save_path)
    
    # 计算mAP
    if all_predictions and all_ground_truths:
        map_score = calculate_map(all_predictions, all_ground_truths, iou_threshold)
    else:
        map_score = 0.0
    
    # 计算检测统计
    total_detections = sum(len(pred) for pred in all_predictions)
    total_ground_truths = sum(len(gt) for gt in all_ground_truths)
    
    return map_score, total_detections, total_ground_truths

# 保存模型
def save_model(model, optimizer, epoch, loss, is_best=False):
    # 保存模型检查点，实现YOLOv3风格的模型保存
    # 使用TensorFlow的标准检查点机制
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    # 保存检查点
    if is_best:
        # 保存最佳模型
        model_path = os.path.join(config.checkpoint_dir, 'best_model')
        model.save(model_path, save_format='tf')
        
        # 同时保存检查点文件，方便恢复训练
        checkpoint_path = os.path.join(config.checkpoint_dir, 'best_checkpoint')
        checkpoint.save(file_prefix=checkpoint_path)
        print(f"最佳模型已保存到 {model_path}")
        print(f"最佳检查点已保存到 {checkpoint_path}")
    else:
        # 保存普通检查点
        model_path = os.path.join(config.checkpoint_dir, f'model_epoch_{epoch}')
        model.save(model_path, save_format='tf')
        
        # 保存检查点文件
        checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}')
        checkpoint.save(file_prefix=checkpoint_path)
        print(f"模型检查点已保存到 {model_path}")
        print(f"优化器检查点已保存到 {checkpoint_path}")
    
    # 保存训练元数据
    metadata = {
        'epoch': epoch,
        'loss': float(loss),
        'config': vars(config)
    }
    
    metadata_path = os.path.join(config.checkpoint_dir, f'metadata_epoch_{epoch}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

# 主训练函数
def main():
    # 主训练函数，实现YOLOv3风格的训练流程
    print(f'开始YOLOv3风格的YOLO Face训练')
    print(f"配置: 批量大小={config.batch_size}, 学习率={config.learning_rate}, 轮数={config.epochs}")
    print(f"多尺度训练: {config.multiscale_training}, 马赛克增强: {config.mosaic_augmentation}")
    
    # 创建评估和可视化目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.eval_images_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
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
    model, loss_function, optimizer = initialize_model()
    
    # 创建检查点管理器
    checkpoint_dir = config.checkpoint_dir
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)
    
    # 加载检查点（如果存在）
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f'已恢复检查点: {manager.latest_checkpoint}')
    
    # 初始化训练日志
    train_loss_history = []
    val_loss_history = []
    map_history = []
    best_map = 0.0
    best_epoch = 0
    
    # 初始化TensorBoard日志记录器
    train_summary_writer = tf.summary.create_file_writer(os.path.join(config.log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(config.log_dir, 'validation'))
    
    # 训练循环
    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch+1}/{config.epochs}')
        
        # 获取当前学习率
        current_lr = get_cosine_lr(epoch, config.learning_rate, config.epochs, config.warmup_epochs)
        optimizer.lr.assign(current_lr)
        print(f'当前学习率: {current_lr}')
        
        # 设置当前图像大小（多尺度训练）
        if config.multiscale_training:
            # YOLOv3风格的多尺度训练，随机选择32的倍数
            current_img_size = 32 * (10 + random.randint(0, 16))  # 320-608，步长32
            print(f'当前图像大小: {current_img_size}x{current_img_size}')
            # 重新创建训练数据集
            train_dataset = create_dataset(
                config.train_dir,
                current_img_size,
                config.batch_size,
                augment=True
            )
        
        # 训练一个epoch
        start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for images, targets in train_dataset:
            # 执行训练步骤
            loss = train_step(model, loss_function, optimizer, images, targets, epoch)
            epoch_loss += loss.numpy()
            num_batches += 1
            
            # 打印批次信息
            if num_batches % 10 == 0:
                print(f'  批次 {num_batches}, 损失: {loss.numpy():.4f}')
                
            # TensorBoard记录
            with train_summary_writer.as_default():
                tf.summary.scalar('batch_loss', loss, step=epoch * len(list(train_dataset)) + num_batches)
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / num_batches
        train_loss_history.append(avg_train_loss)
        print(f'训练损失: {avg_train_loss:.4f}, 耗时: {time.time() - start_time:.2f}秒')
        
        # TensorBoard记录训练损失
        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', avg_train_loss, step=epoch)
            tf.summary.scalar('learning_rate', current_lr, step=epoch)
        
        # 验证模型
        val_loss = validate_model(model, loss_function, val_dataset)
        val_loss_history.append(val_loss)
        print(f'验证损失: {val_loss:.4f}')
        
        # TensorBoard记录验证损失
        with val_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', val_loss, step=epoch)
        
        # 计算mAP和详细评估（每5个epoch）
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            print(f'评估模型性能...')
            
            # 执行模型评估
            map_score, class_ap, precision, recall, total_detections, total_ground_truths = evaluate_model(
                model, val_dataset, visualize=(epoch + 1) % 10 == 0, epoch=epoch
            )
            
            # 更新最佳模型信息
            if map_score > best_map:
                best_map = map_score
                best_epoch = epoch + 1
            
            map_history.append(map_score)
            
            # 输出详细评估结果
            print(f'mAP: {map_score:.4f}, 最佳mAP: {best_map:.4f} (Epoch {best_epoch})')
            print(f'检测到的目标: {total_detections}, 真值目标: {total_ground_truths}')
            print(f'精确率: {precision:.4f}, 召回率: {recall:.4f}')
            
            # 输出各分类的AP（虽然只有人脸一个类别）
            for cls_id, ap in class_ap.items():
                print(f'类别 {cls_id} AP: {ap:.4f}')
            
            # TensorBoard记录评估指标
            with val_summary_writer.as_default():
                tf.summary.scalar('mAP', map_score, step=epoch)
                tf.summary.scalar('precision', precision, step=epoch)
                tf.summary.scalar('recall', recall, step=epoch)
            
            # 保存评估日志
            eval_log_path = os.path.join(config.log_dir, f'eval_log_epoch_{epoch+1}.txt')
            with open(eval_log_path, 'w') as f:
                f.write(f'Epoch: {epoch+1}\n')
                f.write(f'mAP: {map_score:.4f}\n')
                f.write(f'Best mAP: {best_map:.4f} (Epoch {best_epoch})\n')
                f.write(f'Precision: {precision:.4f}\n')
                f.write(f'Recall: {recall:.4f}\n')
                f.write(f'Total Detections: {total_detections}\n')
                f.write(f'Total Ground Truths: {total_ground_truths}\n')
        
        # 保存检查点
        if (epoch + 1) % config.save_interval == 0 or epoch == config.epochs - 1:
            # 保存常规检查点
            save_model(model, optimizer, epoch + 1, val_loss, is_best=(map_score == best_map))
            
            # 如果是最佳模型，特别标记
            if map_score == best_map:
                print(f'新的最佳模型已保存！mAP: {best_map:.4f}')
                best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{best_epoch}.h5')
                model.save(best_model_path)
            
            # 保存训练历史
            history = {
                'train_loss': train_loss_history,
                'val_loss': val_loss_history,
                'map': map_history,
                'best_map': best_map,
                'best_epoch': best_epoch
            }
            with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f)
    
    # 训练结束，输出最终评估结果
    print('\n训练完成!')
    print(f'最佳模型在第{best_epoch}轮达到mAP: {best_map:.4f}')
    
    # 最终模型评估
    print('执行最终模型评估...')
    final_model = tf.keras.models.load_model(
        os.path.join(checkpoint_dir, f'best_model_epoch_{best_epoch}.h5'),
        custom_objects={'yolo_loss': YoloV3Loss()}
    )
    
    final_map, _, final_precision, final_recall, final_detections, final_ground_truths = evaluate_model(
        final_model, val_dataset, visualize=True, epoch=config.epochs
    )
    
    print(f'最终评估结果:')
    print(f'mAP: {final_map:.4f}')
    print(f'精确率: {final_precision:.4f}')
    print(f'召回率: {final_recall:.4f}')
    print(f'检测到的目标: {final_detections}, 真值目标: {final_ground_truths}')
    
    # 保存最终评估报告
    final_report_path = os.path.join(config.log_dir, 'final_evaluation_report.txt')
    with open(final_report_path, 'w') as f:
        f.write('YOLO Face 训练完成评估报告\n')
        f.write(f'==============================\n')
        f.write(f'训练轮数: {config.epochs}\n')
        f.write(f'最佳轮次: {best_epoch}\n')
        f.write(f'最终mAP: {final_map:.4f}\n')
        f.write(f'最佳mAP: {best_map:.4f}\n')
        f.write(f'精确率: {final_precision:.4f}\n')
        f.write(f'召回率: {final_recall:.4f}\n')
        f.write(f'检测到的目标: {final_detections}\n')
        f.write(f'真值目标: {final_ground_truths}\n')
        f.write(f'配置信息:\n')
        for key, value in vars(config).items():
            f.write(f'  {key}: {value}\n')

if __name__ == '__main__':
    main()