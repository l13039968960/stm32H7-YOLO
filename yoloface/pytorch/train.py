# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import onnx
import onnxruntime

# 导入yoloface模型定义
from yoloface import yoloface

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 配置参数
class Config:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 0.0005
        self.img_size = 56
        self.grid_size = 7  # 最终特征图大小
        self.num_anchors = 3
        self.anchors = np.array([[9,14], [12,17], [22,21]])
        self.train_dir = '../small_dataset'  # 训练数据集目录
        self.val_dir = '../small_dataset'    # 验证数据集目录
        self.checkpoint_dir = 'checkpoints'  # 模型保存目录
        self.save_interval = 10              # 每10个epoch保存一次模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multiscale_training = False     # 是否使用多尺度训练
        self.mosaic_augmentation = False     # 是否使用马赛克增强

config = Config()

# 创建检查点目录
os.makedirs(config.checkpoint_dir, exist_ok=True)

# 数据集定义
class FaceDataset(Dataset):
    def __init__(self, img_dir, transform=None, img_size=56, mosaic=False):
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.mosaic = mosaic
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        # 默认标签 (在实际应用中，应该从标注文件中读取)
        # 这里为了演示，我们假设有一个中心人脸
        h, w = img.shape[:2]
        # 创建默认标签（实际训练时应该从标注文件中加载）
        # format: [x_center, y_center, width, height, class_id]
        # 归一化到0-1范围
        labels = np.array([[0.5, 0.5, 0.3, 0.3, 0]])
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        # 调整图像大小
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 将BGR转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为张量
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # 处理标签
        # 将归一化坐标转换为网格坐标
        labels[:, 0] *= self.img_size
        labels[:, 1] *= self.img_size
        labels[:, 2] *= self.img_size
        labels[:, 3] *= self.img_size
        
        # 创建目标张量
        target = torch.zeros((config.num_anchors, config.grid_size, config.grid_size, 6))
        
        # 分配标签到网格和锚框
        for label in labels:
            x_center, y_center, width, height, class_id = label
            
            # 计算目标所在的网格
            grid_x = int(x_center / (self.img_size / config.grid_size))
            grid_y = int(y_center / (self.img_size / config.grid_size))
            
            # 计算相对于网格的偏移量
            tx = x_center / (self.img_size / config.grid_size) - grid_x
            ty = y_center / (self.img_size / config.grid_size) - grid_y
            
            # 计算相对于锚框的宽高比例的对数
            tw = np.log(width / config.anchors[:, 0])
            th = np.log(height / config.anchors[:, 1])
            
            # 计算IoU并选择最佳锚框
            ious = []
            for anchor in config.anchors:
                # 计算预测框和锚框的IoU
                iou = self._calculate_iou(np.array([0, 0, width, height]), 
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
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        # box1和box2的格式: [x1, y1, w, h]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        
        return intersection / (area1 + area2 - intersection + 1e-10)

# 数据增强函数定义
def _random_hue(img, delta=18.0):
    """随机调整色调"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = img[:, :, 0].astype(np.int16)
    h += np.random.uniform(-delta, delta)
    h = np.mod(h + 180, 180).astype(np.uint8)
    img[:, :, 0] = h
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def _random_saturation(img, lower=0.5, upper=1.5):
    """随机调整饱和度"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = img[:, :, 1].astype(np.float32)
    s *= np.random.uniform(lower, upper)
    s = np.clip(s, 0, 255).astype(np.uint8)
    img[:, :, 1] = s
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def _random_brightness(img, delta=32):
    """随机调整亮度"""
    img = img.astype(np.float32)
    img += np.random.uniform(-delta, delta)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _random_flip(img, flip_prob=0.5):
    """随机水平翻转"""
    if np.random.random() < flip_prob:
        img = cv2.flip(img, 1)
    return img

# 数据增强和变换
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: _random_hue(img)),
    transforms.Lambda(lambda img: _random_saturation(img)),
    transforms.Lambda(lambda img: _random_brightness(img)),
    transforms.Lambda(lambda img: _random_flip(img)),
])

# 创建数据集和数据加载器
train_dataset = FaceDataset(config.train_dir, transform=train_transform, 
                           img_size=config.img_size, mosaic=config.mosaic_augmentation)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                         shuffle=True, num_workers=0, pin_memory=False)  # 降低num_workers以避免Windows上的问题

val_dataset = FaceDataset(config.val_dir, img_size=config.img_size)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                       shuffle=False, num_workers=0, pin_memory=False)

# 定义YOLO损失函数
class YoloLoss(nn.Module):
    def __init__(self, anchors, grid_size, device):
        super(YoloLoss, self).__init__()
        self.anchors = torch.tensor(anchors).to(device)
        self.grid_size = grid_size
        self.device = device
        
        # 损失权重
        self.lambda_coord = 5.0  # 坐标损失权重
        self.lambda_noobj = 0.5  # 无目标损失权重
        
        # 损失函数
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
    
    def forward(self, predictions, targets):
        """
        计算YOLO损失
        predictions: [batch_size, num_anchors*6, grid_size, grid_size]
        targets: [batch_size, num_anchors, grid_size, grid_size, 6]
        """
        batch_size = predictions.size(0)
        num_anchors = len(self.anchors)
        
        # 重塑预测结果: [batch_size, num_anchors, 6, grid_size, grid_size] -> [batch_size, num_anchors, grid_size, grid_size, 6]
        predictions = predictions.view(batch_size, num_anchors, 6, self.grid_size, self.grid_size).permute(0, 1, 3, 4, 2)
        
        # 计算置信度掩码
        obj_mask = targets[..., 4] == 1  # 有目标的位置
        noobj_mask = targets[..., 4] == 0  # 无目标的位置
        
        # 初始化损失
        loss = 0
        
        # 计算坐标损失 (仅对有目标的位置)
        loss_x = self.mse_loss(predictions[obj_mask][..., 0], targets[obj_mask][..., 0])
        loss_y = self.mse_loss(predictions[obj_mask][..., 1], targets[obj_mask][..., 1])
        loss_w = self.mse_loss(predictions[obj_mask][..., 2], targets[obj_mask][..., 2])
        loss_h = self.mse_loss(predictions[obj_mask][..., 3], targets[obj_mask][..., 3])
        
        loss += self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)
        
        # 计算置信度损失
        # 有目标的置信度损失
        loss_obj = self.bce_loss(predictions[obj_mask][..., 4], targets[obj_mask][..., 4])
        # 无目标的置信度损失
        loss_noobj = self.bce_loss(predictions[noobj_mask][..., 4], targets[noobj_mask][..., 4])
        
        loss += loss_obj + self.lambda_noobj * loss_noobj
        
        # 计算类别损失 (仅对有目标的位置)
        loss_cls = self.bce_loss(predictions[obj_mask][..., 5], targets[obj_mask][..., 5])
        loss += loss_cls
        
        # 平均损失
        loss /= batch_size
        
        return loss

# 初始化模型、损失函数和优化器
def initialize_model():
    model = yoloface().to(config.device)
    criterion = YoloLoss(config.anchors, config.grid_size, config.device)
    
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    # 学习率调度器 - 余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                   T_max=config.epochs,
                                                   eta_min=0.00001)
    
    return model, criterion, optimizer, scheduler

# 训练函数
def train_one_epoch(model, criterion, optimizer, train_loader, epoch):
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (inputs, targets) in progress_bar:
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        # 统计损失
        running_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_description(f'Epoch {epoch+1}/{config.epochs}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

# 验证函数
def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

# 保存模型
def save_model(model, optimizer, epoch, loss, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    if is_best:
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, 'best_model.pth'))
        # 导出为ONNX格式
        export_to_onnx(model, os.path.join(config.checkpoint_dir, 'yoloface_best.onnx'))
    else:
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, f'model_epoch_{epoch}.pth'))

# 导出模型为ONNX格式
def export_to_onnx(model, onnx_path):
    """
    将PyTorch模型导出为ONNX格式
    
    Args:
        model: 训练好的PyTorch模型
        onnx_path: ONNX文件保存路径
    """
    # 设置模型为推理模式
    model.eval()
    
    # 创建一个虚拟输入（与模型期望的输入尺寸相同）
    dummy_input = torch.randn(1, 3, config.img_size, config.img_size, device=config.device)
    
    # 导出模型为ONNX格式
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 验证导出的模型
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX模型导出成功并通过验证: {onnx_path}")
        
        # 打印模型信息
        print(f"输入名称: {[input.name for input in onnx_model.graph.input]}")
        print(f"输出名称: {[output.name for output in onnx_model.graph.output]}")
        print(f"输入形状: {dummy_input.shape}")
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")

# 加载模型
def load_model(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

# 主训练函数
def main():
    print(f'使用设备: {config.device}')
    
    # 初始化模型
    model, criterion, optimizer, scheduler = initialize_model()
    
    # 加载预训练权重（如果有）
    # model, optimizer, start_epoch, _ = load_model(model, optimizer, 'checkpoints/best_model.pth')
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    print('开始训练...')
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # 训练一个epoch
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, epoch)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, criterion, val_loader)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step()
        
        # 打印epoch信息
        print(f'Epoch {epoch+1}/{config.epochs}, ' 
              f'Train Loss: {train_loss:.4f}, ' 
              f'Val Loss: {val_loss:.4f}, ' 
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 保存检查点
        if (epoch + 1) % config.save_interval == 0:
            save_model(model, optimizer, epoch + 1, val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch + 1, val_loss, is_best=True)
    
    end_time = time.time()
    print(f'训练完成! 总用时: {end_time - start_time:.2f}秒')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config.epochs + 1), train_losses, label='训练损失')
    plt.plot(range(1, config.epochs + 1), val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    
    # 最后导出训练好的模型为ONNX格式
    # print("\n训练完成，导出最终模型为ONNX格式...")
    # export_to_onnx(model, os.path.join(config.checkpoint_dir, 'yoloface_final.onnx'))
    
    # 可选：显示损失曲线
    try:
        plt.show()
    except:
        print("无法显示图像，但已保存到 loss_curve.png")

if __name__ == '__main__':
    main()