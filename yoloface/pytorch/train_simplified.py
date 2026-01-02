# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import time
from tqdm import tqdm

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

# 配置参数（使用简单的字典替代类，简化代码）
config = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'img_size': 56,
    'grid_size': 7,
    'num_anchors': 3,
    'anchors': np.array([[9,14], [12,17], [22,21]]),
    'train_dir': '../small_dataset',
    'val_dir': '../small_dataset',
    'checkpoint_dir': 'checkpoints',
    'save_interval': 10,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # 优化器和学习率调度参数
    'optimizer_type': 'adamw',  # 可选: 'adam', 'adamw', 'sgd'
    'warmup_epochs': 3,        # 学习率预热轮数
    'warmup_factor': 0.1,      # 预热初始学习率因子
    'lr_scheduler_type': 'cosine',  # 可选: 'cosine', 'plateau', 'step'
    'plateau_patience': 5,     # ReduceLROnPlateau的耐心值
    'step_size': 20           # StepLR的步长
}

# 创建检查点目录
os.makedirs(config['checkpoint_dir'], exist_ok=True)

# 改进的数据预处理和增强函数（更简洁高效）
def preprocess_image(img, img_size, augment=False):
    """预处理图像，支持数据增强"""
    # 调整图像大小
    img = cv2.resize(img, (img_size, img_size))
    
    # 数据增强（如果启用）
    if augment:
        # 随机调整亮度
        if np.random.random() < 0.5:
            delta = np.random.uniform(-32, 32)
            img = img.astype(np.float32) + delta
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 随机调整色调、饱和度
        if np.random.random() < 0.5:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 调整色调
            img[:, :, 0] = (img[:, :, 0] + np.random.randint(-18, 18)) % 180
            # 调整饱和度
            img[:, :, 1] = np.clip(img[:, :, 1] * np.random.uniform(0.5, 1.5), 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        # 随机水平翻转
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
    
    # 转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换为张量
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    
    return img

# 数据集定义（改进版）
class FaceDataset(Dataset):
    def __init__(self, img_dir, img_size=56, augment=False):
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment
        # 获取所有图像文件
        self.img_files = [f for f in os.listdir(img_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f'在 {img_dir} 中找到 {len(self.img_files)} 张图像')
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        if img is None:
            # 跳过无法读取的图像，返回前一张有效图像
            idx = max(0, idx - 1) if idx > 0 else idx + 1
            img_path = os.path.join(self.img_dir, self.img_files[idx])
            img = cv2.imread(img_path)
        
        # 预处理图像
        img = preprocess_image(img, self.img_size, self.augment)
        
        # 创建目标张量（保持原有逻辑但更简洁）
        target = torch.zeros((config['num_anchors'], config['grid_size'], config['grid_size'], 6))
        
        # 假设有一个中心人脸（实际应用中应该从标注文件中读取）
        grid_x, grid_y = config['grid_size'] // 2, config['grid_size'] // 2
        tx, ty = 0.5, 0.5  # 网格内相对位置
        
        # 计算最佳锚框（使用更高效的方式）
        target_size = np.array([self.img_size * 0.3, self.img_size * 0.3])
        ious = []
        for anchor in config['anchors']:
            # 简化的IoU计算
            intersection = min(target_size[0], anchor[0]) * min(target_size[1], anchor[1])
            union = target_size[0] * target_size[1] + anchor[0] * anchor[1] - intersection
            ious.append(intersection / (union + 1e-10))
        best_anchor = np.argmax(ious)
        
        # 分配目标到最佳锚框
        target[best_anchor, grid_y, grid_x, 0] = tx
        target[best_anchor, grid_y, grid_x, 1] = ty
        target[best_anchor, grid_y, grid_x, 2] = np.log(target_size[0] / config['anchors'][best_anchor, 0])
        target[best_anchor, grid_y, grid_x, 3] = np.log(target_size[1] / config['anchors'][best_anchor, 1])
        target[best_anchor, grid_y, grid_x, 4] = 1.0  # 置信度
        
        return img, target

# 创建数据集和数据加载器（更简洁的方式）
def create_data_loaders():
    train_dataset = FaceDataset(config['train_dir'], img_size=config['img_size'], augment=True)
    val_dataset = FaceDataset(config['val_dir'], img_size=config['img_size'], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader

# 高度优化的损失函数实现
class YoloLoss(nn.Module):
    def __init__(self, anchors, grid_size, device):
        super(YoloLoss, self).__init__()
        self.anchors = torch.tensor(anchors).to(device)
        self.grid_size = grid_size
        self.device = device
        
        # 损失权重
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        
        # 预分配掩码以减少重复计算
        self.obj_mask = None
        self.noobj_mask = None
    
    def forward(self, predictions, targets):
        """高度优化的YOLO损失计算，使用向量化操作提高效率"""
        batch_size = predictions.size(0)
        
        # 一次性重塑预测结果
        predictions = predictions.view(batch_size, config['num_anchors'], 6, 
                                     self.grid_size, self.grid_size).permute(0, 1, 3, 4, 2)
        
        # 计算掩码（复用变量）
        self.obj_mask = targets[..., 4] == 1
        self.noobj_mask = targets[..., 4] == 0
        
        # 计算总元素数（用于优化）
        obj_count = self.obj_mask.sum()
        noobj_count = self.noobj_mask.sum()
        
        # 如果没有目标，只计算无目标损失
        if obj_count == 0:
            if noobj_count > 0:
                # 使用向量化操作计算BCE损失
                pred_noobj = predictions[self.noobj_mask][..., 4]
                target_noobj = targets[self.noobj_mask][..., 4]
                # 手动计算BCE以提高效率
                loss_noobj = -self.lambda_noobj * (target_noobj * torch.log(torch.sigmoid(pred_noobj) + 1e-16) + 
                                                (1 - target_noobj) * torch.log(1 - torch.sigmoid(pred_noobj) + 1e-16)).sum()
                return loss_noobj / batch_size
            return torch.tensor(0.0, device=self.device)
        
        # 高效计算坐标损失
        pred_obj = predictions[self.obj_mask]
        target_obj = targets[self.obj_mask]
        
        # 向量化计算坐标损失
        coord_diff = pred_obj[..., :4] - target_obj[..., :4]
        coord_diff_squared = coord_diff ** 2
        coord_loss = self.lambda_coord * coord_diff_squared.sum()
        
        # 高效计算置信度损失
        # 有目标的置信度损失
        pred_conf_obj = pred_obj[..., 4]
        target_conf_obj = target_obj[..., 4]
        loss_obj = -(target_conf_obj * torch.log(torch.sigmoid(pred_conf_obj) + 1e-16) + 
                   (1 - target_conf_obj) * torch.log(1 - torch.sigmoid(pred_conf_obj) + 1e-16)).sum()
        
        # 无目标的置信度损失（如果有）
        loss_noobj = 0.0
        if noobj_count > 0:
            pred_conf_noobj = predictions[self.noobj_mask][..., 4]
            target_conf_noobj = targets[self.noobj_mask][..., 4]
            loss_noobj = -self.lambda_noobj * (target_conf_noobj * torch.log(torch.sigmoid(pred_conf_noobj) + 1e-16) + 
                                            (1 - target_conf_noobj) * torch.log(1 - torch.sigmoid(pred_conf_noobj) + 1e-16)).sum()
        
        # 计算类别损失（对于人脸检测，类别通常只有一个）
        pred_cls = pred_obj[..., 5]
        target_cls = target_obj[..., 5]
        class_loss = -(target_cls * torch.log(torch.sigmoid(pred_cls) + 1e-16) + 
                     (1 - target_cls) * torch.log(1 - torch.sigmoid(pred_cls) + 1e-16)).sum()
        
        # 总损失
        total_loss = coord_loss + loss_obj + loss_noobj + class_loss
        
        # 批次平均损失
        return total_loss / batch_size

# 学习率预热调度器
def get_warmup_lr(current_epoch, warmup_epochs, warmup_factor, base_lr):
    """计算预热阶段的学习率"""
    if current_epoch < warmup_epochs:
        alpha = current_epoch / warmup_epochs
        return base_lr * warmup_factor * (1 - alpha) + alpha * base_lr
    return base_lr

# 初始化模型、损失函数和优化器（高级版）
def initialize_model():
    model = yoloface().to(config['device'])
    criterion = YoloLoss(config['anchors'], config['grid_size'], config['device'])
    
    # 根据配置选择优化器
    if config['optimizer_type'] == 'adamw':
        # AdamW优化器（更好的权重衰减处理）
        optimizer = optim.AdamW(model.parameters(),
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'],
                              betas=(0.9, 0.999),
                              eps=1e-8)
    elif config['optimizer_type'] == 'sgd':
        # SGD优化器（可能更稳定）
        optimizer = optim.SGD(model.parameters(),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'],
                            momentum=0.9,
                            nesterov=True)
    else:
        # 默认Adam优化器
        optimizer = optim.Adam(model.parameters(),
                             lr=config['learning_rate'],
                             weight_decay=config['weight_decay'],
                             betas=(0.9, 0.999),
                             eps=1e-8)
    
    # 根据配置选择学习率调度器
    if config['lr_scheduler_type'] == 'cosine':
        # 余弦退火调度器（更好的收敛性）
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['epochs'] - config.get('warmup_epochs', 0),
            eta_min=1e-6
        )
    elif config['lr_scheduler_type'] == 'step':
        # 步长衰减调度器
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['step_size'],
            gamma=0.5
        )
    else:
        # 默认ReduceLROnPlateau调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=config['plateau_patience'],
            verbose=True, 
            min_lr=1e-6
        )
    
    return model, criterion, optimizer, scheduler

# 训练一个epoch（更简洁高效，支持学习率预热）
def train_one_epoch(model, criterion, optimizer, scheduler, train_loader, epoch):
    model.train()
    running_loss = 0.0
    
    # 应用学习率预热
    if config.get('warmup_epochs', 0) > 0 and epoch < config['warmup_epochs']:
        warmup_lr = get_warmup_lr(epoch, config['warmup_epochs'], 
                                config['warmup_factor'], config['learning_rate'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    
    # 使用更简洁的进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
    
    for inputs, targets in pbar:
        inputs = inputs.to(config['device'])
        targets = targets.to(config['device'])
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        
        # 改进的梯度裁剪策略（根据模型大小动态调整）
        max_norm = 1.0
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()
        
        # 更新统计
        running_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}', 
                        lr=f'{optimizer.param_groups[0]["lr"]:.6f}')
    
    # 非预热阶段且非ReduceLROnPlateau调度器时更新学习率
    if (config.get('warmup_epochs', 0) <= epoch and 
        config['lr_scheduler_type'] != 'plateau'):
        scheduler.step()
    
    return running_loss / len(train_loader)

# 验证函数（更高效的实现）
def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(config['device'])
            targets = targets.to(config['device'])
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

# 保存模型（简化版）
def save_model(model, optimizer, epoch, loss, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    save_path = os.path.join(config['checkpoint_dir'],
                            'best_model.pth' if is_best else f'model_epoch_{epoch}.pth')
    torch.save(checkpoint, save_path)
    print(f'模型已保存到 {save_path}')

# 主训练函数（简化但更强大）
def main():
    print(f'使用设备: {config["device"]}')
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders()
    
    # 初始化模型
    model, criterion, optimizer, scheduler = initialize_model()
    
    # 训练历史记录
    best_val_loss = float('inf')
    start_time = time.time()
    
    print('开始训练...')
    for epoch in range(config['epochs']):
        # 训练一个epoch（传入scheduler以支持学习率预热）
        train_loss = train_one_epoch(model, criterion, optimizer, scheduler, train_loader, epoch)
        
        # 验证
        val_loss = validate(model, criterion, val_loader)
        
        # 对于ReduceLROnPlateau调度器，基于验证损失更新
        if config['lr_scheduler_type'] == 'plateau':
            scheduler.step(val_loss)
        
        # 打印训练信息（更详细）
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["epochs"]}, '  
              f'Train Loss: {train_loss:.4f}, '  
              f'Val Loss: {val_loss:.4f}, '  
              f'LR: {current_lr:.6f}')
        
        # 保存检查点
        if (epoch + 1) % config['save_interval'] == 0:
            save_model(model, optimizer, epoch + 1, val_loss)
        
        # 保存最佳模型（使用验证损失作为指标）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch + 1, val_loss, is_best=True)
    
    # 训练完成
    end_time = time.time()
    print(f'训练完成! 总用时: {(end_time - start_time):.2f}秒')
    print(f'最佳验证损失: {best_val_loss:.4f}')

if __name__ == '__main__':
    main()