# 导入必要的库
import torch                # PyTorch深度学习框架
import torch.nn as nn       # PyTorch神经网络模块
import torchvision          # PyTorch计算机视觉库
from itertools import chain  # 用于链式操作
import cv2                  # OpenCV库，用于图像处理
import numpy as np          # NumPy库，用于数学计算

def Conv2D(in_channel, out_channel, filter_size, stride, pad, is_relu=True, groups=1):
    """
    创建2D卷积层，包含卷积、批归一化和可选的激活函数
    
    参数：
    in_channel: 输入通道数
    out_channel: 输出通道数
    filter_size: 卷积核大小
    stride: 步长
    pad: 填充大小
    is_relu: 是否添加LeakyReLU激活函数
    groups: 分组卷积的组数，默认为1（标准卷积）
    
    返回：
    nn.Sequential: 包含卷积层和批归一化层的序列模型，可选包含激活函数
    """
    if is_relu:
        return nn.Sequential(
            # 卷积层，不使用偏置（由批归一化处理）
            nn.Conv2d(in_channel, out_channel, filter_size, stride, padding=pad, bias=False, groups=groups),
            # 批归一化层，加速训练和提高稳定性
            nn.BatchNorm2d(out_channel),
            # LeakyReLU激活函数，参数0.1表示负区间的斜率
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channel, out_channel, filter_size, stride, padding=pad, bias=False, groups=groups),
            # 批归一化层
            nn.BatchNorm2d(out_channel)
        )

def depthwise_conv(in_channel, hidden_channel, out_channel, stride1=1, stride2=1, relu=False):
    """
    创建深度可分离卷积块，包含深度卷积和逐点卷积
    
    深度可分离卷积将标准卷积分解为深度卷积（depthwise）和逐点卷积（pointwise），
    可以显著减少计算量和参数量，同时保持相似的性能
    
    参数：
    in_channel: 输入通道数
    hidden_channel: 中间通道数，也是深度卷积的组数
    out_channel: 输出通道数
    stride1: 深度卷积的步长
    stride2: 逐点卷积的步长
    relu: 最后一层是否使用激活函数
    
    返回：
    nn.Sequential: 包含深度卷积和逐点卷积的序列模型
    """
    return nn.Sequential(
        # 深度卷积（depthwise convolution）：每个输入通道单独用一个卷积核处理
        Conv2D(in_channel, hidden_channel, 3, stride1, 1, is_relu=True, groups=hidden_channel),
        # 逐点卷积（pointwise convolution）：1x1卷积核，用于通道数调整和特征融合
        Conv2D(hidden_channel, out_channel, 1, stride2, 0, is_relu=relu)
    )

class yoloface(nn.Module):
    """
    YOLO Face模型类，基于YOLO（You Only Look Once）架构的人脸检测模型
    
    该模型使用深度可分离卷积构建轻量级网络，专为实时人脸检测优化
    使用3个锚框（anchors）检测不同大小的人脸
    """
    def __init__(self):
        """
        初始化YOLO Face模型，定义网络层结构
        
        模型架构基于简化版的YOLOv3，使用深度可分离卷积减少参数量和计算量
        包含特征提取网络和检测层
        """
        super(yoloface, self).__init__()
        # 初始卷积层，将3通道输入（RGB图像）转换为8通道特征图
        self.conv1 = Conv2D(3, 8, 3, 2, 1, is_relu=True)# 0
        # 深度可分离卷积块，特征图缩小，通道数降至4
        self.conv2 = depthwise_conv(8, 8, 4)# 1-2
        # 1x1卷积，通道数调整至18
        self.conv3 = Conv2D(4, 18, 1, 1, 0, is_relu=True)# 3
        # 带步长的深度可分离卷积，特征图尺寸减半
        self.conv4 = depthwise_conv(18, 18, 6, stride1=2)# 4-5
        # 1x1卷积，通道数调整至36
        self.conv5 = Conv2D(6, 36, 1, 1, 0, is_relu=True)# 6
        # 深度可分离卷积块
        self.conv6 = depthwise_conv(36, 36, 6)# 7-8
        # 用于shortcut连接的调整层
        self.conv7 = Conv2D(6, 18, 1, 1, 0, is_relu=True)# 10
        # 最大池化层，用于特征融合
        self.maxpool1 = nn.MaxPool2d(kernel_size=8, stride=2, padding=int((8 - 1) // 2))# 12
        # 特征融合后的通道调整
        self.conv8 = Conv2D(36, 24, 1, 1, 0, is_relu=True)# 14
        # 带步长的深度可分离卷积，特征图进一步缩小
        self.conv9 = depthwise_conv(24, 24, 8, stride1=2)# 15-16
        # 1x1卷积，通道数调整至40
        self.conv10 = Conv2D(8, 40, 1, 1, 0, is_relu=True)# 17
        # 深度可分离卷积块
        self.conv11 = depthwise_conv(40, 40, 8)# 18-19
        # 用于shortcut连接的调整层
        self.conv12 = Conv2D(8, 40, 1, 1, 0, is_relu=True)# 21
        # 深度可分离卷积块
        self.conv13 = depthwise_conv(40, 40, 8)# 22-23
        # 用于shortcut连接的调整层
        self.conv14 = Conv2D(8, 24, 1, 1, 0, is_relu=True)# 25
        # 最大池化层，用于特征融合
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=int((4 - 1) // 2))# 27
        # 特征融合后的通道调整
        self.conv15 = Conv2D(48, 40, 1, 1, 0, is_relu=True)# 29
        # 深度可分离卷积块，保持激活
        self.conv16 = depthwise_conv(40, 40, 32, relu=True)# 30-31
        # 输出层，将特征图转换为预测结果（18通道 = 3个锚框 × 6个参数）
        self.conv17 = nn.Conv2d(32, 18, 1, 1, padding=0, bias=True)# 32
        # 检测层，处理原始预测结果，转换为边界框坐标和置信度
        self.detector = yolo_layer([[9,14], [12,17], [22,21]])
    
    def forward(self, input):
        """
        模型前向传播函数，定义数据在网络中的流动路径
        
        参数：
        input: 输入张量，形状为 [batch_size, 3, height, width]，表示RGB图像
        
        返回：
        通过检测层处理后的边界框预测结果，包含坐标和置信度
        """
        # 第一层：初始卷积和深度可分离卷积
        conv3 = self.conv3(self.conv2(self.conv1(input)))
        
        # 特征提取路径1：带shortcut连接
        conv4 = self.conv4(conv3)               # 下采样
        conv6 = self.conv6(self.conv5(conv4))   # 特征处理
        conv6 = conv4 + conv6                   # shortcut连接（残差连接）
        conv7 = self.conv7(conv6)               # 通道调整
        
        # 特征融合：池化特征与路径1特征
        maxpool1 = self.maxpool1(conv3)         # 对早期特征进行池化
        route1 = torch.cat([maxpool1, conv7], axis=1)  # 特征拼接
        conv8 = self.conv8(route1)              # 通道调整
        
        # 特征提取路径2：带shortcut连接
        conv9 = self.conv9(conv8)               # 下采样
        conv11 = self.conv11(self.conv10(conv9)) # 特征处理
        conv11 = conv9 + conv11                 # shortcut连接
        
        # 特征提取路径3：带shortcut连接
        conv13 = self.conv13(self.conv12(conv11)) # 特征处理
        conv13 = conv11 + conv13                # shortcut连接
        conv14 = self.conv14(conv13)            # 通道调整
        
        # 特征融合：池化特征与路径3特征
        maxpool2 = self.maxpool2(conv8)         # 对中期特征进行池化
        route2 = torch.cat([maxpool2, conv14], axis=1)  # 特征拼接
        
        # 最终预测头
        conv17 = self.conv17(self.conv16(self.conv15(route2)))
        
        # 以下是可视化调试代码（已注释）
        # tmp_input = conv17.numpy()
        # print(tmp_input.shape)
        # for i in range(8):
        #     tmp = np.abs(tmp_input[0,i,:,:])
        #     tmp *= 255/np.max(tmp)
        #     cv2.imshow('imgg%d'%i, tmp.astype(np.uint8))
        #     cv2.waitKey(0)
        # return conv17
        
        # 通过检测层处理预测结果，转换为边界框
        return self.detector(conv17, input.shape[2])
    
    def load_conv_bn_weights(self, weights, ptr, conv_layer, bn_layer):
        """
        从预训练权重中加载卷积层和批归一化层的权重
        
        参数：
        weights: 包含所有权重的NumPy数组
        ptr: 当前权重加载位置的指针
        conv_layer: 要加载权重的卷积层
        bn_layer: 要加载权重的批归一化层
        
        返回：
        更新后的指针位置
        """
        # 获取批归一化层参数数量（偏置、权重、运行均值、运行方差的数量相同）
        num_b = bn_layer.bias.numel()  # Number of biases
        
        # 加载批归一化层的偏置（bias）
        bn_b = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_b)
        ptr += num_b
        
        # 加载批归一化层的权重（weight）
        bn_w = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_w)
        ptr += num_b
        
        # 加载批归一化层的运行均值（running mean）
        bn_rm = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_rm)
        ptr += num_b
        
        # 加载批归一化层的运行方差（running variance）
        bn_rv = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_rv)
        ptr += num_b

        # 加载卷积层的权重
        num_w = conv_layer.weight.numel()
        conv_w = torch.from_numpy(
            weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_w)
        ptr += num_w

        return ptr
    
    def load_darknet_weights(self, weights_path):
        """
        从Darknet格式的权重文件中解析并加载权重到模型
        
        Darknet是一个开源神经网络框架，yoloface-50k.weights文件使用这种格式存储
        
        参数：
        weights_path: Darknet权重文件的路径
        """

        # 打开权重文件
        with open(weights_path, "rb") as f:
            # 前5个整数是文件头信息
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # 保存文件头信息，以便在保存权重时使用
            self.seen = header[3]  # 训练过程中看到的图像数量
            weights = np.fromfile(f, dtype=np.float32)  # 其余部分是权重数据

        # 初始化指针，用于跟踪当前加载位置
        ptr = 0
        
        # 依次加载模型各层的权重
        # 按照模型定义的顺序加载每层权重
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv1[0], self.conv1[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv2[0][0], self.conv2[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv2[1][0], self.conv2[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv3[0], self.conv3[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv4[0][0], self.conv4[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv4[1][0], self.conv4[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv5[0], self.conv5[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv6[0][0], self.conv6[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv6[1][0], self.conv6[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv7[0], self.conv7[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv8[0], self.conv8[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv9[0][0], self.conv9[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv9[1][0], self.conv9[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv10[0], self.conv10[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv11[0][0], self.conv11[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv11[1][0], self.conv11[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv12[0], self.conv12[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv13[0][0], self.conv13[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv13[1][0], self.conv13[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv14[0], self.conv14[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv15[0], self.conv15[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv16[0][0], self.conv16[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv16[1][0], self.conv16[1][1])
        
        # 加载最后一层卷积的偏置（注意该层不使用批归一化）
        num_b = self.conv17.bias.numel()
        conv_b = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(self.conv17.bias)
        self.conv17.bias.data.copy_(conv_b)
        ptr += num_b
        
        # 加载最后一层卷积的权重
        num_w = self.conv17.weight.numel()
        conv_w = torch.from_numpy(
            weights[ptr: ptr + num_w]).view_as(self.conv17.weight)
        self.conv17.weight.data.copy_(conv_w)
        ptr += num_w
        

class yolo_layer(nn.Module):
    """
    YOLO检测层，将网络输出的原始特征图转换为边界框坐标和置信度
    
    该层实现了YOLO算法的核心解码过程，将网络预测的偏移量转换为实际图像上的
    边界框坐标，同时应用激活函数处理置信度
    """

    def __init__(self, anchors):
        """
        初始化YOLO检测层
        
        参数：
        anchors: 锚框尺寸列表，每个锚框表示为[宽度, 高度]
        """
        super(yolo_layer, self).__init__()
        self.num_anchors = len(anchors)  # 锚框数量
        self.no = 6  # 每个锚框输出的参数数量：[x, y, w, h, conf, cls]，但这里仅用于人脸检测
        self.grid = torch.zeros(1)  # 网格坐标，将在forward中初始化

        # 将锚框转换为张量并注册为缓冲区（不会被视为模型参数）
        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(-1, 1, 1, 2))
        self.stride = None  # 特征图与原始图像之间的缩放步长

    def forward(self, x, img_size):
        """
        处理网络输出的特征图，生成边界框预测
        
        参数：
        x: 网络输出的特征图，形状为 [batch_size, num_anchors*no, grid_h, grid_w]
        img_size: 输入图像的尺寸
        
        返回：
        处理后的边界框预测，形状为 [num_predictions, 6]，每行包含 [x, y, w, h, conf, cls]
        """
        # 移除批次维度（假设batch_size=1）
        x = x.squeeze(0)
        # 计算特征图到原始图像的缩放步长
        stride = img_size // x.shape[1]
        self.stride = stride
        
        # 重塑特征图：从 [18, 7, 7] 转换为 [3, 7, 7, 6]
        _, ny, nx = x.shape  # 获取网格尺寸
        x = x.view(self.num_anchors, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()

        # 生成网格坐标
        self.grid = self._make_grid(nx, ny).to(x.device)

        # 解码边界框坐标
        # 中心坐标(x,y)：应用sigmoid函数获取偏移量，加上网格坐标，再乘以步长
        x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
        # 宽度和高度(w,h)：应用指数函数，乘以锚框尺寸
        x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
        # 置信度应用sigmoid函数归一化到0-1范围
        x[..., 4:] = x[..., 4:].sigmoid()
        
        # 将预测结果展平为 [num_anchors*grid_h*grid_w, 6]
        x = x.view(-1, self.no)

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """
        生成网格坐标矩阵
        
        参数：
        nx: 网格宽度
        ny: 网格高度
        
        返回：
        网格坐标张量，形状为 [1, ny, nx, 2]
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        # 堆叠x和y坐标，形状变为 [1, ny, nx, 2]
        return torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()

def xywh2xyxy(x):
    """
    将边界框格式从 [中心x, 中心y, 宽度, 高度] 转换为 [左上x, 左上y, 右下x, 右下y]
    
    参数：
    x: 输入张量，包含边界框坐标，格式为 [中心x, 中心y, 宽度, 高度]
    
    返回：
    y: 转换后的张量，格式为 [左上x, 左上y, 右下x, 右下y]
    """
    # 创建与输入相同形状的新张量
    y = x.new(x.shape)
    # 计算左上角x坐标 = 中心x - 宽度/2
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    # 计算左上角y坐标 = 中心y - 高度/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    # 计算右下角x坐标 = 中心x + 宽度/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    # 计算右下角y坐标 = 中心y + 高度/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.25):
    """
    非极大值抑制函数，过滤低置信度的边界框
    
    注意：这是一个简化版的NMS，仅根据置信度阈值过滤，没有实现完整的NMS算法（IoU阈值）
    
    参数：
    prediction: 模型预测结果，每行包含 [x, y, w, h, conf, cls]
    conf_thres: 置信度阈值，低于此值的边界框会被过滤掉
    
    返回：
    过滤后的边界框，格式为 [左上x, 左上y, 右下x, 右下y]
    如果没有符合条件的边界框，返回None
    """
    # 过滤掉置信度低于阈值的预测框
    x = prediction[prediction[..., 4] > conf_thres]

    # 检查是否有符合条件的边界框
    if not x.shape[0]:
        return None
    
    # 将边界框格式从 [中心x, 中心y, 宽度, 高度] 转换为 [左上x, 左上y, 右下x, 右下y]
    box = xywh2xyxy(x[:, :4])

    return box

# ===================================
# 主函数：加载模型并进行人脸检测演示
# ===================================

# 创建YOLO Face模型实例
model = yoloface()

# 加载预训练权重
model.load_darknet_weights('yoloface-50k.weights')

# 读取测试图像
img = cv2.imread('small_dataset/img_82.jpg')

# 将BGR格式（OpenCV默认）转换为RGB格式（模型输入要求）
input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 获取原始图像尺寸
H, W, _ = img.shape

# 计算缩放比例，因为模型输入尺寸固定为56x56
w_scale = W/56.  # 宽度缩放因子
h_scale = H/56.  # 高度缩放因子

# 调整输入图像尺寸为模型所需的56x56
input = cv2.resize(input, (56, 56))

# 转换为PyTorch张量：
# 1. 从numpy数组转为torch张量
# 2. 调整通道顺序（HWC -> CHW）
# 3. 添加批次维度
# 4. 转换为浮点型
input = torch.from_numpy(input).permute(2, 0, 1).float().unsqueeze(0)

# 图像归一化，像素值从0-255缩放到0-1
input = input/255.

# 执行模型推理
# 使用torch.no_grad()上下文管理器，关闭梯度计算以提高推理速度并减少内存使用
with torch.no_grad():
    output = model(input)

# 以下是将模型导出为ONNX格式的代码（已注释）
# torch.onnx.export(model, input, 'yoloface.onnx',
#                     export_params=True,      # 导出训练好的权重参数
#                     verbose=True,           # 打印详细信息
#                     input_names=['input'],  # 输入节点名称
#                     output_names=["output"],  # 输出节点名称
#                     # opset_version=11,      # ONNX算子集版本
#                     training=False)          # 导出为推理模式

# 应用非极大值抑制过滤低置信度预测框，这里使用较高的置信度阈值0.7
output = non_max_suppression(output, 0.7)

# 如果检测到人脸
if output is not None:
    # 遍历每个检测到的人脸
    for detect in output:
        # 将边界框坐标从模型输出尺寸（56x56）缩放回原始图像尺寸
        detect[[0,2]] *= w_scale  # 缩放左右边界
        detect[[1,3]] *= h_scale  # 缩放上下边界
        
        # 转换为整数坐标并转为numpy数组
        detect = detect.numpy().astype(np.int32)
        
        # 在原始图像上绘制红色矩形框标注人脸
        # 参数：图像，左上角坐标，右下角坐标，颜色(BGR)，线条粗细
        cv2.rectangle(img, (detect[0], detect[1]), (detect[2], detect[3]), (0,0,255), 2)

# 显示检测结果图像
cv2.imshow('img', img)

# 等待用户按键，按任意键关闭窗口
cv2.waitKey(0)
