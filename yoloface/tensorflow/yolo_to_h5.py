#! /usr/bin/env python
"""
Darknet模型转Keras模型工具

此脚本用于读取Darknet配置文件(cfg)和权重文件(weights)，然后创建并保存使用TensorFlow后端的Keras模型。
主要用于将YOLO系列模型从Darknet框架转换到Keras框架，便于在TensorFlow环境中使用和进一步开发。

支持的网络层类型：
- 卷积层(convolutional)：支持常规卷积和深度可分离卷积
- 池化层(maxpool)
- 快捷连接(shortcut)：用于实现残差连接
- 上采样(upsample)
- 路由层(route)：用于特征融合
- YOLO检测层(yolo)
"""

# 导入必要的库
import argparse  # 用于解析命令行参数
import configparser  # 用于解析配置文件
import io  # 用于处理字符串IO操作
import os  # 用于文件路径操作
from collections import defaultdict  # 用于创建默认字典

import numpy as np  # 用于数组操作
from tensorflow.keras import backend as K  # Keras后端操作
# 导入需要的Keras层
from tensorflow.keras.layers import (
    Conv2D,  # 二维卷积层
    Input,  # 输入层
    ZeroPadding2D,  # 零填充层
    Add,  # 加法层，用于残差连接
    LeakyReLU,  # LeakyReLU激活函数
    DepthwiseConv2D,  # 深度可分离卷积层
    UpSampling2D,  # 上采样层
    MaxPooling2D,  # 最大池化层
    Concatenate,  # 连接层，用于特征融合
    BatchNormalization  # 批量归一化层
)

from tensorflow.keras.models import Model  # Keras模型类
from tensorflow.keras.regularizers import l2  # L2正则化
# from tensorflow.keras.utils.vis_utils import plot_model as plot  # 用于绘制模型结构图(注释掉)


parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')
parser.add_argument(
    '-p',
    '--plot_model',
    help='Plot generated Keras model and save as image.',
    action='store_true')
parser.add_argument(
    '-w',
    '--weights_only',
    help='Save as Keras weights file instead of model file.',
    action='store_true')

def unique_config_sections(config_file):
    """将配置文件中的所有配置段转换为具有唯一名称

    为了与configparser兼容，为重复的配置段添加唯一后缀。
    Darknet配置文件中可能存在多个相同名称的段（如多个卷积层都叫[convolutional]），
    这个函数会将它们转换为唯一的名称（如[convolutional_0], [convolutional_1]等）
    
    Args:
        config_file: Darknet配置文件路径
        
    Returns:
        output_stream: 包含唯一配置段名称的StringIO对象
    """
    section_counters = defaultdict(int)  # 用于跟踪每个配置段出现的次数
    output_stream = io.StringIO()  # 创建字符串IO对象，用于输出处理后的配置内容
    
    with open(config_file) as fin:
        for line in fin:
            # 检查是否是配置段的开始行
            if line.startswith('['):
                section = line.strip().strip('[]')  # 提取配置段名称
                # 为配置段名称添加计数器后缀
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1  # 增加计数器
                line = line.replace(section, _section)  # 替换原始配置段名称
            output_stream.write(line)  # 写入处理后的行
    
    output_stream.seek(0)  # 将指针移到开头
    return output_stream

# %%
def _main(args):
    """主函数，处理命令行参数并执行模型转换流程
    
    Args:
        args: 命令行参数对象，包含配置文件路径、权重文件路径等信息
    """
    # 扩展用户路径并验证文件格式
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path)
    assert weights_path.endswith(
        '.weights'), '{} is not a .weights file'.format(weights_path)

    # 处理输出路径
    output_path = os.path.expanduser(args.output_path)
    assert output_path.endswith(
        '.h5'), 'output path {} is not a .h5 file'.format(output_path)
    output_root = os.path.splitext(output_path)[0]  # 用于生成图表文件名

    # 加载权重和配置文件
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    
    # 读取Darknet权重文件头信息
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12))
    
    # 根据版本信息读取已处理图像数量
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config.')
    # 获取处理后的唯一配置段
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating Keras model.')
    # 创建输入层，接受任意尺寸的RGB图像
    input_layer = Input(shape=(None, None, 3))
    prev_layer = input_layer  # 前一层初始化为输入层
    all_layers = []  # 保存所有层，用于route和shortcut层引用

    # 获取权重衰减参数，如果不存在则使用默认值5e-4
    weight_decay = float(cfg_parser['net_0']['decay']
                         ) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0  # 层计数器
    out_index = []  # 输出层索引列表
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            groups = int(cfg_parser[section]['groups'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            weights_shape = (size, size, prev_layer_shape[-1]//groups, filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn'
                  if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            if groups == 1:
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            else:
                # 对于深度可分离卷积，调整权重顺序
                conv_weights = np.transpose(conv_weights, [2, 3, 0, 1])
            
            # 根据是否使用批量归一化，准备权重列表
            # 如果使用批量归一化，bias由BatchNormalization层处理，所以只传入卷积权重
            # 否则，同时传入卷积权重和偏置
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            # 处理激活函数
            act_fn = None  # 初始化激活函数为None
            if activation == 'leaky':
                # LeakyReLU将在后面单独添加，因为它是一个高级激活函数
                pass
            elif activation != 'linear':
                # 如果激活函数不是linear或leaky，则报错
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            # 创建卷积层
            if stride>1:
                # Darknet在步长大于1时使用左上角填充而非'same'模式
                prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
            
            if groups==1:
                # 常规卷积层
                conv_layer = (Conv2D(
                    filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),  # 添加L2正则化
                    use_bias=not batch_normalize,  # 如果使用批量归一化，则不需要偏置
                    weights=conv_weights,  # 设置权重
                    activation=act_fn,  # 设置激活函数
                    padding=padding))(prev_layer)
            else:
                # 深度可分离卷积层
                conv_layer = (DepthwiseConv2D(
                    kernel_size=(size, size),
                    strides=(stride, stride),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=act_fn,
                    padding=padding))(prev_layer)

            # 添加批量归一化层
            if batch_normalize:
                conv_layer = (BatchNormalization(
                    weights=bn_weight_list))(conv_layer)
            
            prev_layer = conv_layer  # 更新前一层

            # 根据激活函数类型处理输出
            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                # 添加LeakyReLU激活函数，负斜率为0.1
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'):
            # 处理路由层，用于特征融合或跳连接
            # 获取需要路由的层的索引
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            
            if len(layers) > 1:
                # 多个层时，进行通道拼接
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                # 单个层时，直接引用该层的输出
                skip_layer = layers[0]  # 只有一个层需要路由
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            # 处理最大池化层
            size = int(cfg_parser[section]['size'])  # 池化核大小
            stride = int(cfg_parser[section]['stride'])  # 池化步长
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            # 处理快捷连接（残差连接）
            index = int(cfg_parser[section]['from'])  # 要连接的层的索引
            activation = cfg_parser[section]['activation']  # 激活函数
            # 目前仅支持线性激活
            assert activation == 'linear', 'Only linear activation supported.'
            # 创建加法层，将当前层输出与指定层输出相加
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            # 处理上采样层
            stride = int(cfg_parser[section]['stride'])  # 上采样步长
            # 目前仅支持步长为2的上采样
            assert stride == 2, 'Only stride=2 supported.'
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('yolo'):
            # 处理YOLO检测层
            # 记录输出层的索引
            out_index.append(len(all_layers)-1)
            # 在all_layers中添加None占位
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            # 忽略net配置段，因为这些参数主要用于训练，转换过程不需要
            pass

        else:
            # 如果遇到不支持的层类型，则报错
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # 创建并保存模型
    # 如果没有明确的输出层，使用最后一个层作为输出
    if len(out_index)==0: out_index.append(len(all_layers)-1)
    
    # 创建Keras模型，指定输入和输出
    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    print(model.summary())  # 打印模型结构摘要
    
    # 根据参数决定保存方式
    if args.weights_only:
        # 只保存权重
        model.save_weights('{}'.format(output_path))
        print('Saved Keras weights to {}'.format(output_path))
    else:
        # 保存完整模型（结构+权重）
        model.save('{}'.format(output_path))
        print('Saved Keras model to {}'.format(output_path))

    # 检查是否所有权重都已读取
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count +
                                                       remaining_weights))
    
    # 如果有未使用的权重，打印警告
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))

    # 如果启用了绘制模型结构选项
    if args.plot_model:
        # 注意：plot函数已被注释掉，需要取消导入注释才能使用
        # plot(model, to_file='{}.png'.format(output_root), show_shapes=True)
        # print('Saved model plot to {}.png'.format(output_root))
        print('Model plotting is disabled. Please uncomment the import statement to enable it.')


if __name__ == '__main__':
    # 主函数入口，解析命令行参数并执行转换
    _main(parser.parse_args())