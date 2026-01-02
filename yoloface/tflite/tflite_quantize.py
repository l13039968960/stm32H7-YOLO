#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorFlow模型量化为INT8 TFLite模型工具

本脚本用于将TensorFlow冻结的PB模型量化为INT8精度的TFLite模型，以减小模型体积并加速推理。
主要功能：
- 加载小型数据集用于量化校准
- 定义代表性数据集生成器函数
- 配置TFLite转换器进行INT8量化
- 执行转换并保存量化后的模型

适用于将YOLOFace等模型转换为在移动设备或嵌入式设备上高效运行的格式。
"""

# 导入必要的库
import tensorflow.compat.v1 as tf  # 使用TensorFlow 1.x兼容模式
from tensorflow import keras  # 导入Keras模块
import numpy as np  # 用于数组操作
import os  # 用于文件和目录操作
import cv2  # OpenCV库，用于图像处理

# 获取小型数据集目录中的所有图像文件
img_lists = []
for root, dirs, files in os.walk("small_dataset"):
    img_lists = files  # 将所有文件名保存到img_lists


def representative_dataset_gen():
    """
    生成代表性数据集的函数，用于量化校准
    
    这个函数遍历小型数据集目录中的所有图像，对图像进行预处理后，
    生成用于TFLite量化校准的代表性数据集。量化器将使用这些样本
    来确定量化参数（如最大值、最小值等）。
    
    Yields:
        list: 包含预处理后图像数据的列表
    """
    for img in img_lists:
        # 读取图像
        img = cv2.imread('small_dataset/'+img)
        # 将BGR格式转换为RGB格式（因为TensorFlow模型通常期望RGB输入）
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 获取图像原始尺寸
        H, W, _ = img.shape
        # 计算缩放比例（原始图像到模型输入尺寸）
        w_scale = W/56.
        h_scale = H/56.
        # 调整图像大小为模型输入尺寸 (56, 56)
        input = cv2.resize(input, (56, 56))
        # 添加批次维度，使其形状为 (1, 56, 56, 3)
        input = input[np.newaxis,:,:,:]
        # 归一化像素值到 [0, 1] 范围
        input = input/255.
        # 使用实际图像作为校准数据，而不是随机数据
        # data = np.random.rand(1, 56, 56, 3)
        yield [input.astype(np.float32)]  # 返回float32类型的输入数据


# 定义TFLite转换器，从冻结的TensorFlow图创建
# 参数说明：
# - "model.pb": 输入的冻结模型文件路径
# - ["Input"]: 输入节点名称列表
# - ["Identity"]: 输出节点名称列表
# - {"Input":[1,56,56,3]}: 输入张量形状字典
converter = tf.lite.TFLiteConverter.from_frozen_graph("model.pb", ["Input"], ["Identity"], {"Input":[1,56,56,3]})

# 定义量化配置
# 设置代表性数据集生成器，用于校准量化参数
converter.representative_dataset = representative_dataset_gen
# 启用默认优化，包括量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# 指定支持的操作集为仅INT8内置操作
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# 指定支持的数据类型为INT8
converter.target_spec.supported_types = [tf.int8]
# 设置推理时输入数据类型为INT8
converter.inference_input_type = tf.int8
# 设置推理时输出数据类型为INT8
converter.inference_output_type = tf.int8

# 以下是其他可能的配置选项，当前被注释掉
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
# converter.allow_custom_ops=True
 
# 这些是较旧版本的配置方法，当前使用的是更新的API
# converter.inference_type = tf.int8    #tf.lite.constants.QUANTIZED_UINT8
# input_arrays = converter.get_input_arrays()
# converter.quantized_input_stats = {input_arrays[0]: (0, 0)} # mean, std_dev
# converter.default_ranges_stats = (-128, 127)


# 执行模型转换
print("开始转换模型...")
quantize_model = converter.convert()

# 保存量化后的TFLite模型
open("yoloface_int8.tflite", "wb").write(quantize_model)
print("INT8量化模型已保存为 'yoloface_int8.tflite'")

# 以下是测试代码，当前被注释掉
# data = np.random.rand(1, 56, 56, 3).astype(np.float32)
# output = loaded_model(data)
# print(output.shape)