#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO Face 测试脚本

此脚本演示如何使用训练好的YOLO Face模型进行人脸检测推理。
支持单个图像检测、批量检测和视频检测模式。

使用说明:
1. 确保已训练完成或下载了预训练模型
2. 指定输入图像/视频路径
3. 运行脚本查看检测结果
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class YoloFaceDetector:
    """
    YOLO Face检测器类，封装模型加载、预处理、推理和后处理功能
    """
    
    def __init__(self, model_path, input_shape=(416, 416), confidence_threshold=0.5, iou_threshold=0.4):
        """
        初始化YOLO Face检测器
        
        参数：
            model_path: 模型路径
            input_shape: 模型输入尺寸，默认(416, 416)
            confidence_threshold: 置信度阈值，默认0.5
            iou_threshold: IoU阈值，用于非极大值抑制，默认0.4
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # 加载模型
        self.model = self._load_model()
        print(f"成功加载模型: {model_path}")
    
    def _load_model(self):
        """
        加载TensorFlow模型
        
        返回：
            加载的模型实例
        """
        try:
            return tf.keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        预处理输入图像
        
        参数：
            image: 原始图像
            
        返回：
            预处理后的图像，可直接输入模型
        """
        # 复制图像以避免修改原始图像
        img_copy = image.copy()
        
        # 调整图像大小
        img_resized = cv2.resize(img_copy, self.input_shape)
        
        # 归一化图像
        img_normalized = img_resized / 255.0
        
        # 添加批次维度
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized.shape[:2]
    
    def postprocess_detections(self, predictions, original_size, image_size):
        """
        后处理检测结果
        
        参数：
            predictions: 模型预测结果
            original_size: 原始图像尺寸 (高度, 宽度)
            image_size: 输入模型的图像尺寸 (高度, 宽度)
            
        返回：
            过滤后的检测框列表，每个元素为 [x1, y1, x2, y2, confidence]
        """
        detections = []
        
        # 解析预测结果
        # YOLO输出格式通常为 [batch_size, num_boxes, 5] 其中5为 [x, y, w, h, confidence]
        boxes = predictions[0]  # 假设predictions是一个列表，第一个元素是边界框
        
        # 如果预测结果是多输出，需要根据具体网络结构调整
        if isinstance(boxes, list):
            boxes = np.concatenate(boxes, axis=0)
        
        # 缩放比例
        h_scale = original_size[0] / image_size[0]
        w_scale = original_size[1] / image_size[1]
        
        # 处理每个检测框
        for box in boxes:
            # 提取边界框坐标和置信度
            # 根据具体的输出格式调整索引
            try:
                x, y, w, h, conf = box[:5]
            except ValueError:
                # 如果输出格式不同，尝试其他索引
                continue
            
            # 过滤低置信度检测框
            if conf < self.confidence_threshold:
                continue
            
            # 转换中心点坐标为左上角和右下角坐标
            x1 = int((x - w / 2) * w_scale)
            y1 = int((y - h / 2) * h_scale)
            x2 = int((x + w / 2) * w_scale)
            y2 = int((y + h / 2) * h_scale)
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_size[1] - 1, x2)
            y2 = min(original_size[0] - 1, y2)
            
            # 添加到检测结果列表
            detections.append([x1, y1, x2, y2, float(conf)])
        
        # 应用非极大值抑制
        detections = self.non_max_suppression(detections)
        
        return detections
    
    def non_max_suppression(self, boxes):
        """
        非极大值抑制 (NMS)
        
        参数：
            boxes: 检测框列表，每个元素为 [x1, y1, x2, y2, confidence]
            
        返回：
            应用NMS后的检测框列表
        """
        if not boxes:
            return []
        
        # 转换为numpy数组
        boxes = np.array(boxes)
        
        # 提取坐标和置信度
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        conf = boxes[:, 4]
        
        # 计算面积
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # 按置信度排序
        order = conf.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算交集
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            
            intersection = w * h
            union = area[i] + area[order[1:]] - intersection
            
            # 计算IoU
            iou = intersection / union
            
            # 保留IoU小于阈值的框
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]
        
        return boxes[keep].tolist()
    
    def detect(self, image):
        """
        执行完整的检测过程
        
        参数：
            image: 输入图像
            
        返回：
            检测框列表
        """
        original_size = image.shape[:2]
        
        # 预处理图像
        processed_image, image_size = self.preprocess_image(image)
        
        # 推理
        predictions = self.model.predict(processed_image, verbose=0)
        
        # 后处理
        detections = self.postprocess_detections(predictions, original_size, image_size)
        
        return detections
    
    def visualize_detections(self, image, detections, save_path=None):
        """
        可视化检测结果
        
        参数：
            image: 原始图像
            detections: 检测框列表
            save_path: 保存路径，如果为None则不保存
            
        返回：
            可视化后的图像
        """
        # 复制图像以避免修改原始图像
        viz_image = image.copy()
        
        # 绘制检测框
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            
            # 绘制边界框
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加置信度标签
            label = f"Face: {confidence:.2f}"
            cv2.putText(viz_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 添加检测信息
        info_text = f"检测到 {len(detections)} 个人脸"
        cv2.putText(viz_image, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 保存图像
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            cv2.imwrite(save_path, viz_image)
            print(f"检测结果已保存到: {save_path}")
        
        return viz_image
    
    def detect_image(self, image_path, output_dir=None):
        """
        检测单个图像
        
        参数：
            image_path: 图像路径
            output_dir: 输出目录
            
        返回：
            检测框列表
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return []
        
        # 转换为RGB格式（OpenCV默认读取为BGR）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测
        detections = self.detect(image_rgb)
        
        # 可视化
        if output_dir:
            # 创建保存路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_detected_{timestamp}{ext}")
            
            # 可视化并保存
            viz_image = self.visualize_detections(image_rgb, detections, output_path)
            
            # 使用matplotlib显示结果
            plt.figure(figsize=(10, 8))
            plt.imshow(viz_image)
            plt.title(f"人脸检测结果 ({image_path})")
            plt.axis('off')
            plt.show()
        else:
            # 仅显示结果
            viz_image = self.visualize_detections(image_rgb, detections)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(viz_image)
            plt.title(f"人脸检测结果 ({image_path})")
            plt.axis('off')
            plt.show()
        
        return detections
    
    def detect_video(self, video_path, output_path=None, display=True, fps=30):
        """
        检测视频中的人脸
        
        参数：
            video_path: 视频路径
            output_path: 输出视频路径
            display: 是否实时显示
            fps: 输出视频帧率
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print(f"开始处理视频，共 {total_frames} 帧")
        
        try:
            while cap.isOpened():
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 检测人脸
                detections = self.detect(frame)
                
                # 可视化结果
                viz_frame = self.visualize_detections(frame, detections)
                
                # 写入输出视频
                if writer:
                    writer.write(viz_frame)
                
                # 显示结果
                if display:
                    cv2.imshow("Face Detection", viz_frame)
                    
                    # 按 'q' 退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 显示进度
                if frame_count % 10 == 0:
                    print(f"处理进度: {frame_count}/{total_frames}")
        finally:
            # 释放资源
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        print(f"视频处理完成，共处理 {frame_count} 帧")
    
    def detect_images_batch(self, image_paths, output_dir):
        """
        批量处理多个图像
        
        参数：
            image_paths: 图像路径列表
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        print(f"开始批量处理，共 {len(image_paths)} 个图像")
        
        for i, image_path in enumerate(image_paths):
            print(f"处理图像 {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                # 检测图像
                detections = self.detect_image(image_path, output_dir)
                results[image_path] = detections
                
            except Exception as e:
                print(f"处理图像 {image_path} 失败: {e}")
                results[image_path] = []
        
        # 生成报告
        report_path = os.path.join(output_dir, "detection_report.txt")
        with open(report_path, 'w') as f:
            f.write("人脸检测批量处理报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"处理图像总数: {len(image_paths)}\n")
            
            # 统计检测到人脸的图像数量
            detected_count = sum(1 for detections in results.values() if len(detections) > 0)
            f.write(f"检测到人脸的图像数: {detected_count}\n\n")
            
            # 详细结果
            f.write("详细结果:\n")
            for image_path, detections in results.items():
                f.write(f"图像: {image_path}\n")
                f.write(f"检测到人脸数量: {len(detections)}\n")
                
                if detections:
                    f.write("检测框详情: [x1, y1, x2, y2, confidence]\n")
                    for j, detection in enumerate(detections):
                        f.write(f"  人脸 {j+1}: {detection}\n")
                
                f.write("\n")
        
        print(f"批量处理完成，报告已保存到: {report_path}")
        return results

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLO Face 测试脚本')
    
    # 模式选择
    parser.add_argument('--mode', choices=['image', 'video', 'batch'], default='image',
                        help='检测模式: image (单图像), video (视频), batch (批量图像)')
    
    # 通用参数
    parser.add_argument('--model', required=True,
                        help='模型路径')
    parser.add_argument('--input', required=True,
                        help='输入路径: 图像路径、视频路径或包含图像的目录')
    parser.add_argument('--output', default='./detection_results',
                        help='输出目录或文件路径')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.4,
                        help='IoU阈值')
    
    # 视频特定参数
    parser.add_argument('--display', action='store_true',
                        help='是否显示处理后的视频')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 创建检测器实例
    detector = YoloFaceDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    # 根据模式执行不同的检测逻辑
    if args.mode == 'image':
        # 单图像检测
        detector.detect_image(args.input, args.output)
    
    elif args.mode == 'video':
        # 视频检测
        detector.detect_video(args.input, args.output, args.display)
    
    elif args.mode == 'batch':
        # 批量图像检测
        if os.path.isdir(args.input):
            # 获取目录中的所有图像
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            image_paths = []
            
            for root, _, files in os.walk(args.input):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(os.path.join(root, file))
            
            if image_paths:
                detector.detect_images_batch(image_paths, args.output)
            else:
                print(f"目录中未找到图像: {args.input}")
        else:
            print(f"输入必须是目录: {args.input}")
    
    print("检测完成！")

if __name__ == '__main__':
    main()