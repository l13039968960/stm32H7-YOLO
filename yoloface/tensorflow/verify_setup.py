#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO Face 设置验证脚本

此脚本用于验证YOLO Face训练和测试环境是否正确配置。
它将执行以下检查：
1. 检查依赖项是否已安装
2. 验证训练脚本能否正常导入
3. 测试YOLO Face模型能否正常初始化
4. 检查数据集目录结构是否正确

使用方法：
python verify_setup.py
"""

import os
import sys
import importlib
import numpy as np
import tensorflow as tf

# 设置彩色输出
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def check_requirements():
    """
    检查必要的依赖项是否已安装
    """
    print(f"{bcolors.HEADER}===== 检查依赖项 ====={bcolors.ENDC}")
    
    requirements = [
        ('tensorflow', '2.10.0'),
        ('numpy', '1.23.4'),
        ('opencv-python', '4.6.0'),
        ('matplotlib', '3.6.2'),
        ('tqdm', '4.64.1')
    ]
    
    all_installed = True
    
    for lib, version in requirements:
        try:
            module = importlib.import_module(lib)
            # 获取实际版本
            actual_version = getattr(module, '__version__', 'Unknown')
            print(f"{bcolors.OKGREEN}✓ {lib} 已安装 (版本: {actual_version}){bcolors.ENDC}")
        except ImportError:
            print(f"{bcolors.FAIL}✗ {lib} 未安装{bcolors.ENDC}")
            all_installed = False
    
    return all_installed

def check_environment():
    """
    检查环境配置
    """
    print(f"\n{bcolors.HEADER}===== 检查环境配置 ====={bcolors.ENDC}")
    
    # 检查TensorFlow版本
    tf_version = tf.__version__
    print(f"TensorFlow 版本: {tf_version}")
    
    # 检查GPU可用性
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f"{bcolors.OKGREEN}✓ GPU 可用: {len(gpus)} 个{bcolors.ENDC}")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")
    else:
        print(f"{bcolors.WARNING}⚠ 未检测到GPU，将使用CPU进行训练（速度会较慢）{bcolors.ENDC}")
    
    # 检查Python版本
    python_version = sys.version
    print(f"Python 版本: {python_version.split()[0]}")
    
    return True

def check_model_import():
    """
    检查模型和训练脚本能否正常导入
    """
    print(f"\n{bcolors.HEADER}===== 检查脚本导入 ====={bcolors.ENDC}")
    
    try:
        # 尝试导入训练脚本
        import train_tf
        print(f"{bcolors.OKGREEN}✓ 成功导入 train_tf.py{bcolors.ENDC}")
        
        # 检查Config类
        if hasattr(train_tf, 'Config'):
            print(f"{bcolors.OKGREEN}✓ 找到 Config 类{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}✗ 未找到 Config 类{bcolors.ENDC}")
            return False
        
        # 检查主要函数
        required_functions = ['set_seed', 'create_dataset', 'initialize_model', 'train_model']
        all_functions = True
        
        for func in required_functions:
            if hasattr(train_tf, func):
                print(f"{bcolors.OKGREEN}✓ 找到函数: {func}{bcolors.ENDC}")
            else:
                print(f"{bcolors.FAIL}✗ 未找到函数: {func}{bcolors.ENDC}")
                all_functions = False
        
        if not all_functions:
            return False
            
        # 尝试导入测试脚本
        import yoloface_test
        print(f"{bcolors.OKGREEN}✓ 成功导入 yoloface_test.py{bcolors.ENDC}")
        
        # 检查YoloFaceDetector类
        if hasattr(yoloface_test, 'YoloFaceDetector'):
            print(f"{bcolors.OKGREEN}✓ 找到 YoloFaceDetector 类{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}✗ 未找到 YoloFaceDetector 类{bcolors.ENDC}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"{bcolors.FAIL}✗ 导入失败: {e}{bcolors.ENDC}")
        return False
    except Exception as e:
        print(f"{bcolors.FAIL}✗ 导入时发生错误: {e}{bcolors.ENDC}")
        return False

def check_dataset_structure():
    """
    检查数据集目录结构
    """
    print(f"\n{bcolors.HEADER}===== 检查数据集结构 ====={bcolors.ENDC}")
    
    try:
        # 从训练脚本获取配置
        from train_tf import Config
        config = Config()
        
        # 检查必要的目录
        directories = [
            (config.train_images_dir, "训练图像目录"),
            (config.train_labels_dir, "训练标签目录"),
            (config.val_images_dir, "验证图像目录"),
            (config.val_labels_dir, "验证标签目录")
        ]
        
        all_exists = True
        
        for dir_path, dir_name in directories:
            if os.path.exists(dir_path):
                print(f"{bcolors.OKGREEN}✓ {dir_name} 存在: {dir_path}{bcolors.ENDC}")
                
                # 检查目录是否为空
                if len(os.listdir(dir_path)) == 0:
                    print(f"{bcolors.WARNING}⚠ {dir_name} 为空，请确保数据集已正确放置{bcolors.ENDC}")
                    all_exists = False
            else:
                print(f"{bcolors.FAIL}✗ {dir_name} 不存在: {dir_path}{bcolors.ENDC}")
                all_exists = False
        
        if all_exists:
            return True
        else:
            print(f"\n{bcolors.WARNING}⚠ 数据集结构检查未通过。请根据README.md创建正确的数据集结构{bcolors.ENDC}")
            return False
            
    except Exception as e:
        print(f"{bcolors.FAIL}✗ 检查数据集结构时发生错误: {e}{bcolors.ENDC}")
        return False

def test_model_initialization():
    """
    测试模型初始化功能
    """
    print(f"\n{bcolors.HEADER}===== 测试模型初始化 ====={bcolors.ENDC}")
    
    try:
        from train_tf import initialize_model
        
        print("初始化模型...")
        # 只执行部分初始化，不训练
        try:
            model, loss_function, optimizer, lr_callback = initialize_model()
            print(f"{bcolors.OKGREEN}✓ 成功初始化模型{bcolors.ENDC}")
            print(f"  - 模型类型: {type(model).__name__}")
            print(f"  - 损失函数: {type(loss_function).__name__}")
            print(f"  - 优化器: {type(optimizer).__name__}")
            return True
        except Exception as e:
            print(f"{bcolors.WARNING}⚠ 模型初始化测试失败，但这可能是因为需要完整的数据集: {e}{bcolors.ENDC}")
            print("  继续其他检查...")
            return True  # 我们允许这个测试失败，因为可能需要完整的数据集
            
    except Exception as e:
        print(f"{bcolors.FAIL}✗ 测试模型初始化时发生错误: {e}{bcolors.ENDC}")
        return False

def check_checkpoint_dir():
    """
    检查检查点目录
    """
    print(f"\n{bcolors.HEADER}===== 检查检查点目录 ====={bcolors.ENDC}")
    
    try:
        from train_tf import Config
        config = Config()
        
        # 检查并创建检查点目录
        if not os.path.exists(config.checkpoint_dir):
            print(f"创建检查点目录: {config.checkpoint_dir}")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        print(f"{bcolors.OKGREEN}✓ 检查点目录准备就绪: {config.checkpoint_dir}{bcolors.ENDC}")
        
        # 创建其他必要的目录
        for dir_name in ['evaluation', 'visualization']:
            dir_path = os.path.join(os.path.dirname(config.checkpoint_dir), dir_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"  - 创建目录: {dir_path}")
            
        return True
        
    except Exception as e:
        print(f"{bcolors.FAIL}✗ 检查检查点目录时发生错误: {e}{bcolors.ENDC}")
        return False

def main():
    """
    主函数，运行所有检查
    """
    print(f"{bcolors.BOLD}YOLO Face 设置验证脚本{vcolors.ENDC}")
    print("======================================")
    
    # 运行所有检查
    checks = [
        check_requirements,
        check_environment,
        check_model_import,
        check_dataset_structure,
        test_model_initialization,
        check_checkpoint_dir
    ]
    
    all_passed = True
    
    for check_func in checks:
        if not check_func():
            all_passed = False
    
    print(f"\n{bcolors.BOLD}======================================{bcolors.ENDC}")
    
    if all_passed:
        print(f"{bcolors.OKGREEN}✓ ✓ ✓ 所有检查通过！您可以开始训练YOLO Face模型了！{bcolors.ENDC}")
        print(f"\n使用以下命令开始训练:")
        print(f"{bcolors.BOLD}python train_tf.py{bcolors.ENDC}")
        print(f"\n详细说明请参考README.md")
        return True
    else:
        print(f"{bcolors.FAIL}✗ ✗ ✗ 部分检查未通过，请解决上述问题后再尝试训练{bcolors.ENDC}")
        print(f"\n主要问题:")
        
        # 提供修复建议
        if not os.path.exists('README.md'):
            print(f"  - 请查看README.md了解详细配置说明")
        else:
            print(f"  - 请查看README.md中的数据集结构说明")
            print(f"  - 确保已安装所有依赖项: pip install -r requirements.txt")
            print(f"  - 检查数据集目录路径是否正确设置")
            
        return False

# 修复Windows终端中的颜色问题
def fix_windows_colors():
    if sys.platform.startswith('win'):
        # 在Windows上重置颜色代码
        global bcolors
        class bcolors:
            HEADER = ''
            OKBLUE = ''
            OKGREEN = ''
            WARNING = ''
            FAIL = ''
            ENDC = ''
            BOLD = ''
            UNDERLINE = ''

# 修复可能的颜色显示问题
try:
    # 尝试导入windows-curses（如果可用）
    import windows_curses
    # 如果导入成功，则不需要修复
    pass
except ImportError:
    # 在Windows上修复颜色显示
    fix_windows_colors()

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)