"""
Mini-ImageNet数据集检查脚本
用于验证下载的数据集是否符合论文要求
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

def check_dataset_structure(data_dir):
    """
    检查数据集目录结构
    
    Args:
        data_dir: 数据集根目录路径
    
    Returns:
        dict: 检查结果
    """
    results = {
        'structure_ok': False,
        'has_train': False,
        'has_val': False,
        'has_test': False,
        'structure_details': {}
    }
    
    data_path = Path(data_dir)
    
    # 检查常见的目录结构
    possible_structures = [
        # 结构1: train/val/test 目录
        {
            'train': data_path / 'train',
            'val': data_path / 'val',
            'test': data_path / 'test'
        },
        # 结构2: train_images/val_images/test_images
        {
            'train': data_path / 'train_images',
            'val': data_path / 'val_images',
            'test': data_path / 'test_images'
        },
        # 结构3: 包含images目录和JSON文件
        {
            'images': data_path / 'images',
            'train_json': data_path / 'train.json',
            'val_json': data_path / 'val.json',
            'test_json': data_path / 'test.json'
        }
    ]
    
    for i, structure in enumerate(possible_structures):
        if 'images' in structure:
            # JSON文件结构
            if structure['images'].exists() and structure['train_json'].exists():
                results['structure_ok'] = True
                results['structure_type'] = f'JSON_structure_{i+1}'
                results['has_train'] = structure['train_json'].exists()
                results['has_val'] = structure['val_json'].exists()
                results['has_test'] = structure['test_json'].exists()
                results['structure_details'] = {
                    'images_dir': str(structure['images']),
                    'train_json': str(structure['train_json']),
                    'val_json': str(structure['val_json']) if 'val_json' in structure else None,
                    'test_json': str(structure['test_json']) if 'test_json' in structure else None
                }
                break
        else:
            # 目录结构
            if structure['train'].exists():
                results['structure_ok'] = True
                results['structure_type'] = f'Directory_structure_{i+1}'
                results['has_train'] = structure['train'].exists()
                results['has_val'] = structure['val'].exists()
                results['has_test'] = structure['test'].exists()
                results['structure_details'] = {
                    'train_dir': str(structure['train']),
                    'val_dir': str(structure['val']) if structure['val'].exists() else None,
                    'test_dir': str(structure['test']) if structure['test'].exists() else None
                }
                break
    
    return results


def check_class_splits(data_dir, structure_info):
    """
    检查类别划分是否符合要求（64/16/20）
    
    Args:
        data_dir: 数据集根目录
        structure_info: 结构信息
    
    Returns:
        dict: 类别划分检查结果
    """
    results = {
        'splits_ok': False,
        'num_train_classes': 0,
        'num_val_classes': 0,
        'num_test_classes': 0,
        'expected': {'train': 64, 'val': 16, 'test': 20},
        'details': {}
    }
    
    if 'images_dir' in structure_info:
        # JSON文件结构
        train_json_path = structure_info['train_json']
        val_json_path = structure_info.get('val_json')
        test_json_path = structure_info.get('test_json')
        
        if os.path.exists(train_json_path):
            with open(train_json_path, 'r') as f:
                train_data = json.load(f)
            results['num_train_classes'] = len(train_data)
            results['details']['train_classes'] = list(train_data.keys())
        
        if val_json_path and os.path.exists(val_json_path):
            with open(val_json_path, 'r') as f:
                val_data = json.load(f)
            results['num_val_classes'] = len(val_data)
            results['details']['val_classes'] = list(val_data.keys())
        
        if test_json_path and os.path.exists(test_json_path):
            with open(test_json_path, 'r') as f:
                test_data = json.load(f)
            results['num_test_classes'] = len(test_data)
            results['details']['test_classes'] = list(test_data.keys())
    else:
        # 目录结构
        train_dir = structure_info.get('train_dir')
        val_dir = structure_info.get('val_dir')
        test_dir = structure_info.get('test_dir')
        
        if train_dir and os.path.exists(train_dir):
            train_classes = [d for d in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, d))]
            results['num_train_classes'] = len(train_classes)
            results['details']['train_classes'] = train_classes
        
        if val_dir and os.path.exists(val_dir):
            val_classes = [d for d in os.listdir(val_dir) 
                          if os.path.isdir(os.path.join(val_dir, d))]
            results['num_val_classes'] = len(val_classes)
            results['details']['val_classes'] = val_classes
        
        if test_dir and os.path.exists(test_dir):
            test_classes = [d for d in os.listdir(test_dir) 
                           if os.path.isdir(os.path.join(test_dir, d))]
            results['num_test_classes'] = len(test_classes)
            results['details']['test_classes'] = test_classes
    
    # 检查是否符合要求
    results['splits_ok'] = (
        results['num_train_classes'] == results['expected']['train'] and
        results['num_val_classes'] == results['expected']['val'] and
        results['num_test_classes'] == results['expected']['test']
    )
    
    return results


def check_image_properties(data_dir, structure_info, num_samples=10):
    """
    检查图像属性（尺寸、格式等）
    
    Args:
        data_dir: 数据集根目录
        structure_info: 结构信息
        num_samples: 采样检查的图像数量
    
    Returns:
        dict: 图像属性检查结果
    """
    results = {
        'image_size_ok': False,
        'expected_size': (84, 84),
        'sample_sizes': [],
        'image_formats': set(),
        'num_images_per_class': defaultdict(int)
    }
    
    if 'images_dir' in structure_info:
        # JSON文件结构
        images_dir = Path(structure_info['images_dir'])
        train_json_path = structure_info['train_json']
        
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
        
        # 采样检查
        sample_count = 0
        for class_name, image_list in train_data.items():
            if sample_count >= num_samples:
                break
            for img_name in image_list[:min(2, len(image_list))]:
                if sample_count >= num_samples:
                    break
                img_path = images_dir / img_name
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        results['sample_sizes'].append(img.size)
                        results['image_formats'].add(img.format)
                        results['num_images_per_class'][class_name] = len(image_list)
                        sample_count += 1
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
    else:
        # 目录结构
        train_dir = Path(structure_info.get('train_dir'))
        if train_dir.exists():
            sample_count = 0
            for class_dir in train_dir.iterdir():
                if sample_count >= num_samples:
                    break
                if class_dir.is_dir():
                    for img_file in class_dir.glob('*'):
                        if sample_count >= num_samples:
                            break
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            try:
                                img = Image.open(img_file)
                                results['sample_sizes'].append(img.size)
                                results['image_formats'].add(img.format)
                                results['num_images_per_class'][class_dir.name] = len(
                                    list(class_dir.glob('*'))
                                )
                                sample_count += 1
                            except Exception as e:
                                print(f"Error reading {img_file}: {e}")
    
    # 检查尺寸
    if results['sample_sizes']:
        unique_sizes = set(results['sample_sizes'])
        results['image_size_ok'] = (
            len(unique_sizes) == 1 and 
            unique_sizes.pop() == results['expected_size']
        )
        results['unique_sizes'] = list(unique_sizes)
    
    results['image_formats'] = list(results['image_formats'])
    
    return results


def check_total_images(data_dir, structure_info):
    """
    检查总图像数量（每个类别应该有600张）
    
    Returns:
        dict: 图像数量检查结果
    """
    results = {
        'expected_per_class': 600,
        'class_image_counts': {},
        'total_images': 0
    }
    
    if 'images_dir' in structure_info:
        train_json_path = structure_info['train_json']
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
        
        for class_name, image_list in train_data.items():
            results['class_image_counts'][class_name] = len(image_list)
            results['total_images'] += len(image_list)
    else:
        train_dir = Path(structure_info.get('train_dir'))
        if train_dir.exists():
            for class_dir in train_dir.iterdir():
                if class_dir.is_dir():
                    num_images = len(list(class_dir.glob('*')))
                    results['class_image_counts'][class_dir.name] = num_images
                    results['total_images'] += num_images
    
    return results


def print_check_results(structure_results, splits_results, image_results, count_results):
    """打印检查结果"""
    import sys
    import io
    # 设置UTF-8编码输出
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("Mini-ImageNet 数据集检查报告")
    print("=" * 60)
    
    print("\n1. 目录结构检查:")
    print(f"   [OK] 结构类型: {structure_results.get('structure_type', 'Unknown')}")
    print(f"   [OK] 训练集: {'YES' if structure_results['has_train'] else 'NO'}")
    print(f"   [OK] 验证集: {'YES' if structure_results['has_val'] else 'NO'}")
    print(f"   [OK] 测试集: {'YES' if structure_results['has_test'] else 'NO'}")
    
    print("\n2. 类别划分检查:")
    print(f"   训练集类别数: {splits_results['num_train_classes']} (期望: {splits_results['expected']['train']})")
    print(f"   验证集类别数: {splits_results['num_val_classes']} (期望: {splits_results['expected']['val']})")
    print(f"   测试集类别数: {splits_results['num_test_classes']} (期望: {splits_results['expected']['test']})")
    if splits_results['splits_ok']:
        print("   [PASS] 类别划分符合要求！")
    else:
        print("   [FAIL] 类别划分不符合要求！")
    
    print("\n3. 图像属性检查:")
    if image_results['sample_sizes']:
        print(f"   采样图像尺寸: {image_results['sample_sizes'][:5]}...")
        print(f"   期望尺寸: {image_results['expected_size']}")
        if image_results['image_size_ok']:
            print("   [PASS] 图像尺寸符合要求（84x84）！")
        else:
            print(f"   [FAIL] 图像尺寸不符合要求！发现尺寸: {image_results.get('unique_sizes', [])}")
        print(f"   图像格式: {image_results['image_formats']}")
    
    print("\n4. 图像数量检查:")
    if count_results['class_image_counts']:
        sample_classes = list(count_results['class_image_counts'].keys())[:5]
        print(f"   示例类别图像数: {[(k, v) for k, v in count_results['class_image_counts'].items() if k in sample_classes]}")
        print(f"   总图像数: {count_results['total_images']}")
        avg_per_class = count_results['total_images'] / len(count_results['class_image_counts']) if count_results['class_image_counts'] else 0
        print(f"   平均每类: {avg_per_class:.1f} (期望: {count_results['expected_per_class']})")
    
    print("\n" + "=" * 60)
    print("总体评估:")
    
    all_ok = (
        structure_results['structure_ok'] and
        splits_results['splits_ok'] and
        image_results['image_size_ok']
    )
    
    if all_ok:
        print("[PASS] 数据集检查通过！可以用于Table 2实验。")
    else:
        print("[FAIL] 数据集检查未完全通过，请检查上述问题。")
    
    print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查Mini-ImageNet数据集')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Mini-ImageNet数据集根目录路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"错误: 目录不存在: {args.data_dir}")
        return
    
    print(f"正在检查数据集: {args.data_dir}\n")
    
    # 1. 检查目录结构
    structure_results = check_dataset_structure(args.data_dir)
    
    if not structure_results['structure_ok']:
        print("错误: 无法识别数据集结构，请检查目录是否正确。")
        return
    
    # 2. 检查类别划分
    splits_results = check_class_splits(args.data_dir, structure_results['structure_details'])
    
    # 3. 检查图像属性
    image_results = check_image_properties(args.data_dir, structure_results['structure_details'])
    
    # 4. 检查图像数量
    count_results = check_total_images(args.data_dir, structure_results['structure_details'])
    
    # 5. 打印结果
    print_check_results(structure_results, splits_results, image_results, count_results)


if __name__ == "__main__":
    main()
