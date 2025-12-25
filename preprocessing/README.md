# 数据预处理流程

## 数据说明

### A类数据（结构光相机数据）
- **位置**: `G:\大黄鱼结构光\大黄鱼结构光`
- **格式**: 每组数据包含 `.png` (RGB图像) 和 `.ply` (点云)
- **特点**: 高精度3D点云数据，包含深度信息

### B类数据（运动相机数据）
- **位置**: `E:\ECSF\dahuangyu\code\datasets`
- **格式**: YOLO格式（train/valid/test，每个包含images和labels），图像已resize到640×640
- **特点**: 仅RGB图像，无深度信息

## 预处理流程

### 步骤0: 扫描并整理数据（首次运行需要）
```bash
python 1_scan_and_organize_data.py
```
- 扫描A类数据目录，排除"盲样"文件夹
- 整理数据到统一目录结构
- 生成数据索引文件

**输出**: `G:\dhy_data\organized_data/`

**注意**: 此步骤只需运行一次，后续处理可跳过

### 步骤1: 生成深度图和Mask（使用YOLO）
```bash
python 3_generate_depth_maps_with_yolo.py
```
- 从PLY有序点云直接提取深度图（0偏差对齐）
- 使用YOLO分割模型生成Mask（只包含鱼体）
- 根据Mask裁剪数据（扩大10%边界框）
- 将所有图像resize到256×256

**输出**: 
- `rgb_256.png`: RGB图像 (256×256)
- `depth_256.png`: 深度图可视化 (256×256, 0-255)
- `depth_256.npy`: 原始深度值 (256×256, float32)
- `mask_256.png`: Mask (256×256, 0/255)

### 步骤2: 准备A类数据集
```bash
python 4_prepare_dataset.py
```
- 整合A类数据到最终数据集
- 统一数据格式和尺寸
- 生成训练用的数据集索引

**输出**: `G:\dhy_data/`

### 步骤3: 处理B类数据（单独运行）
```bash
python 5_prepare_b_class_data.py
```
- 使用YOLO生成Mask（只包含鱼体）
- Resize图像到256×256
- 统一编号（sample_XXXX，不区分train/val/test）
- 生成B类数据索引

**注意**: B类数据需要单独处理，因为需要YOLO模型生成mask

### 一键运行（跳过步骤0）
```bash
python run_all.py
```
- 自动执行步骤1和步骤2
- 假设数据已经通过步骤0处理完成

## 数据集结构

```
G:\dhy_data/
├── A_class/
│   ├── wild/
│   │   └── sample_0001/
│   │       ├── rgb.png      # RGB图像 (256×256)
│   │       ├── depth.npy    # 深度图 (256×256, float32)
│   │       └── mask.png     # Mask (256×256)
│   └── farmed/
│       └── ...
├── B_class/
│   ├── wild/
│   │   └── b_00001/
│   │       └── rgb.png      # RGB图像 (256×256)
│   └── farmed/
│       └── ...
└── dataset_index.json        # 数据集索引
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 技术说明

### 深度图生成（有序点云方法）

Zivid相机输出的PLY是有序点云，第N个点严格对应图像第N个像素。
- 直接`reshape(H, W, 3)`即可完美对齐，0偏差
- 提取Z通道作为深度值
- 无需复杂的投影计算

### Mask生成（YOLO分割模型）

- 使用已训练的YOLO分割模型
- 只包含鱼体，不包含托盘和背景
- 更精确，适合分类任务
- 自动检测CUDA，使用GPU加速

### 数据裁剪

- 根据Mask计算边界框
- 扩大10%作为边距
- 裁剪RGB、深度图和Mask
- 减少背景干扰，提高训练效率

## 文件说明

### 核心脚本
- `1_scan_and_organize_data.py`: 扫描并整理原始数据（首次运行需要）
- `3_generate_depth_maps_with_yolo.py`: 生成深度图和Mask（使用YOLO）
- `4_prepare_dataset.py`: 准备最终数据集
- `run_all.py`: 一键运行批处理（跳过步骤0）
- `test_single_sample_complete.py`: 单个样本测试脚本（用于调试）

### 配置文件
- `config.json`: 所有路径和参数配置

### 文档
- `README.md`: 本文件
- `MODEL_ARCHITECTURE.md`: 模型架构说明

## 下一步

完成预处理后，使用 `G:\dhy_data\dataset_index.json` 加载数据进行模型训练。
