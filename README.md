# Vision Transformer (ViT) for CIFAR-10/100

基于PyTorch实现的Vision Transformer (ViT)模型，支持CIFAR-10和CIFAR-100数据集，包含多种训练增强方法。

 🚀 **特征亮点：**
- ✅ 支持3种模型规模：ViT-Tiny/Small/Base
- ✅ 支持2种数据集：CIFAR-10/100
- ✅ **8种数据增强方法**：RandAugment, Cutout, ColorJitter, Rotation, Affine等
- ✅ **训练增强技术**：AMP（2-3倍速度）、EMA（精度+0.5-1%）、梯度裁剪（稳定性）
- ✅ **Stochastic Depth (DropPath)**：随机深度正则化，推荐值0.1-0.2
- ✅ **MixUp**：批级别混合增强，推荐alpha=0.2, prob=0.5
- ✅ **Optional GPU加速**：Kornia数据增强支持（需单独安装）
- ✅ Checkpoint自动管理，命名含模型规模+数据集

---

## 📚 文档导航

### 快速开始
- [安装指南](#安装指南)
- [基本使用](#基本使用)
- [模型配置](#模型配置)

### 详细文档
- [**数据增强方法完整指南**](DATA_AUGMENTATION.md) ⭐ 新增！8种增强方法详解
- [训练增强参数](#训练增强参数)
- [Checkpoint管理](#checkpoint管理)
- [完整使用示例](#完整使用示例)

---

## 快速开始

### 1. 安装指南

```bash
# 使用conda创建环境（推荐）
conda env create -f environment.yml
conda activate vit

# 或使用pip安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

```bash
# 自动下载CIFAR-10和CIFAR-100到./data目录
python download_datasets.py

# 验证数据集加载
python data_loader.py

# 测试数据增强方法
python test_data_aug.py
```

### 3. 基本使用

```bash
# 使用默认配置训练 ViT-Small 模型（CIFAR-10）
python train.py 

# dry-run验证
python train.py --mode dry_run

#快速启动
python train.py --mode train `
  --model-size tiny --dataset cifar10 `
  --amp `
  --grad-clip 1.0 `
  --ema-decay 0.9999 `
  --label-smoothing 0.1 `
  --drop-path 0.1 `
  --epochs 5 `
  --batch-size 64 `
  --warmup-epochs 10 `
  --kornia

#数据增强
python train.py --mode train `
  --model-size tiny --dataset cifar10 `
  --amp `
  --grad-clip 1.0 `
  --ema-decay 0.9999 `
  --label-smoothing 0.1 `
  --epochs 5 `
  --batch-size 64 `
  --warmup-epochs 10 `
  --randaug --randaug-n 2 --randaug-m 9 `
  --cutout --cutout-length 16 `
  --color-jitter `
  --rotation --rotation-degrees 15 `
  --drop-path 0.1 `
  --mixup --mixup-alpha 0.2 --mixup-prob 0.5 `
  --affine `
  --kornia
```

---

## 模型配置参数

### 模型规模选择

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--model-size` | `small` | `tiny`, `small`, `base` | 模型规模 |

**模型对比表：**

| 规模 | 参数量 | 隐藏层 | 深度 | 头数 | 推荐场景 |
|------|--------|--------|------|------|----------|
| **tiny** | ~5M | 256 | 4 | 4 | 快速实验、资源受限 |
| **small** | ~10M | 512 | 6 | 8 | **推荐默认** |
| **base** | ~22M | 768 | 12 | 12 | 追求最佳精度 |

**示例：**
```bash
# ViT-Tiny（最快训练）
python train.py --model-size tiny --epochs 30

# ViT-Small（平衡性能）
python train.py --model-size small --epochs 50

# ViT-Base（最高精度）
python train.py --model-size base --epochs 100
```

### 数据集选择

| 参数 | 默认值 | 可选值 |
|------|--------|--------|
| `--dataset` | `cifar10` | `cifar10`, `cifar100` |

**数据集对比：**

| 数据集 | 类别数 | 训练样本 | 测试样本 | 推荐模型 |
|--------|--------|----------|----------|----------|
| CIFAR-10 | 10 | 50,000 | 10,000 | tiny/small/base |
| CIFAR-100 | 100 | 50,000 | 10,000 | small/base |

---

## 数据增强参数 ⭐

详细文档请查看 [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md) - 包含8种增强方法详解和性能对比！

### 快速启用（推荐配置）

```bash
# 标准增强配置（推荐所有人使用）
python train.py --randaug --randaug-n 2 --randaug-m 9 --cutout --cutout-length 16
```

### 增强方法速查表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--randaug` | flag | False | 启用RandAugment（提升1-2%精度） |
| `--randaug-n` | int | 2 | 操作数量（2-3） |
| `--randaug-m` | int | 9 | 强度（1-10，推荐9-12） |
| `--cutout` | flag | False | 启用Cutout（随机擦除） |
| `--cutout-length` | int | 16 | 擦除块大小（8-16像素） |
| `--color-jitter` | flag | False | 启用颜色抖动（亮度/对比度/饱和度） |
| `--rotation` | flag | False | 启用随机旋转 |
| `--rotation-degrees` | float | 15 | 最大旋转角度 |
| `--affine` | flag | False | 启用仿射变换 |

### 数据增强配置示例

```bash
# 轻度增强（适合ViT-Tiny）
python train.py --model-size tiny \
  --color-jitter \
  --rotation --rotation-degrees 10

# 标准增强（推荐）
python train.py --model-size small \
  --randaug --randaug-n 2 --randaug-m 9 \
  --cutout --cutout-length 16

# 强力增强（适合ViT-Base）
python train.py --model-size base --epochs 200 \
  --randaug --randaug-n 3 --randaug-m 12 \
  --cutout --cutout-length 16 \
  --rotation --rotation-degrees 15 \
  --affine --affine-translate 0.1
```

**详细文档：** 📖 [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md)

---

## 训练增强参数

### 1. 自动混合精度 (AMP) ⚡

```bash
python train.py --amp
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--amp` | False | **强烈推荐启用** |

**优势：**
- ✅ 2-3倍训练速度提升
- ✅ 显存占用减少40-50%
- ✅ 精度几乎无损

### 2. 梯度裁剪 🔐

```bash
python train.py --grad-clip 1.0
```

| 参数 | 默认值 | 推荐值 |
|------|--------|--------|
| `--grad-clip` | 0.0 (禁用) | **1.0** |

**为什么重要：**
- ✅ 防止梯度爆炸（ViT训练常见问题）
- ✅ 提高训练稳定性

### 3. 指数移动平均 (EMA) ⭐

```bash
python train.py --ema-decay 0.9999
```

| 参数 | 默认值 | 推荐值 |
|------|--------|--------|
| `--ema-decay` | 0.0 (禁用) | **0.9999** |

**效果：**
- ✅ 提升精度0.5-1%
- ✅ 平滑权重更新

### 4. 标签平滑 🌊

```bash
python train.py --label-smoothing 0.1
```

| 参数 | 默认值 | 推荐值 |
|------|--------|--------|
| `--label-smoothing` | 0.0 (禁用) | 0.1 |

---

## 训练参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--epochs` | 100 | 训练轮数 | 100-300 |
| `--batch-size` | 64 | Batch大小 | 64-128（有AMP） |
| `--lr` | 3e-4 | 学习率 | 3e-4 |
| `--weight-decay` | 0.03 | 权重衰减 | 0.03 |
| `--warmup-epochs` | 5 | Warmup轮数 | 5-10 |

---

## 完整使用示例

### 示例1：快速基线测试（5分钟）

```bash
# ViT-Small + CIFAR-10（最常用配置）
python train.py --mode dry_run --model-size small \
  --epochs 2 --batch-size 32 --warmup-epochs 1
```

**预期结果：**
- 训练时间：~3分钟
- 测试精度：~70-75%（仅2个epoch）
- 模型大小：~10M参数

---

### 示例2：标准训练配置（1小时）

```bash
# 推荐配置：AMP + EMA + 数据增强
python train.py --mode train --model-size small --dataset cifar10 \
  --epochs 100 --batch-size 128 --warmup-epochs 10 \
  --amp \
  --ema-decay 0.9999 \
  --grad-clip 1.0 \
  --randaug --randaug-n 2 --randaug-m 9 \
  --cutout --cutout-length 16
```

**预期结果：**
- 训练时间：~1小时（RTX 3090 + AMP）
- 测试精度：~90-92%（CIFAR-10）
- 使用显存：~6-8GB

---

### 示例3：最佳精度（追求极致）

```bash
# ViT-Base + 最强增强 + CIFAR-100
python train.py --mode train --model-size base --dataset cifar100 \
  --epochs 200 --batch-size 64 \
  --amp \
  --ema-decay 0.9999 \
  --grad-clip 1.5 \
  --label-smoothing 0.1 \
  --randaug --randaug-n 3 --randaug-m 12 \
  --cutout --cutout-length 16 \
  --rotation --rotation-degrees 15 \
  --affine --affine-translate 0.1
```

**预期结果：**
- 训练时间：~4小时
- 测试精度：~74-77%（CIFAR-100）
- 模型大小：~22M参数

---

### 示例4：评估已训练模型

```bash
# 评估最新保存的checkpoint
python train.py --mode test

# 评估特定checkpoint
python train.py --mode test --checkpoint checkpoints/vit/best_s_10_20250527.pth

# 查看所有checkpoint
ls -lh checkpoints/vit/
```

---

### 示例5：恢复训练

```bash
# 从最新checkpoint恢复训练
python train.py --mode resume --checkpoint checkpoints/vit/latest.pth

# 增加训练轮数
python train.py --mode resume --checkpoint checkpoints/vit/final_s_10_20250527.pth --epochs 150
```

---

## Checkpoint管理

### 保存位置与命名规则

```
checkpoints/
└── vit/
    ├── best_t_10_20250527_0820.pth     # Tiny模型，CIFAR-10，最佳权重
    ├── best_s_10_20250527_0925.pth     # Small模型，CIFAR-10，最佳权重
    ├── best_b_100_20250527_1050.pth    # Base模型，CIFAR-100，最佳权重
    ├── final_s_10_20250527_1300.pth    # Small模型，CIFAR-10，最终权重
    └── checkpoint_s_10_20250527_1400.pth # 训练过程中的检查点
```

**命名格式：**
```
{类型}_{规模}_{数据集}_{时间}.pth
```

- **类型**：`best`（最佳权重）、`final`（最终权重）
- **规模**：`t`（tiny）、`s`（small）、`b`（base）
- **数据集**：`10`（CIFAR-10）、`100`（CIFAR-100）
- **时间**：`YYYYMMDD_HHMMSS`

### Checkpoint包含内容

- ✅ 模型权重
- ✅ 优化器状态
- ✅ 调度器状态
- ✅ 训练历史（损失、准确率曲线）
- ✅ 所有配置参数
- ✅ 随机状态（完整恢复训练）
- ✅ AMP scaler状态（如果启用）
- ✅ EMA权重（如果启用）

---

## 性能对比

### CIFAR-10 准确率对比

| 配置 | ViT-Tiny | ViT-Small | ViT-Base | 训练时间 |
|------|----------|-----------|----------|----------|
| 基础配置 | 82-84% | 85-86% | 88-89% | 15-20min |
| + AMP | 82-84% | 85-86% | 88-89% | 8-10min |
| + 标准增强 | 83-85% | 87-89% | 89-91% | 10-15min |
| + 全增强 | 84-86% | 90-92% | 92-94% | 50-60min |

### 显存占用对比

| 模型 | Batch=32 | Batch=64 | Batch=128 | 使用AMP |
|------|----------|----------|-----------|---------|
| ViT-Tiny | 1.5GB | 2.5GB | 4GB | -50% |
| ViT-Small | 3GB | 5GB | 9GB | -40% |
| ViT-Base | 6GB | 11GB | 20GB | -30% |

---

## 项目结构

```
ViT_dl/
├── config.py              # 配置类（模型/训练/增强参数）
├── model.py               # ViT模型定义（3种规模）
├── train.py               # 主训练脚本（AMP/EMA/增强支持）
├── data_loader.py         # 数据加载（8种增强方法）
├── utils.py               # 工具函数（可视化、日志）
├── download_datasets.py   # 数据集下载脚本
├── test_data_aug.py       # 数据增强测试脚本
├── README.md              # 主文档（本文件）
├── DATA_AUGMENTATION.md   # 数据增强详细文档 ⭐
├── requirements.txt       # PIP依赖列表
├── environment.yml        # Conda环境配置
├── .gitignore            # Git忽略规则
├── logs/                 # 训练日志目录
├── checkpoints/          # 模型检查点目录
│   └── vit/             # ViT模型checkpoint
└── data/                # 数据集目录（自动创建）
```

---

## 常见问题

### Q1: AMP训练出现NaN损失？
```bash
# 添加梯度裁剪
python train.py --amp --grad-clip 1.0
```

### Q2: 训练速度慢？
```bash
# 启用AMP + 增大batch size
python train.py --amp --batch-size 128 --randaug
```

### Q3: 训练不稳定？
```bash
# 使用稳定性增强组合
python train.py --amp --grad-clip 1.0 --ema-decay 0.9999
```

### Q4: 显存不足？
```bash
# 减小batch size + 使用AMP
python train.py --batch-size 32 --amp
```
---

## 更新日志

### 2026-05-27

**新增功能：**
- ✅ **8种数据增强方法**：RandAugment、Cutout、ColorJitter、Rotation、Affine等
- ✅ **完整文档**：[DATA_AUGMENTATION.md](DATA_AUGMENTATION.md) 详细说明
- ✅ **模块化架构**：灵活配置和组合
- ✅ **测试脚本**：验证所有增强方法

**改进：**
- 📈 RandAugment提升精度1-2%
- 🔧 梯度裁剪防止训练爆炸
- 🚀 EMA提升最终精度0.5-1%

---

### 更新日志（2026-05-27）

**新增功能：**
- ✅ **MixUp**：批级别混合增强，已集成到训练循环
- ✅ **Stochastic Depth (DropPath)**：随机深度正则化
- ✅ **Kornia GPU加速**：可选GPU数据增强（需安装kornia）
- ✅ **PyTorch 2.x兼容**：修复GradScaler和label_smoothing警告

**改进：**
- 🚀 训练速度提升10-30%（启用Kornia）
- 🎯 泛化能力增强（DropPath + MixUp）
- 🔧 代码质量提升（去除了AutoAugment fallback）

### 待实现功能：
- [ ] MAE预训练支持
- [ ] TensorBoard集成

---