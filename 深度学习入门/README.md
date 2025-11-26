# 深度学习通俗教程 - 完整目录

> Copyright 2025 石旭乔 306开放创新实验室

> 用比喻和故事，让深度学习变得简单易懂

---

## 第一章：深度学习基础概念

### 1.1 什么是深度学习
- 从机器学习到深度学习
- 神经网络的诞生故事
- 为什么叫"深度"学习

### 1.2 神经网络基础
- 神经元：大脑的基本单位
- 激活函数：神经元的"开关"
- 前向传播：信息如何流动
- 反向传播：模型如何学习
- 损失函数：衡量对错的标尺
- 优化器：寻找最优解的导航

---

## 第二章：学习方式分类

### 2.1 监督学习 (Supervised Learning)
- 核心概念：有老师指导的学习
- 分类任务与回归任务
- 常见算法与应用

### 2.2 无监督学习 (Unsupervised Learning)
- 核心概念：自己探索规律
- 聚类与降维
- 异常检测

### 2.3 半监督学习 (Semi-supervised Learning)
- 少量标注 + 大量无标注
- 伪标签技术

### 2.4 自监督学习 (Self-supervised Learning)
- 让数据自己教自己
- 对比学习、掩码预测
- BERT、GPT的预训练方法


### 2.5 强化学习 (Reinforcement Learning)
- 试错中学习
- 奖励与惩罚机制
- 应用：游戏AI、机器人控制

### 2.6 少样本学习 (Few-Shot / Zero-Shot / One-Shot)
- 举一反三的能力
- 元学习与知识迁移

---

## 第三章：经典网络架构

### 3.1 全连接神经网络 (MLP)
- 最基础的网络结构
- 优缺点分析

### 3.2 卷积神经网络 (CNN)
- 局部感知与权值共享
- 卷积层、池化层
- 经典架构：LeNet、AlexNet、VGG、ResNet、EfficientNet

### 3.3 循环神经网络 (RNN)
- 记忆与序列处理
- 梯度消失/爆炸问题

### 3.4 长短期记忆网络 (LSTM)
- 遗忘门、输入门、输出门
- 解决长期依赖问题
- 双向LSTM

### 3.5 门控循环单元 (GRU)
- LSTM的简化版
- 何时选择GRU vs LSTM

### 3.6 Transformer架构
- 注意力机制 (Attention)
- Self-Attention与Multi-Head Attention
- Position Encoding
- Encoder-Decoder结构
- BERT、GPT系列

### 3.7 生成对抗网络 (GAN)
- 生成器 vs 判别器
- 对抗训练过程
- 变体：DCGAN、WGAN、StyleGAN、CycleGAN

### 3.8 自编码器 (AutoEncoder)
- 压缩与重建
- 变分自编码器 (VAE)

### 3.9 图神经网络 (GNN)
- 处理图结构数据
- GCN、GraphSAGE、GAT

---

## 第四章：训练技巧与优化

### 4.1 迁移学习 (Transfer Learning)
- 站在巨人的肩膀上
- 预训练模型的使用

### 4.2 微调 (Fine-tuning)
- 冻结层 vs 全部训练
- 学习率策略

### 4.3 数据增强 (Data Augmentation)
- 图像增强：旋转、翻转、裁剪
- 文本增强：同义词替换、回译
- MixUp、CutMix

### 4.4 正则化技术
- Dropout：随机失活
- Batch Normalization
- Layer Normalization
- Weight Decay
- Early Stopping

---

## 第五章：优化器与学习率调度

### 5.1 优化器详解
- SGD：随机梯度下降
- Momentum：动量法
- Adam：自适应学习率
- AdaGrad、RMSprop、AdamW

### 5.2 学习率调度
- Step Decay
- Cosine Annealing
- Warm-up策略
- 循环学习率

### 5.3 梯度裁剪
- 防止梯度爆炸

---

## 第六章：损失函数与评估指标

### 6.1 回归损失函数
- MSE、MAE、Huber Loss

### 6.2 分类损失函数
- Cross-Entropy
- Focal Loss
- Label Smoothing

### 6.3 特殊任务损失
- 目标检测、分割
- GAN损失
- 度量学习

### 6.4 评估指标
- 分类：Accuracy、Precision、Recall、F1、AUC
- 回归：MAE、RMSE、R²
- 检测/分割：mAP、mIoU

---

## 第七章：超参数调优与模型调试

### 7.1 超参数搜索方法
- 手动调参
- 网格搜索
- 随机搜索
- 贝叶斯优化

### 7.2 调优技巧
- 对数空间搜索
- 从粗到细
- LR Range Test
- 早停与检查点

### 7.3 问题诊断
- 训练不动
- 过拟合
- 欠拟合
- 训练不稳定

---

## 第八章：实战项目与综合应用

### 8.1 项目开发通用流程
- 6个阶段与时间分配

### 8.2 完整项目案例：猫狗识别器
- 数据准备（40%时间）
- 模型开发（30%时间）
- 评估部署（30%时间）

### 8.3 最佳实践
- 代码组织
- 配置管理
- 实验追踪
- 文档编写

---

## 学习路线建议

**新手路线** (0-3个月)
```
第一章 → 第二章(2.1-2.5) → 第三章(3.1-3.2, 3.6) → 第四章 → 第八章
```

**进阶路线** (3-6个月)
```
第三章(全部) → 第五章 → 第六章 → 第七章 → 完整项目实战
```

**高级路线** (6个月+)
```
阅读论文 → 复现经典模型 → 参与竞赛 → 开源贡献
```

---

## 推荐资源

**在线课程**
- 吴恩达《深度学习专项课程》
- Fast.ai《程序员深度学习实践》
- CS231n、CS224n

**书籍**
- 《深度学习》(花书) - Ian Goodfellow
- 《动手学深度学习》- 李沐
- 《Python深度学习》- François Chollet

**实践平台**
- Kaggle - 数据竞赛
- Papers with Code - 论文复现
- Google Colab - 免费GPU

---

**让我们一起把深度学习学透！**
