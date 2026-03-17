# 实验二：概率建模、KNN分类与梯度下降优化

## 实验目的

### 知识目标
1. 理解概率模型中的参数估计问题
2. 理解最大似然估计（MLE）与贝叶斯估计（MAP）
3. 理解 KNN 分类的基本思想
4. 理解梯度下降法的优化机制

### 能力目标
1. 能使用 Python 实现参数估计
2. 能实现 KNN 分类模型
3. 能实现梯度下降算法
4. 能通过图像分析模型行为

## 实验内容

### 环境要求
- Python 3.x
- numpy
- matplotlib
- scipy
- scikit-learn

### 安装依赖
```bash
pip install numpy matplotlib scipy scikit-learn
```

### 运行实验
```bash
python experiment_2.py
```

## 实验结果

### 1. 参数估计（MLE与MAP）
- MLE = 0.6
- MAP = 0.5714
- 样本量: 5
- 正样本数: 3
- 先验参数: alpha=2, beta=2

### 2. KNN分类
- 训练数据:
  - 样本1: [1 2], 标签: 0
  - 样本2: [2 3], 标签: 0
  - 样本3: [3 3], 标签: 0
  - 样本4: [6 5], 标签: 1
  - 样本5: [7 7], 标签: 1
  - 样本6: [8 6], 标签: 1
- K=1, accuracy=1.0, 预测结果: [0 0 0 1 1 1]
- K=3, accuracy=1.0, 预测结果: [0 0 0 1 1 1]
- K=5, accuracy=1.0, 预测结果: [0 0 0 1 1 1]

### 3. 梯度下降
- 优化过程: 20次迭代，x从5.0收敛到-0.9308
- 损失函数: 从36.0下降到0.0048

## 实验分析

1. **参数估计**：
   - MLE仅基于样本数据，结果为样本均值
   - MAP结合了先验知识，结果向先验均值（0.5）靠拢
   - 当样本量较小时，先验对结果影响较大

2. **KNN分类**：
   - 对于简单数据集，不同K值都能达到100%准确率
   - K值越大，模型越简单，决策边界越平滑
   - K值越小，模型越复杂，容易过拟合

3. **梯度下降**：
   - 算法成功收敛到函数最小值
   - 学习率选择合适，收敛速度稳定
   - 损失函数单调递减，说明优化方向正确

## 如何推送到GitHub

1. 克隆仓库：
   ```bash
   git clone https://github.com/comic111111/ML-Course-2026.git
   ```

2. 复制文件到仓库目录：
   - experiment_2.py
   - README.md

3. 提交并推送：
   ```bash
   git add .
   git commit -m "Add experiment 2: Probability modeling, KNN classification and gradient descent"
   git push origin main
   ```
