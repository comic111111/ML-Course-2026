#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# 读取 Excel 文件
file_name = 'smart_agri_tomato_timeseries_raw.xlsx'  # 改成你实际的文件名

try:
    df = pd.read_excel(file_name)
    print("成功读取 Excel 文件\n")

    # 1. 查看数据规模
    print("=" * 50)
    print("数据集形状:", df.shape)
    print("行数:", df.shape[0], "列数:", df.shape[1])

    # 2. 查看字段名称
    print("\n" + "=" * 50)
    print("字段列表:")
    print(df.columns.tolist())

    # 3. 查看缺失值情况
    print("\n" + "=" * 50)
    print("缺失值统计:")
    print(df.isnull().sum())

    # 4. 查看前几行数据
    print("\n" + "=" * 50)
    print("前5行数据:")
    print(df.head())

    # 5. 查看数据类型
    print("\n" + "=" * 50)
    print("数据类型:")
    print(df.dtypes)

except FileNotFoundError:
    print(f"错误：找不到文件 '{file_name}'")
    print("\n请检查：")
    print("1. 文件名是否正确")
    print("2. 文件是否在当前目录")

    # 显示当前目录的文件
    print("\n当前目录下的文件：")
    import os
    for file in os.listdir('.'):
        print(f"  - {file}")

except Exception as e:
    print(f"读取文件时出错：{e}")
    print("\n请确保已安装 openpyxl：")
    print("pip install openpyxl")


# In[2]:


pip install openpyxl


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置图形风格
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 1. 缺失值可视化
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]

plt.figure(figsize=(10, 6))
bars = plt.bar(missing_counts.index, missing_counts.values, color='skyblue')
plt.title('Missing Values by Column', fontsize=14)
plt.xlabel('Column Name', fontsize=12)
plt.ylabel('Missing Count', fontsize=12)
plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Missing values statistics:")
print(missing_counts)

# 2. 数值特征分布直方图
numeric_cols = [
    "temp", "humidity", "light", "co2", "irrigation", "fertilizer_ec",
    "ph", "canopy_temp", "temp_24h_mean", "light_24h_sum",
    "co2_24h_mean", "irrigation_24h_sum", "growth_index",
    "yield_now", "yield_next_24h"
]

fig, axes = plt.subplots(5, 3, figsize=(15, 18))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}', fontsize=10)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, alpha=0.3)

for i in range(len(numeric_cols), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Numerical Features Distribution', fontsize=16)
plt.tight_layout()
plt.show()

# 3. 相关性矩阵热力图
corr = df[numeric_cols].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 打印与目标变量相关性最强的特征
print("\nTop features correlated with yield_next_24h (absolute value):")
corr_with_target = corr['yield_next_24h'].abs().sort_values(ascending=False)
for var, corr_val in corr_with_target.items():
    if var != 'yield_next_24h':
        original_corr = corr.loc[var, 'yield_next_24h']
        print(f"  {var}: {original_corr:.3f} (abs: {corr_val:.3f})")

# 4. 关键变量与产量的散点图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

important_vars = ['temp', 'light', 'co2', 'humidity', 'growth_index', 'yield_now']
titles = ['Temp vs Next 24h Yield', 'Light vs Next 24h Yield', 
          'CO2 vs Next 24h Yield', 'Humidity vs Next 24h Yield',
          'Growth Index vs Next 24h Yield', 'Current Yield vs Next 24h Yield']

for i, (var, title) in enumerate(zip(important_vars, titles)):
    axes[i].scatter(df[var], df['yield_next_24h'], alpha=0.5, s=10)
    axes[i].set_title(title, fontsize=12)
    axes[i].set_xlabel(var, fontsize=10)
    axes[i].set_ylabel('yield_next_24h', fontsize=10)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. 时间序列可视化
df['timestamp'] = pd.to_datetime(df['timestamp'])
gh1 = df[df['greenhouse_id'] == 1].sort_values('timestamp')

plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.plot(gh1['timestamp'][:500], gh1['temp'][:500], label='Temperature', linewidth=1)
plt.plot(gh1['timestamp'][:500], gh1['canopy_temp'][:500], label='Canopy Temp', linewidth=1)
plt.title('Greenhouse 1 - Temperature Time Series (first 500 points)', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(gh1['timestamp'][:500], gh1['yield_now'][:500], label='Current Yield', linewidth=1)
plt.plot(gh1['timestamp'][:500], gh1['yield_next_24h'][:500], label='Next 24h Yield', linewidth=1)
plt.title('Greenhouse 1 - Yield Time Series (first 500 points)', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Yield')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. 统计摘要
print("=" * 60)
print("Numerical Features Statistics Summary")
print("=" * 60)
print(df[numeric_cols].describe())

print("\n" + "=" * 60)
print("Feature Skewness (measure of distribution asymmetry)")
print("=" * 60)
skewness = df[numeric_cols].skew().sort_values(ascending=False)
for var, skew_val in skewness.items():
    if abs(skew_val) > 1:
        print(f"  {var}: {skew_val:.3f} (highly skewed)")
    elif abs(skew_val) > 0.5:
        print(f"  {var}: {skew_val:.3f} (moderately skewed)")
    else:
        print(f"  {var}: {skew_val:.3f} (approximately symmetric)")


# In[6]:


# 安装 seaborn
get_ipython().system('pip install seaborn')


# In[11]:


# ============================================
# 模块三：预处理与 Pipeline
# ============================================

# 导入必要的库
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. 分离特征和目标变量
X = df.drop(columns=["yield_next_24h", "timestamp"])
y = df["yield_next_24h"]

# 2. 定义数值特征和类别特征
numeric_features = [
    "temp", "humidity", "light", "co2", "irrigation", "fertilizer_ec",
    "ph", "canopy_temp", "temp_24h_mean", "light_24h_sum",
    "co2_24h_mean", "irrigation_24h_sum", "growth_index", "yield_now"
]

categorical_features = ["greenhouse_id"]

# 3. 数值特征预处理：均值填补 + 标准化
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),      # 用均值填补缺失值
    ("scaler", StandardScaler())                      # 标准化
])

# 4. 类别特征预处理：众数填补 + 独热编码
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # 用众数填补缺失值
    ("onehot", OneHotEncoder(handle_unknown="ignore"))     # 独热编码
])

# 5. 组合预处理步骤
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

print("=" * 50)
print("模块三：预处理流程构建完成")
print("=" * 50)
print(f"数值特征数量: {len(numeric_features)}")
print(f"类别特征数量: {len(categorical_features)}")
print("\n预处理流程:")
print(preprocessor)


# In[12]:


# ============================================
# 模块四：防数据泄漏（时间序列划分）
# ============================================

# 1. 按时间排序（关键！防止数据泄漏）
df = df.sort_values("timestamp").reset_index(drop=True)

# 2. 重新分离特征和目标变量
X = df.drop(columns=["yield_next_24h", "timestamp"])
y = df["yield_next_24h"]

# 3. 按时间顺序划分训练集和测试集（80%训练，20%测试）
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx].copy()   # 训练集特征
X_test = X.iloc[split_idx:].copy()    # 测试集特征
y_train = y.iloc[:split_idx].copy()   # 训练集目标
y_test = y.iloc[split_idx:].copy()    # 测试集目标

print("=" * 50)
print("模块四：数据划分完成")
print("=" * 50)
print(f"数据集总行数: {len(df)}")
print(f"训练集大小: {len(X_train)} 行 ({len(X_train)/len(df)*100:.1f}%)")
print(f"测试集大小: {len(X_test)} 行 ({len(X_test)/len(df)*100:.1f}%)")
print(f"\n训练集时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[split_idx-1]}")
print(f"测试集时间范围: {df['timestamp'].iloc[split_idx]} 到 {df['timestamp'].iloc[-1]}")

# 4. 应用预处理（只在训练集上 fit！）
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\n训练集处理后形状: {X_train_processed.shape}")
print(f"测试集处理后形状: {X_test_processed.shape}")

# 显示处理后的特征数量变化
print(f"\n原始特征数: {X.shape[1]}")
print(f"处理后特征数: {X_train_processed.shape[1]}")
print("（特征数增加是因为温室编号进行了独热编码）")


# In[13]:


# ============================================
# 模块五：手写梯度下降线性回归
# ============================================

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 50)
print("模块五：手写梯度下降线性回归")
print("=" * 50)

# 1. 转换数据格式（如果是稀疏矩阵，转为密集数组）
if hasattr(X_train_processed, "toarray"):
    X_train_gd = X_train_processed.toarray()
    X_test_gd = X_test_processed.toarray()
else:
    X_train_gd = X_train_processed
    X_test_gd = X_test_processed

# 2. 添加偏置项（全1列，对应截距项 b）
X_train_b = np.c_[np.ones((X_train_gd.shape[0], 1)), X_train_gd]
X_test_b = np.c_[np.ones((X_test_gd.shape[0], 1)), X_test_gd]

# 3. 转换目标变量为列向量
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)

print(f"训练集特征维度: {X_train_b.shape}")
print(f"测试集特征维度: {X_test_b.shape}")

# 4. 初始化参数
theta = np.zeros((X_train_b.shape[1], 1))
lr = 0.01          # 学习率
epochs = 500       # 迭代次数

print(f"\n训练参数:")
print(f"  学习率: {lr}")
print(f"  迭代次数: {epochs}")

# 5. 梯度下降训练
loss_history = []

for epoch in range(epochs):
    # 前向传播：计算预测值
    y_pred = X_train_b @ theta

    # 计算误差
    error = y_pred - y_train_np

    # 计算损失（MSE）
    loss = np.mean(error ** 2)
    loss_history.append(loss)

    # 计算梯度
    grad = (2 / X_train_b.shape[0]) * (X_train_b.T @ error)

    # 更新参数
    theta = theta - lr * grad

    # 每100轮打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

# 6. 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Gradient Descent Loss Curve", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7. 在测试集上评估
y_test_pred_gd = X_test_b @ theta
mse_gd = mean_squared_error(y_test_np, y_test_pred_gd)
r2_gd = r2_score(y_test_np, y_test_pred_gd)

print("\n" + "=" * 50)
print("手写梯度下降线性回归结果")
print("=" * 50)
print(f"MSE (均方误差): {mse_gd:.4f}")
print(f"RMSE (均方根误差): {np.sqrt(mse_gd):.4f}")
print(f"R² (决定系数): {r2_gd:.4f}")

# 输出参数统计
print(f"\n参数统计:")
print(f"  截距项 (b): {theta[0, 0]:.4f}")
print(f"  参数数量: {len(theta) - 1}")
print(f"  参数均值: {np.mean(theta[1:]):.4f}")
print(f"  参数标准差: {np.std(theta[1:]):.4f}")


# In[14]:


# ============================================
# 模块六：sklearn 线性回归对照
# ============================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 50)
print("模块六：sklearn 线性回归对照")
print("=" * 50)

# 1. 构建完整 Pipeline（预处理 + 回归器）
model = Pipeline([
    ("preprocessor", preprocessor),      # 预处理步骤
    ("regressor", LinearRegression())    # 线性回归模型
])

# 2. 训练模型
print("正在训练 sklearn 线性回归模型...")
model.fit(X_train, y_train)
print("训练完成！")

# 3. 预测
y_test_pred_sklearn = model.predict(X_test)

# 4. 评估
mse_sklearn = mean_squared_error(y_test, y_test_pred_sklearn)
r2_sklearn = r2_score(y_test, y_test_pred_sklearn)

print("\n" + "=" * 50)
print("sklearn 线性回归结果")
print("=" * 50)
print(f"MSE (均方误差): {mse_sklearn:.4f}")
print(f"RMSE (均方根误差): {np.sqrt(mse_sklearn):.4f}")
print(f"R² (决定系数): {r2_sklearn:.4f}")

# 获取模型系数（需要先获取特征名称）
regressor = model.named_steps['regressor']
print(f"\n模型系数统计:")
print(f"  截距项: {regressor.intercept_:.4f}")
print(f"  系数数量: {len(regressor.coef_)}")
print(f"  系数均值: {np.mean(regressor.coef_):.4f}")
print(f"  系数标准差: {np.std(regressor.coef_):.4f}")

# 5. 对比两个模型
print("\n" + "=" * 50)
print("结果对比")
print("=" * 50)
print(f"{'指标':<15} {'手写GD':<15} {'sklearn':<15}")
print("-" * 45)
print(f"{'MSE':<15} {mse_gd:.6f}      {mse_sklearn:.6f}")
print(f"{'RMSE':<15} {np.sqrt(mse_gd):.6f}      {np.sqrt(mse_sklearn):.6f}")
print(f"{'R²':<15} {r2_gd:.6f}      {r2_sklearn:.6f}")

# 判断是否接近
diff_mse = abs(mse_gd - mse_sklearn)
diff_r2 = abs(r2_gd - r2_sklearn)
print(f"\n差异分析:")
print(f"  MSE差异: {diff_mse:.6f}")
print(f"  R²差异: {diff_r2:.6f}")

if diff_mse < 0.01 and diff_r2 < 0.01:
    print("  ✅ 两个模型结果非常接近，手写GD实现正确！")
else:
    print("  ⚠️ 两个模型存在差异，请检查参数设置")


# In[15]:


# ============================================
# 模块七：残差分析
# ============================================

print("=" * 50)
print("模块七：残差分析")
print("=" * 50)

# 1. 计算残差
residuals = y_test - y_test_pred_sklearn

# 2. 残差图（Residual Plot）
plt.figure(figsize=(10, 5))
plt.scatter(y_test_pred_sklearn, residuals, alpha=0.5, s=10)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title("Residual Plot", fontsize=14)
plt.xlabel("Predicted Yield", fontsize=12)
plt.ylabel("Residual", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. 残差分布直方图
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.title("Residual Distribution", fontsize=14)
plt.xlabel("Residual", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. 真实值 vs 预测值散点图
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred_sklearn, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.title("True vs Predicted Yield", fontsize=14)
plt.xlabel("True Yield", fontsize=12)
plt.ylabel("Predicted Yield", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. 残差统计分析
print("\n" + "=" * 50)
print("残差统计分析")
print("=" * 50)
print(f"残差均值: {residuals.mean():.6f}")
print(f"残差标准差: {residuals.std():.6f}")
print(f"残差最小值: {residuals.min():.6f}")
print(f"残差最大值: {residuals.max():.6f}")
print(f"残差中位数: {residuals.median():.6f}")
print(f"残差25%分位数: {residuals.quantile(0.25):.6f}")
print(f"残差75%分位数: {residuals.quantile(0.75):.6f}")

# 6. 残差正态性检验
try:
    from scipy import stats
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Shapiro-Wilk检验（样本限制5000）
    print(f"\nShapiro-Wilk正态性检验:")
    print(f"  统计量: {shapiro_stat:.4f}")
    print(f"  p值: {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print("  结论: 残差符合正态分布 (p > 0.05)")
    else:
        print("  结论: 残差不符合正态分布 (p < 0.05)")
except:
    print("\n无法进行Shapiro-Wilk检验（可能未安装scipy）")

# 7. 残差模式判断
print("\n" + "=" * 50)
print("残差模式判断")
print("=" * 50)

# 判断残差均值是否接近0
if abs(residuals.mean()) < 0.05:
    print("✅ 残差均值接近0，无系统性偏差")
else:
    print(f"❌ 残差均值偏离0 ({residuals.mean():.4f})，存在系统性偏差")

# 判断残差是否随机分布
# 计算残差与预测值的相关系数
corr_resid_pred = np.corrcoef(y_test_pred_sklearn, residuals)[0, 1]
print(f"残差与预测值相关系数: {corr_resid_pred:.4f}")
if abs(corr_resid_pred) < 0.1:
    print("✅ 残差与预测值无明显相关性，模型拟合良好")
else:
    print("❌ 残差与预测值存在相关性，模型可能存在非线性关系")

# 8. 残差与时间的关系（检验时间自相关性）
# 按时间顺序排列残差
df_resid_time = pd.DataFrame({
    'timestamp': df['timestamp'].iloc[split_idx:].reset_index(drop=True),
    'residuals': residuals.values
})
df_resid_time = df_resid_time.sort_values('timestamp')

plt.figure(figsize=(12, 5))
plt.plot(df_resid_time['timestamp'][:200], df_resid_time['residuals'][:200], 
         linewidth=1, marker='o', markersize=2)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title("Residuals Over Time (first 200 test points)", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Residual", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("残差分析总结")
print("=" * 50)

# 9. 综合诊断
issues = []
if abs(residuals.mean()) > 0.05:
    issues.append("残差均值不为0，存在系统性偏差")
if abs(corr_resid_pred) > 0.1:
    issues.append("残差与预测值相关，可能存在非线性关系")
if residuals.std() > y_test.std() * 0.5:
    issues.append("残差标准差较大，预测精度有待提高")

if len(issues) == 0:
    print("✅ 模型诊断良好：")
    print("   - 残差围绕0随机分布")
    print("   - 无明显系统性偏差")
    print("   - 模型基本满足线性回归假设")
else:
    print("⚠️ 模型存在问题：")
    for issue in issues:
        print(f"   - {issue}")
    print("\n💡 建议：")
    print("   - 考虑使用非线性模型（如随机森林、XGBoost）")
    print("   - 考虑引入特征交互项")
    print("   - 考虑使用时序模型（如LSTM）")


# In[ ]:




