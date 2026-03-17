import numpy as np
from scipy.stats import beta
from sklearn.neighbors import KNeighborsClassifier

# 第一部分：参数估计（MLE与MAP）
def parameter_estimation():
    print("=== 第一部分：参数估计（MLE与MAP） ===")
    
    # 构造0-1数据
    data = np.array([1, 1, 0, 1, 0])
    
    # 计算MLE
    p_mle = np.mean(data)
    
    # 设置Beta先验
    alpha = 2
    beta_param = 2
    p_map = (np.sum(data) + alpha - 1) / (len(data) + alpha + beta_param - 2)
    
    print(f"MLE = {p_mle}")
    print(f"MAP = {p_map}")
    
    # 输出关键数据点
    N = len(data)
    sum_x = np.sum(data)
    print(f"样本量: {N}")
    print(f"正样本数: {sum_x}")
    print(f"先验参数: alpha={alpha}, beta={beta_param}")

# 第二部分：KNN分类
def knn_classification():
    print("\n=== 第二部分：KNN分类 ===")
    
    # 构造二维数据
    X = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 7], [8, 6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    print("训练数据:")
    for i, (x, label) in enumerate(zip(X, y)):
        print(f"样本{i+1}: {x}, 标签: {label}")
    
    # 测试不同K值
    for k in [1, 3, 5]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)
        pred = model.predict(X)
        accuracy = np.mean(pred == y)
        print(f"K={k}, accuracy={accuracy}, 预测结果: {pred}")

# 第三部分：梯度下降
def gradient_descent():
    print("\n=== 第三部分：梯度下降 ===")
    
    def f(x):
        return x ** 2 + 2 * x + 1
    
    def grad(x):
        return 2 * x + 2
    
    # 优化过程
    x = 5.0
    lr = 0.1
    
    print("优化过程:")
    for i in range(20):
        x = x - lr * grad(x)
        print(f"第{i+1}次迭代: x={x:.6f}, f(x)={f(x):.6f}")
    
    # 输出损失曲线数据
    print("\n损失曲线数据:")
    loss = []
    x = 5.0
    
    for i in range(20):
        loss_val = f(x)
        loss.append(loss_val)
        x = x - 0.1 * grad(x)
        if i % 5 == 0:
            print(f"迭代{i+1}: 损失={loss_val:.6f}")

if __name__ == "__main__":
    parameter_estimation()
    knn_classification()
    gradient_descent()
