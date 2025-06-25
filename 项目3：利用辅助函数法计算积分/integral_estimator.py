import numpy as np

def generate_random_numbers(N):
    """生成满足权重函数 p(x) = 1/(2√x) 分布的随机数"""
    # 从均匀分布 [0,1) 生成随机数
    u = np.random.rand(N)
    # 通过逆变换方法生成满足 p(x) 分布的随机数
    return u**2

def integrand(x):
    """原始被积函数 f(x) = x^(-1/2) e^(x+1)"""
    return x**(-0.5) * np.exp(x + 1)

def weight_function(x):
    """权重函数 p(x) = 1/(2√x)"""
    return 1 / (2 * np.sqrt(x))

def estimate_integral(N):
    """使用重要性采样估计积分"""
    # 生成满足权重函数分布的随机数
    x_samples = generate_random_numbers(N)
    
    # 计算每个样本点的 f(x)/p(x)
    ratios = integrand(x_samples) / weight_function(x_samples)
    
    # 计算积分估计值
    integral_estimate = np.mean(ratios)
    
    # 计算方差和标准误差
    f_squared_mean = np.mean(ratios**2)
    f_mean_squared = np.mean(ratios)**2
    variance = f_squared_mean - f_mean_squared
    standard_error = np.sqrt(variance / N)
    
    return integral_estimate, standard_error

# 执行积分估计
N = 1000000
integral, error = estimate_integral(N)

print(f"积分估计值: {integral:.6f}")
print(f"统计误差: {error:.6f}")
print(f"结果范围: [{integral - error:.6f}, {integral + error:.6f}]")
