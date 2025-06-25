import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def simulate_buffon_needle(num_trials, needle_length=1.0, line_distance=1.0):
    """
    模拟布冯投针实验
    
    参数:
    num_trials (int): 实验次数
    needle_length (float): 针的长度
    line_distance (float): 平行线之间的距离
    
    返回:
    float: 估算的π值
    int: 与线相交的次数
    """
    # 设置Decimal精度以处理大数字
    getcontext().prec = 50
    
    # 生成随机落点和角度
    center_points = np.random.uniform(0, line_distance/2, num_trials)
    angles = np.random.uniform(0, np.pi/2, num_trials)
    
    # 计算针的端点到最近线的距离
    distances = center_points - (needle_length/2) * np.sin(angles)
    
    # 统计相交次数
    crossings = np.sum(distances <= 0)
    
    # 计算π的估计值
    if crossings == 0:
        return None, crossings
    
    pi_estimate = (2 * needle_length * num_trials) / (line_distance * crossings)
    return pi_estimate, crossings

def analyze_accuracy():
    """
    分析不同实验次数对π估计精度的影响
    """
    # 实验次数列表
    trial_counts = [100, 1000, 10000, 100000, 1000000, 10000000]
    # 每种实验次数重复的次数
    repetitions = 5
    
    results = []
    
    for trials in trial_counts:
        estimates = []
        for _ in range(repetitions):
            pi_est, _ = simulate_buffon_needle(trials)
            if pi_est is not None:
                estimates.append(pi_est)
        
        # 计算平均值和误差
        avg_pi = np.mean(estimates)
        error = abs(avg_pi - np.pi)
        
        results.append((trials, avg_pi, error))
        
        print(f"实验次数: {trials:,}")
        print(f"π的平均估计值: {avg_pi:.8f}")
        print(f"与真实值的绝对误差: {error:.8f}")
        print("-" * 40)
    
    # 绘制实验次数与误差的关系图
    plt.figure(figsize=(10, 6))
    plt.plot([t for t, _, _ in results], [e for _, _, e in results], 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('实验次数 (对数刻度)')
    plt.ylabel('绝对误差 (对数刻度)')
    plt.title('实验次数对π估计精度的影响')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig('buffon_needle_accuracy.png')
    plt.show()

if __name__ == "__main__":
    # 运行单次实验
    num_trials = 1000000
    pi_estimate, crossings = simulate_buffon_needle(num_trials)
    
    if pi_estimate is not None:
        print(f"实验次数: {num_trials:,}")
        print(f"相交次数: {crossings:,}")
        print(f"π的估计值: {pi_estimate:.8f}")
        print(f"真实π值: {np.pi:.8f}")
        print(f"绝对误差: {abs(pi_estimate - np.pi):.8f}")
    else:
        print("未观察到相交事件，无法计算π的估计值。")
    
    # 分析实验次数对精度的影响
    analyze_accuracy()
