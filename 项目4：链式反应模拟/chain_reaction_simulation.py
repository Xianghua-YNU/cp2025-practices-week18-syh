import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import sys

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class ChainReactionSimulation:
    def __init__(self, size=20, initial_neutrons=5, fission_prob=0.8, absorption_prob=0.1, 
                 leakage_prob=0.1, reproduction_ratio=2.5, max_steps=100):
        """
        初始化链式反应模拟参数
        
        参数:
        size: 模拟网格大小
        initial_neutrons: 初始中子数
        fission_prob: 中子撞击原子核引发裂变的概率
        absorption_prob: 中子被吸收的概率
        leakage_prob: 中子泄漏出系统的概率
        reproduction_ratio: 每次裂变产生的平均中子数
        max_steps: 最大模拟步数
        """
        self.size = size
        self.initial_neutrons = initial_neutrons
        self.fission_prob = fission_prob
        self.absorption_prob = absorption_prob
        self.leakage_prob = leakage_prob
        self.reproduction_ratio = reproduction_ratio
        self.max_steps = max_steps
        
        # 初始化网格和中子计数器
        self.grid = np.zeros((size, size), dtype=int)  # 网格中每个位置的原子核数
        self.neutrons = []  # 当前中子的位置列表
        self.neutron_count_history = []  # 中子数随时间变化
        self.fission_count_history = []  # 裂变数随时间变化
        
        self.initialize_simulation()
    
    def initialize_simulation(self):
        """初始化模拟状态"""
        # 均匀分布原子核
        self.grid.fill(10)
        
        # 随机放置初始中子
        for _ in range(self.initial_neutrons):
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            self.neutrons.append((x, y))
        
        # 初始化历史记录
        self.neutron_count_history = [self.initial_neutrons]
        self.fission_count_history = [0]
    
    def simulate_step(self):
        """模拟一个时间步的链式反应"""
        new_neutrons = []
        fission_count = 0
        
        # 处理每个中子
        for x, y in self.neutrons:
            # 检查中子是否泄漏
            if random.random() < self.leakage_prob:
                continue
                
            # 检查中子是否被吸收
            if random.random() < self.absorption_prob:
                continue
                
            # 检查是否发生裂变
            if self.grid[x, y] > 0 and random.random() < self.fission_prob:
                # 发生裂变
                self.grid[x, y] -= 1  # 消耗一个原子核
                fission_count += 1
                
                # 产生新的中子
                n_new_neutrons = max(2, int(np.random.poisson(self.reproduction_ratio)))
                
                for _ in range(n_new_neutrons):
                    # 新中子随机移动到相邻位置
                    dx = random.randint(-1, 1)
                    dy = random.randint(-1, 1)
                    new_x = (x + dx) % self.size
                    new_y = (y + dy) % self.size
                    new_neutrons.append((new_x, new_y))
        
        # 更新中子列表
        self.neutrons = new_neutrons
        self.neutron_count_history.append(len(self.neutrons))
        self.fission_count_history.append(fission_count)
        
        return len(self.neutrons), fission_count
    
    def run_simulation(self):
        """运行完整的模拟过程"""
        results = []
        
        for step in range(self.max_steps):
            neutron_count, fission_count = self.simulate_step()
            results.append((step, neutron_count, fission_count))
            
            # 如果没有中子了，结束模拟
            if neutron_count == 0:
                break
                
        return results
    
    def visualize_results(self):
        """可视化模拟结果"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.neutron_count_history, label='中子数')
        plt.plot(self.fission_count_history, label='裂变数')
        plt.xlabel('时间步')
        plt.ylabel('数量')
        plt.title('链式反应随时间的演化')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        final_grid = self.grid.copy()
        # 标记最后一步中子的位置
        for x, y in self.neutrons:
            final_grid[x, y] = -1
        plt.imshow(final_grid, cmap='viridis')
        plt.colorbar(label='原子核数')
        plt.title('最终状态')
        plt.tight_layout()
        plt.show()

    def create_animation(self):
        """创建模拟过程的动画"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 初始化中子数和裂变数的图表
        line_neutrons, = ax1.plot([], [], 'b-', label='中子数')
        line_fissions, = ax1.plot([], [], 'r-', label='裂变数')
        ax1.set_xlim(0, self.max_steps)
        ax1.set_ylim(0, max(10, max(self.neutron_count_history)))
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('数量')
        ax1.set_title('链式反应随时间的演化')
        ax1.legend()
        ax1.grid(True)
        
        # 初始化网格状态图
        grid_data = self.grid.copy()
        for x, y in self.neutrons:
            grid_data[x, y] = -1
        im = ax2.imshow(grid_data, cmap='viridis')
        ax2.set_title('模拟网格状态')
        fig.colorbar(im, ax=ax2, label='原子核数')
        
        # 初始化模拟
        self.initialize_simulation()
        
        def update(frame):
            # 模拟一步
            if frame < self.max_steps and len(self.neutrons) > 0:
                self.simulate_step()
                
            # 更新中子数和裂变数图表
            line_neutrons.set_data(range(len(self.neutron_count_history)), self.neutron_count_history)
            line_fissions.set_data(range(len(self.fission_count_history)), self.fission_count_history)
            ax1.set_ylim(0, max(10, max(self.neutron_count_history)))
            
            # 更新网格状态图
            grid_data = self.grid.copy()
            for x, y in self.neutrons:
                grid_data[x, y] = -1
            im.set_data(grid_data)
            
            return line_neutrons, line_fissions, im,
        
        ani = FuncAnimation(fig, update, frames=range(self.max_steps + 1), 
                            interval=200, blit=True, repeat=False)
        
        plt.tight_layout()
        return ani

def analyze_parameter_effect(parameter_name, parameter_values):
    """
    分析不同参数值对链式反应结果的影响
    
    参数:
    parameter_name: 要分析的参数名称
    parameter_values: 要测试的参数值列表
    """
    plt.figure(figsize=(10, 6))
    
    for value in parameter_values:
        # 创建模拟实例，修改要测试的参数
        if parameter_name == 'fission_prob':
            sim = ChainReactionSimulation(fission_prob=value)
        elif parameter_name == 'absorption_prob':
            sim = ChainReactionSimulation(absorption_prob=value)
        elif parameter_name == 'leakage_prob':
            sim = ChainReactionSimulation(leakage_prob=value)
        elif parameter_name == 'reproduction_ratio':
            sim = ChainReactionSimulation(reproduction_ratio=value)
        else:
            raise ValueError(f"未知参数: {parameter_name}")
        
        # 运行模拟
        sim.run_simulation()
        
        # 绘制结果
        plt.plot(sim.neutron_count_history, label=f'{parameter_name}={value}')
    
    plt.xlabel('时间步')
    plt.ylabel('中子数')
    plt.title(f'不同{parameter_name}值对链式反应的影响')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 创建模拟实例
    sim = ChainReactionSimulation(
        size=20,
        initial_neutrons=5,
        fission_prob=0.8,
        absorption_prob=0.1,
        leakage_prob=0.1,
        reproduction_ratio=2.5,
        max_steps=50
    )
    
    # 运行模拟
    results = sim.run_simulation()
    
    # 输出结果摘要
    print("链式反应模拟结果:")
    print(f"总模拟步数: {len(results)}")
    print(f"最大中子数: {max(sim.neutron_count_history)}")
    print(f"总裂变数: {sum(sim.fission_count_history)}")
    
    # 可视化结果
    sim.visualize_results()
    
    # 分析参数影响
    if len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        analyze_parameter_effect('reproduction_ratio', [1.8, 2.0, 2.2, 2.5])
        analyze_parameter_effect('fission_prob', [0.6, 0.7, 0.8, 0.9])    
