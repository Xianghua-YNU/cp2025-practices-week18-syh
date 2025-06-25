import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import ipywidgets as widgets
from IPython.display import display

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class HydrogenAtomSimulator:
    """氢原子电子云模拟器"""
    
    def __init__(self):
        # 物理常数（单位：nm）
        self.bohr_radius = 5.29e-2  # 玻尔半径 (nm)
        self.max_density = 1.1      # 最大概率密度
        self.convergence_radius = 0.25  # 收敛半径 (nm)
        
    def probability_density(self, r, a=None):
        """
        计算氢原子基态(n=1, l=0, m=0)的电子概率密度
        
        参数:
            r: 距离原子核的距离 (nm)
            a: 玻尔半径，默认为物理值
        
        返回:
            概率密度值
        """
        if a is None:
            a = self.bohr_radius
            
        # 波函数概率密度公式: D(r) = (4r²/a³) * e^(-2r/a)
        return (4 * r**2 / a**3) * np.exp(-2 * r / a)
    
    def generate_electron_cloud(self, num_points=10000, a=None, r_max=None):
        """
        生成电子云的三维点分布
        
        参数:
            num_points: 生成的点数
            a: 玻尔半径参数
            r_max: 最大模拟半径
        
        返回:
            x, y, z: 三维坐标点
            densities: 对应点的概率密度
        """
        if a is None:
            a = self.bohr_radius
            
        if r_max is None:
            r_max = self.convergence_radius * 3
        
        # 生成随机点
        r = np.random.uniform(0, r_max, num_points)
        theta = np.random.uniform(0, np.pi, num_points)
        phi = np.random.uniform(0, 2*np.pi, num_points)
        
        # 转换为笛卡尔坐标
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # 计算每个点的概率密度
        densities = self.probability_density(r, a)
        
        # 根据概率密度进行采样筛选
        max_density = self.probability_density(a, a)  # 最大值出现在r=a处
        probabilities = densities / max_density
        mask = np.random.rand(num_points) < probabilities
        
        return x[mask], y[mask], z[mask], densities[mask]
    
    def plot_3d_electron_cloud(self, x, y, z, densities, a=None, title=None):
        """
        绘制电子云的三维分布图
        
        参数:
            x, y, z: 三维坐标点
            densities: 概率密度值
            a: 玻尔半径参数
            title: 图表标题
        """
        if a is None:
            a = self.bohr_radius
            
        if title is None:
            title = f"氢原子基态电子云分布 (a = {a:.4f} nm)"
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 归一化密度值以用于颜色映射
        norm = Normalize(vmin=0, vmax=np.percentile(densities, 95))
        colors = cm.jet(norm(densities))
        
        # 绘制散点图
        scatter = ax.scatter(x, y, z, c=colors, s=1, alpha=0.5, edgecolors='none')
        
        # 添加颜色条
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.jet), ax=ax)
        cbar.set_label('概率密度')
        
        # 绘制玻尔半径参考球
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x_sphere = a * np.cos(u) * np.sin(v)
        y_sphere = a * np.sin(u) * np.sin(v)
        z_sphere = a * np.cos(v)
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='r', alpha=0.3, linewidth=1)
        
        # 设置图表属性
        max_range = np.max([np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))])
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_radial_distribution(self, a=None, r_max=None, title=None):
        """
        绘制径向分布函数图
        
        参数:
            a: 玻尔半径参数
            r_max: 最大半径
            title: 图表标题
        """
        if a is None:
            a = self.bohr_radius
            
        if r_max is None:
            r_max = self.convergence_radius * 3
            
        if title is None:
            title = f"氢原子基态径向分布函数 (a = {a:.4f} nm)"
        
        r = np.linspace(0, r_max, 1000)
        p = self.probability_density(r, a)
        
        fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
        
        # 绘制径向分布函数
        ax.plot(r, p, 'b-', linewidth=2)
        
        # 标记玻尔半径位置
        ax.axvline(x=a, color='r', linestyle='--', alpha=0.7)
        ax.text(a, max(p)*0.9, f'玻尔半径 a={a:.4f} nm', rotation=90, va='top')
        
        # 标记最大概率密度位置
        r_max_prob = 2 * a  # 最大概率密度出现在r=2a处
        p_max = self.probability_density(r_max_prob, a)
        ax.scatter([r_max_prob], [p_max], color='g', s=50, zorder=5)
        ax.text(r_max_prob, p_max, f'最大密度: r={r_max_prob:.4f} nm', va='bottom')
        
        # 设置图表属性
        ax.set_xlabel('距离原子核的距离 r (nm)')
        ax.set_ylabel('概率密度 D(r)')
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig, ax
    
    def interactive_simulation(self):
        """创建交互式模拟界面"""
        @widgets.interact(
            bohr_radius=widgets.FloatSlider(value=self.bohr_radius, min=0.01, max=0.1, step=0.005, 
                                           description='玻尔半径 (nm):'),
            num_points=widgets.IntSlider(value=10000, min=1000, max=50000, step=1000, 
                                        description='模拟点数:'),
            show_radial=widgets.Checkbox(value=True, description='显示径向分布图')
        )
        def update_simulation(bohr_radius, num_points, show_radial):
            # 生成电子云数据
            x, y, z, densities = self.generate_electron_cloud(num_points, bohr_radius)
            
            # 绘制3D电子云
            fig1, ax1 = self.plot_3d_electron_cloud(x, y, z, densities, bohr_radius)
            plt.show()
            
            # 绘制径向分布图
            if show_radial:
                fig2, ax2 = self.plot_radial_distribution(bohr_radius)
                plt.show()
    
    def parameter_analysis(self):
        """分析不同参数对电子云分布的影响"""
        # 不同玻尔半径值
        a_values = [self.bohr_radius * 0.5, self.bohr_radius, self.bohr_radius * 1.5]
        labels = ['0.5a₀', 'a₀', '1.5a₀']
        
        # 绘制不同参数下的径向分布函数
        fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
        
        r_max = self.convergence_radius * 3
        r = np.linspace(0, r_max, 1000)
        
        for a, label in zip(a_values, labels):
            p = self.probability_density(r, a)
            ax.plot(r, p, linewidth=2, label=label)
            
            # 标记最大概率密度位置
            r_max_prob = 2 * a
            p_max = self.probability_density(r_max_prob, a)
            ax.scatter([r_max_prob], [p_max], s=50, zorder=5)
        
        ax.set_xlabel('距离原子核的距离 r (nm)')
        ax.set_ylabel('概率密度 D(r)')
        ax.set_title('不同玻尔半径对电子云径向分布的影响', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 分析不同参数下的电子云形状
        fig = plt.figure(figsize=(18, 5))
        
        for i, (a, label) in enumerate(zip(a_values, labels)):
            x, y, z, densities = self.generate_electron_cloud(10000, a)
            
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            
            # 归一化密度值以用于颜色映射
            norm = Normalize(vmin=0, vmax=np.percentile(densities, 95))
            colors = cm.jet(norm(densities))
            
            # 绘制散点图
            ax.scatter(x, y, z, c=colors, s=1, alpha=0.5, edgecolors='none')
            
            # 设置图表属性
            max_range = np.max([np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))])
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_zlabel('Z (nm)')
            ax.set_title(f'电子云分布 ({label})', fontsize=12)
        
        plt.tight_layout()
        plt.show()

# 主程序
if __name__ == "__main__":
    # 创建模拟器实例
    simulator = HydrogenAtomSimulator()
    
    # 运行交互式模拟
    print("=== 氢原子电子云模拟程序 ===")
    print("1. 运行交互式模拟")
    print("2. 分析不同参数对电子云分布的影响")
    choice = input("请选择操作 (1/2): ")
    
    if choice == '1':
        simulator.interactive_simulation()
    elif choice == '2':
        simulator.parameter_analysis()
    else:
        print("无效选择，程序退出。")    
