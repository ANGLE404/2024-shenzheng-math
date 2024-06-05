import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


speed_of_sound = 340
num_debris = 4

# WGS84椭球模型参数
a = 6378137.0  # 赤道半径（米）
b = 6356752.3142  # 极半径（米）
e2 = (a**2 - b**2) / a**2  # 第一偏心率平方

def geographic_to_cartesian(lon, lat, h):
    """将地理坐标转换为笛卡尔坐标"""
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    x = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (b**2 / a**2 * N + h) * np.sin(lat_rad)
    return x, y, z

def calculate_theoretical_arrival_times(debris_positions, debris_times, monitors):
    """计算理论到达时间"""
    arrival_times = np.zeros((len(debris_positions), len(monitors)))
    for i, (debris_pos, debris_time) in enumerate(zip(debris_positions, debris_times)):
        distances = np.sqrt(np.sum((monitors - debris_pos) ** 2, axis=1))
        arrival_times[i, :] = debris_time + distances / speed_of_sound
    return arrival_times

def objective_function(variables, monitor_times, monitors, num_debris):
    """目标函数，计算时间差的平方和"""
    debris_positions = variables[:num_debris * 3].reshape(num_debris, 3)
    debris_times = variables[num_debris * 3:]
    predicted_arrival_times = calculate_theoretical_arrival_times(debris_positions, debris_times, monitors)
    return np.sum((predicted_arrival_times - monitor_times) ** 2)

def objective_function(variables, monitor_times, monitors, num_debris):
    """目标函数，计算时间差的平方和，添加对设备数量的偏好"""
    debris_positions = variables[:num_debris * 3].reshape(num_debris, 3)
    debris_times = variables[num_debris * 3:]
    predicted_arrival_times = calculate_theoretical_arrival_times(debris_positions, debris_times, monitors)
    time_diffs = np.sum((predicted_arrival_times - monitor_times) ** 2)
    # 添加对设备数量的偏好，偏好设备数量为7
    penalty = 0 if len(monitors) == 7 else 1000
    return time_diffs + penalty





# 统计最佳监测设备数量出现次数
best_monitor_counts = {}

# 迭代运行100次
# 在迭代过程中调整初始猜测
for _ in tqdm(range(100), desc="Processing"):
    true_debris_positions = np.random.rand(num_debris, 3) * 1000  # 单位：米
    true_debris_times = np.random.rand(num_debris) * 10  # 单位：秒
    min_volatility = float('inf')
    best_num_monitors = 0
    best_estimates = None

    for num_monitors in range(4, 8):  # 从4到7的设备数量
        monitors = np.random.rand(num_monitors, 3) * 1000  # 随机生成监测设备位置
        simulated_arrival_times = calculate_theoretical_arrival_times(true_debris_positions, true_debris_times, monitors)
        initial_guess = np.concatenate((np.random.rand(num_debris * 3), np.random.rand(num_debris)))
        result = minimize(objective_function, initial_guess, args=(simulated_arrival_times, monitors, num_debris), method='L-BFGS-B')
        estimated_debris_positions = result.x[:num_debris * 3].reshape(num_debris, 3)
        estimated_debris_times = result.x[num_debris * 3:]
        volatility = np.mean((true_debris_positions - estimated_debris_positions) ** 2) + \
                     np.mean((true_debris_times - estimated_debris_times) ** 2)
        if volatility < min_volatility:
            min_volatility = volatility
            best_num_monitors = num_monitors

    # 更新统计数据
    if best_num_monitors in best_monitor_counts:
        best_monitor_counts[best_num_monitors] += 1
    else:
        best_monitor_counts[best_num_monitors] = 1

# 输出统计结果
print("最佳监测设备数量统计结果:", best_monitor_counts)

# 绘制监测设备数量的统计结果
plt.figure(figsize=(8, 6))
plt.bar(best_monitor_counts.keys(), best_monitor_counts.values(), color='skyblue')
plt.xlabel('监测设备数量')
plt.ylabel('出现次数')
plt.title('监测设备数量的统计分布')
plt.grid(True)
plt.show()