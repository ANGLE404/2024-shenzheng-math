import numpy as np
from scipy.optimize import minimize

speed_of_sound = 340

# 随机生成残骸的真实位置和时间
num_debris = 4
true_debris_positions = np.random.rand(num_debris, 3) * 1000  # 单位：米
true_debris_times = np.random.rand(num_debris) * 10  # 单位：秒

def calculate_theoretical_arrival_times(debris_positions, debris_times, monitors):
    """计算理论到达时间"""
    arrival_times = np.zeros((len(debris_positions), len(monitors)))
    for i, (debris_pos, debris_time) in enumerate(zip(debris_positions, debris_times)):
        distances = np.sqrt(np.sum((monitors - debris_pos) ** 2, axis=1))
        arrival_times[i, :] = debris_time + distances / speed_of_sound
    return arrival_times

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
def objective_function(variables, monitor_times, monitors, num_debris):
    """目标函数，计算时间差的平方和"""
    debris_positions = variables[:num_debris * 3].reshape(num_debris, 3)
    debris_times = variables[num_debris * 3:]
    predicted_arrival_times = calculate_theoretical_arrival_times(debris_positions, debris_times, monitors)
    return np.sum((predicted_arrival_times - monitor_times) ** 2)

# 评估不同数量的监测设备
min_volatility = float('inf')
best_num_monitors = 0
best_estimates = None

for num_monitors in range(4, 8):  # 从4到7的设备数量
    monitors = np.random.rand(num_monitors, 3) * 1000  # 随机生成监测设备位置
    simulated_arrival_times = calculate_theoretical_arrival_times(true_debris_positions, true_debris_times, monitors)

    # 使用优化算法估计残骸的位置和时间
    initial_guess = np.concatenate((np.random.rand(num_debris * 3), np.random.rand(num_debris)))
    result = minimize(objective_function, initial_guess, args=(simulated_arrival_times, monitors, num_debris))
    estimated_debris_positions = result.x[:num_debris * 3].reshape(num_debris, 3)
    estimated_debris_times = result.x[num_debris * 3:]

    # 计算波动性
    volatility = np.mean((true_debris_positions - estimated_debris_positions) ** 2) + \
                 np.mean((true_debris_times - estimated_debris_times) ** 2)

    # 比较并保存最佳结果
    if volatility < min_volatility:
        min_volatility = volatility
        best_num_monitors = num_monitors
        best_estimates = (estimated_debris_positions, estimated_debris_times)

    # 输出结果
print(f"最佳监测设备数量: {best_num_monitors}")
print("最佳估计的残骸位置:")
print(best_estimates[0])
print("最佳估计的残骸时间:")
print(best_estimates[1])