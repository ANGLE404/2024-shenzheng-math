import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic

def geodetic_to_cartesian(lon, lat, alt):
    # WGS84椭球体参数
    a = 6378137.0  # 地球赤道半径（米）
    f = 1 / 298.257223563  # 扁率
    b = a * (1 - f)  # 极半径
    # 角度转弧度
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    # 辅助值
    cosLat = np.cos(lat_rad)
    sinLat = np.sin(lat_rad)
    cosLon = np.cos(lon_rad)
    sinLon = np.sin(lon_rad)
    # 第一偏心率平方
    e_sq = (a**2 - b**2) / a**2
    N = a / np.sqrt(1 - e_sq * sinLat**2)
    # 计算笛卡尔坐标
    X = (N + alt) * cosLat * cosLon
    Y = (N + alt) * cosLat * sinLon
    Z = ((b**2 / a**2) * N + alt) * sinLat
    return (X, Y, Z)

# 定义声速常量，单位为米每秒
speed_of_sound = 340  # m/s

# 定义设备的位置数据，每个设备包括经度、纬度和高度
devices = {
    'A': [110.241, 27.204, 814],
    'B': [110.783, 27.256, 627],
    'C': [110.362, 27.785, 742],
    'D': [110.351, 27.945, 850],
    'E': [110.524, 27.617, 711],
    'F': [110.447, 27.081, 648],
    'G': [110.947, 27.521, 987]
}


# 将设备数据转换为numpy数组以便进行数学运算
monitors_data = np.array(list(devices.values()))

# 随机生成残骸的真实位置和时间
num_debris = 4
debris_spread = 0.05  # 分散范围（经纬度）

# 生成残骸位置，使其围绕设备位置稍微分散
true_debris_positions = np.array([
    [
        np.random.uniform(lon - debris_spread, lon + debris_spread),
        np.random.uniform(lat - debris_spread, lat + debris_spread),
        np.random.uniform(alt - 50, alt + 50)
    ]
    for lon, lat, alt in monitors_data[np.random.choice(len(monitors_data), num_debris, replace=False)]
])

# 转换4个随机残骸位置为笛卡尔坐标
true_debris_cartesian_positions = np.array([geodetic_to_cartesian(*true_debris_positions[i]) for i in np.random.choice(len(true_debris_positions), 4, replace=False)])
# 将设备的地理位置转换为笛卡尔坐标
device_cartesian_positions = {}
for device, (lon, lat, alt) in devices.items():
    cartesian_coords = geodetic_to_cartesian(lon, lat, alt)
    device_cartesian_positions[device] = cartesian_coords

# 生成时间范围100到200秒
true_debris_times = np.random.uniform(-100, 60, num_debris)

# 输出生成的残骸位置和时间信息
for i in range(num_debris):
    print(f"残骸{i+1}的位置: 经度 {true_debris_positions[i][0]}, 纬度 {true_debris_positions[i][1]}, 高度 {true_debris_positions[i][2]} 米")
    print(f"残骸{i+1}的时间: {true_debris_times[i]} 秒")
# 输出每个残骸的笛卡尔坐标
for i, coord in enumerate(true_debris_cartesian_positions):
    print(f"残骸{i+1}的笛卡尔坐标: X={coord[0]}, Y={coord[1]}, Z={coord[2]}")
# 遍历设备字典，输出每个设备的笛卡尔坐标
for device, coords in device_cartesian_positions.items():
    print(f"设备 {device} 的笛卡尔坐标: X={coords[0]}, Y={coords[1]}, Z={coords[2]}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 设置matplotlib的字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 监测设备和残骸位置的可视化


def plot_devices_and_debris(devices, debris_positions):
    # 创建3D图表
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制设备位置
    for key, (lon, lat, alt) in devices.items():
        ax.scatter(lon, lat, alt, s=100, label=f'设备 {key}', color='blue')
        
    # 绘制残骸位置
    for i, (lon, lat, alt) in enumerate(debris_positions):
        ax.scatter(lon, lat, alt, marker='s', s=100, label=f'残骸 {i+1}', color='green')
        # 对每个残骸，连接所有设备
        for device_key, device_coords in devices.items():
            ax.plot([lon, device_coords[0]], [lat, device_coords[1]], [alt, device_coords[2]], 'r--', linewidth=1)

    # 设置图表标题和坐标轴标签
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_zlabel('高度')
    ax.set_title('设备与残骸位置 3D 视图')
    # 设置坐标轴范围
    ax.set_xlim([110, 111])
    ax.set_ylim([27, 28])
    # 调整图例位置
    ax.legend(loc='upper left', fontsize='small')
    
    # 显示图表
    plt.show()
    # 保存图表到本地文件
    fig.savefig('设备与残骸位置图.png')

# 调用绘图函数
plot_devices_and_debris(devices, true_debris_positions)
def calculate_arrival_times(debris_positions, debris_times, monitors):
    # 初始化到达时间数组
    arrival_times = np.zeros((num_debris, len(monitors)))
    
    # 转换监测设备的地理坐标到笛卡尔坐标
    monitor_cartesian = np.array([geodetic_to_cartesian(lon, lat, alt) for lon, lat, alt in monitors])
    
    # 计算每个残骸到每个监测设备的到达时间
    for i, (debris_pos, debris_time) in enumerate(zip(debris_positions, debris_times)):
        # 将残骸位置从地理坐标转换为笛卡尔坐标
        debris_cartesian = geodetic_to_cartesian(*debris_pos)
        
        # 计算三维空间中的距离
        distances = np.sqrt(np.sum((monitor_cartesian - debris_cartesian) ** 2, axis=1))
        
        # 计算到达时间
        arrival_times[i, :] = distances / speed_of_sound + debris_time
        
    return arrival_times


def objective_function(variables, monitor_times, monitors, num_debris):
    # 解析优化变量，包括预测的位置和时间
    predicted_positions = variables[:num_debris * 3].reshape(num_debris, 3)
    predicted_times = variables[num_debris * 3:]
    # 确保监测设备坐标也是笛卡尔坐标
    monitor_cartesian = np.array([geodetic_to_cartesian(lon, lat, alt) for lon, lat, alt in monitors])
    # 计算预测的到达时间
    predicted_arrival_times = calculate_arrival_times(predicted_positions, predicted_times, monitor_cartesian)
    # 计算并返回目标函数值
    return np.sum((predicted_arrival_times - monitor_times) ** 2)




# 初始化最小波动性和最佳设备数量
min_volatility = float('inf')
best_configurations = []
num_iterations = 7

# 进行多次迭代以找到最佳设备配置
while len(best_configurations) < num_iterations:
    # 随机生成不同数量的监测设备
    num_monitors = np.random.randint(4, 8)  # 随机选择设备数量
    monitors = monitors_data[np.random.choice(len(devices), num_monitors, replace=False)]

    # 验证生成的残骸位置是否有效
    valid_debris_positions = []
    for _ in range(num_debris):
        while True:
            pos = np.random.rand(3) * 1000
            if np.all(np.sqrt(np.sum((monitors[:, :2] - pos[:2]) ** 2, axis=1)) < 1000):
                valid_debris_positions.append(pos)
                break

    true_debris_positions = np.array(valid_debris_positions)

    # 计算模拟的到达时间
    simulated_arrival_times = calculate_arrival_times(true_debris_positions, true_debris_times, monitors)

    # 使用优化算法估计残骸的位置和时间
    initial_guess = np.concatenate((np.random.rand(num_debris * 3), np.random.rand(num_debris)))
    result = minimize(objective_function, initial_guess, args=(simulated_arrival_times, monitors, num_debris))
    estimated_debris_positions = result.x[:num_debris * 3].reshape(num_debris, 3)
    estimated_debris_times = result.x[num_debris * 3:]

    # 计算波动性
    volatility = np.mean((true_debris_positions - estimated_debris_positions) ** 2) + \
                 np.mean((true_debris_times - estimated_debris_times) ** 2)

    # 保存结果
    if num_monitors == 7:
        best_configurations.append((volatility, num_monitors, tuple(map(tuple, monitors))))

# 统计设备被选中的次数
device_counts = {key: 0 for key in devices.keys()}
for _, _, monitors in best_configurations:
    for device in monitors:
        device_counts[chr(65 + np.where((monitors_data == device).all(axis=1))[0][0])] += 1

# 找到出现次数最多的设备组合
most_common_configuration = max(set(best_configurations), key=best_configurations.count)

# 输出最佳设备组合结果
print("最佳设备组合:")
for i, device in enumerate(most_common_configuration[2]):
    print(f"设备 {chr(65 + i)}: 经度 {device[0]}, 纬度 {device[1]}, 高度 {device[2]}")
print(f"最佳监测设备数量: {most_common_configuration[1]}")



