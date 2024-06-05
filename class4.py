import numpy as np
from pyproj import Transformer
from scipy.optimize import NonlinearConstraint, differential_evolution
from geographiclib.geodesic import Geodesic
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# 设置matplotlib的字体，以支持中文显示
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 在全局作用域中定义 geod 对象
geod = Geodesic.WGS84
# 给定的监测站数据
np.random.seed(42)  # 设置随机种子以保证结果的可重复性
stations = {
    'A': {'coords': (110.241, 27.204, 824), 'times': [100.767, 164.229, 214.850, 270.065]},
    'B': {'coords': (110.783, 27.456, 727), 'times': [92.453, 112.220, 169.362, 196.583]},
    'C': {'coords': (110.762, 27.785, 742), 'times': [75.560, 110.696, 156.936, 188.020]},
    'D': {'coords': (110.251, 28.025, 850), 'times': [94.653, 141.409, 196.517, 258.985]},
    'E': {'coords': (110.524, 27.617, 786), 'times': [78.600, 86.216, 118.443, 126.669]},
    'F': {'coords': (110.467, 28.081, 678), 'times': [67.274, 166.270, 175.482, 266.871]},
    'G': {'coords': (110.047, 27.521, 575), 'times': [103.738, 163.024, 206.789, 210.306]}
}
# 打印每个监测站的时间数据以查看随机值
for station, data in stations.items():
    print(f"增加误差监测站 {station} 的时间数据: {data['times']}")
# 为每个监测站的时间加上随机误差
sigma = 0.5  # 标准偏差为0.5秒

for station in stations.values():
    station['times'] = [time + np.random.uniform(-0.25, 0.25) for time in station['times']]

# 打印每个监测站的时间数据以查看随机值
for station, data in stations.items():
    print(f"增加误差监测站 {station} 的时间数据: {data['times']}")

# 环境温度 (摄氏度)
temperature = 18  # 假设环境温度为18摄氏度

# 声速 m/s，根据温度调整
speed_of_sound = 331.3 + 0.606 * temperature

# 初始化转换器和地理计算
transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)

# 创建转换器对象
transformer_geodetic_to_cartesian = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
transformer_cartesian_to_geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
# 转换经纬度到笛卡尔坐标系 (XYZ)
def convert_to_cartesian(coords):
    return transformer.transform(*coords)

# 转换监测站到笛卡尔坐标
cartesian_stations = {key: {'coords': convert_to_cartesian(value['coords']), 'times': value['times']} for key, value in stations.items()}

def geodesic_distance(coord1, coord2):
    geod = Geodesic.WGS84
    result = geod.Inverse(coord1[1], coord1[0], coord2[1], coord2[0])
    return result['s12']  # 返回两点之间的距离，单位为米

def calculate_cost_matrix(debris_positions, stations):
    num_debris = len(debris_positions)
    num_stations = len(stations)
    cost_matrix = np.zeros((num_debris, num_stations))
    
    for i in range(num_debris):
        for j, station in enumerate(stations.values()):
            predicted_time = debris_positions[i][3]  # 假设时间存储在第四个位置
            actual_time = np.mean(station['times'])  # 使用监测站的平均时间作为实际时间
            cost_matrix[i, j] = (predicted_time - actual_time) ** 2
    
    return cost_matrix

# 设定距离的最小阈值（例如：至少每个残骸之间相隔5000米）
min_distance = 8000

# 在全局作用域中定义 geod 对象
geod = Geodesic.WGS84

# 地球赤道半径（米）
a = 6378137
# 地球椭球的第一偏心率平方
e2 = 0.00669437999014
def calculate_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

def geographic_to_cartesian(phi, lambda_, h):
    """
    将地理坐标转换为笛卡尔坐标
    """
    phi = np.radians(phi)
    lambda_ = np.radians(lambda_)
    N = a / np.sqrt(1 - e2 * np.sin(phi)**2)
    X = (N + h) * np.cos(phi) * np.cos(lambda_)
    Y = (N + h) * np.cos(phi) * np.sin(lambda_)
    Z = ((1 - e2) * N + h) * np.sin(phi)
    return X, Y, Z


old_positions = np.array([
    [110.460076488883, 27.622772784491204, 4818.83604464667],
    [110.53906794835201, 27.682314298902014, 4938.010354628853],
    [110.44359407309996, 27.712741703713736, 4817.767428227544],
    [110.36529645602708, 27.654690879319908, 4707.9417365712025]
])

# 计算经度、纬度和高度的最小和最大值
min_longitude = np.min(old_positions[:, 0])
max_longitude = np.max(old_positions[:, 0])
min_latitude = np.min(old_positions[:, 1])
max_latitude = np.max(old_positions[:, 1])
min_altitude = np.min(old_positions[:, 2])
max_altitude = np.max(old_positions[:, 2])

# 扩展边界范围以允许一定的灵活性
longitude_range = (min_longitude - 0.1, max_longitude + 0.1)
latitude_range = (min_latitude - 0.1, max_latitude + 0.1)
altitude_range = (min_altitude - 100, max_altitude + 100)
num_debris = 4  # 假设有4个残骸
bounds = [longitude_range, latitude_range, altitude_range] * num_debris + [(0, 67.274)] * num_debris + [(0, 0.5)] * num_debris




def objective_function(variables):
    num_debris = 4
    total_error = 0
    debris_positions = variables[:num_debris*3].reshape(num_debris, 3)
    debris_times = variables[num_debris*3:num_debris*4]
    # 计算成本矩阵
    cost_matrix = calculate_cost_matrix(np.hstack((debris_positions, debris_times.reshape(-1, 1))), stations)

    # 应用匈牙利算法找到最小成本匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_error = cost_matrix[row_ind, col_ind].sum()

    # 计算误差项
    for i in range(num_debris):
        lon0, lat0, alt0 = debris_positions[i]
        t0 = debris_times[i]
        for station in cartesian_stations.values():
            times = station['times']
            coord = station['coords']
            lon1, lat1, alt1 = transformer_cartesian_to_geodetic.transform(*coord)
            for j, arrival_time in enumerate(times):
                distance = geod.Inverse(lat0, lon0, lat1, lon1)['s12'] + abs(alt0 - alt1)
                predicted_time = t0 + distance / speed_of_sound
                time_diff = predicted_time - arrival_time
                total_error += (time_diff**2) / (2 * sigma**2)
    
    # 检查所有残骸之间的地理距离并添加惩罚
    for i in range(num_debris):
        for j in range(i + 1, num_debris):
            lon1, lat1, _ = debris_positions[i]
            lon2, lat2, _ = debris_positions[j]
            distance = geod.Inverse(lat1, lon1, lat2, lon2)['s12']
            if distance < min_distance:
                total_error += min_distance - distance 



    # 计算重合度并更新重合度数据
    overlap_data = {
        'scores': [],
        'stagnant_count': 0,
        'last_score': None,
        'previous_total_score': 0  # 初始化为0或其他合适的值
    }

    for key, station in stations.items():
        station_coord = station['coords']
        station_times = station['times']
        station_predicted_times = []
        for i in range(num_debris):
            debris_coord = debris_positions[i]
            debris_time = debris_times[i]
            result = geod.Inverse(station_coord[1], station_coord[0], debris_coord[1], debris_coord[0])
            surface_distance = result['s12']
            height_difference = np.abs(station_coord[2] - debris_coord[2])
            total_distance = np.sqrt(surface_distance**2 + height_difference**2)
            time_to_travel = total_distance / speed_of_sound + debris_time
            station_predicted_times.append(time_to_travel)
        differences = [abs(a - p) for a, p in zip(station_times, station_predicted_times)]
        max_diff = max(differences)
        overlap_score = 1 - (max_diff / max(station_times))
        overlap_data['scores'].append(overlap_score)
    total_overlap_score = sum(overlap_data['scores']) / len(overlap_data['scores'])

    # 增加对低重合度的惩罚
    #    # 增加对低重合度的惩罚
    if sigma < 0:
        total_error += (sigma) * 10000000  # 当sigma大于0时，增加惩罚
    if total_overlap_score < 0.95:
        total_error += (0.95 - total_overlap_score) * 1000000000000  # 当重合度低于0.8时，增加惩罚

    # 更新重合度数据和计数器
    if overlap_data['last_score'] is not None and total_overlap_score == overlap_data['last_score']:
        overlap_data['stagnant_count'] += 1
    else:
        overlap_data['stagnant_count'] = 0  # 重置计数器
    overlap_data['last_score'] = total_overlap_score

    # 根据重合度的变化调整误差
    if overlap_data['previous_total_score'] is not None:
        score_change = total_overlap_score - overlap_data['previous_total_score']
        if score_change >= 0.01:
            total_error -= (abs(score_change)**8)  # 奖励值随增长数值增加
        elif score_change < 0:
            total_error += (abs(score_change)**4)  # 惩罚值随下降数值增加
    overlap_data['previous_total_score'] = total_overlap_score

    # 添加与旧数据位置接近性的惩罚
    for i in range(num_debris):
        target_position = old_positions[i]
        distance = geodesic_distance(debris_positions[i][:2], target_position[:2])
        if distance > 1000:  # 距离大于1公里
            total_error += abs(distance - 1000)**10  # 添加惩罚


    # 加入对sigma的惩罚
    penalty = 10 * np.log(sigma)
    total_error += penalty
    
    return total_error


# 约束条件
def time_difference_constraint(variables):
    # 时间差不超过5秒
    debris_times = variables[-4:]
    return 5 - (np.max(debris_times) - np.min(debris_times))

def time_prior_constraint(variables):
    # 音爆发生时间小于任何抵达时间
    debris_times = variables[-4:]
    min_time = min(min(station['times']) for station in cartesian_stations.values())
    return min_time - max(debris_times)

def altitude_constraint(variables):
    # 高程非负
    num_debris = 4
    debris_positions = variables[:num_debris*3].reshape(num_debris, 3)
    return np.min(debris_positions[:, 2])
# 参数边界，考虑XYZ坐标和时间

# 约束条件
constraints = [
    NonlinearConstraint(time_difference_constraint, 0, np.inf),
    NonlinearConstraint(time_prior_constraint, 0, np.inf),
    NonlinearConstraint(altitude_constraint, 0, np.inf)
]

# 生成符合边界的随机初始猜测
initial_guess = old_positions

# 定义回调函数用于自适应调整参数
def adaptive_callback(xk, convergence):
    estimated_positions = xk[:num_debris*3].reshape(num_debris, 3)
    estimated_times = xk[num_debris*3:num_debris*4]
    sigma = xk[-1]  # 获取sigma的值
    overlap_scores = {}
    for key, station in stations.items():
        station_coord = station['coords']
        station_times = station['times']
        station_predicted_times = []
        for i in range(num_debris):
            debris_coord = estimated_positions[i]
            debris_time = estimated_times[i]
            result = geod.Inverse(station_coord[1], station_coord[0], debris_coord[1], debris_coord[0])
            surface_distance = result['s12']
            height_difference = np.abs(station_coord[2] - debris_coord[2])
            total_distance = np.sqrt(surface_distance**2 + height_difference**2)
            time_to_travel = total_distance / speed_of_sound + debris_time
            station_predicted_times.append(time_to_travel)
        differences = [(abs(a - p) / sigma) for a, p in zip(station_times, station_predicted_times)]
        max_diff = max(differences)
        overlap_score = 1 - (max_diff / max(station_times))
        overlap_scores[key] = overlap_score
    total_overlap_score = sum(overlap_scores.values()) / len(overlap_scores)
    print(f"当前整体重合度: {total_overlap_score:.2f}, 当前 sigma 值为: {sigma:.2f}")
 
    if total_overlap_score >= 0.99 and sigma == 0:
        return True  # 返回True以停止优化
    return False




# 存储差分进化算法的参数
differential_evolution_args = {
    'strategy': 'best1bin',
    'popsize': 25,
    'tol': 0.001,
    'mutation': (0.5, 1.5),
    'recombination': 0.8,
    'seed': None,
    'callback': adaptive_callback,
    'disp': True,
    'polish': True,
    'init': 'random',
    'atol': 0,
    'workers': -1  # 使用所有可用核心
}

# 执行差分进化算法
result = differential_evolution(
    objective_function,
    bounds=bounds,
    maxiter=1000,
    constraints=constraints,
    **differential_evolution_args

)

# 存储优化结果
estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
estimated_times = result.x[num_debris*3:num_debris*4]



# 提取结果和输出重合度
# 优化成功后的代码块
if result.success:
    estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
    estimated_times = result.x[num_debris*3:num_debris*4]
    print("优化成功，找到可能的音爆源位置和时间")
    for i in range(num_debris):
        print(f"残骸 {i+1}: 位置 ({estimated_positions[i][0]}, {estimated_positions[i][1]}, {estimated_positions[i][2]}), 时间 {estimated_times[i]}")

    # 将地理坐标转换为笛卡尔坐标
    old_cartesian_positions = [geographic_to_cartesian(*pos) for pos in old_positions]
    new_cartesian_positions = [geographic_to_cartesian(*pos) for pos in estimated_positions]

    # 计算新数据残骸与旧数据残骸之间的距离
    distances_km = []
    for i in range(len(new_cartesian_positions)):
        distance = calculate_distance(new_cartesian_positions[i], old_cartesian_positions[i])
        distances_km.append(distance / 1000)  # 将距离转换为千米

    # 输出计算结果
    average_distance = np.mean(distances_km)
    for i, distance in enumerate(distances_km):
        print(f"新数据残骸 {i+1} 与旧数据残骸 {i+1} 之间的距离是 {distance:.3f} km")
    print(f"平均距离是 {average_distance:.3f} km")

    # 计算每个位置的误差
    errors = np.array(estimated_positions) - np.array(old_positions)

    # 计算平均绝对误差 (MAE)
    mae = np.mean(np.abs(errors))

    # 计算均方根误差 (RMSE)
    rmse = np.sqrt(np.mean(errors**2))

    print(f"平均绝对误差 (MAE): {mae:.3f}")
    print(f"均方根误差 (RMSE): {rmse:.3f}")
else:
    print("优化失败：", result.message)

# 这段代码将设置所有必要的边界和约束，并运行差分进化算法以找到最优解，这里我们假设每个残骸的音爆发生位置和时间可以通过全局优化方法估计。

# 计算并输出最终的整体重合度
predicted_times = {}
overlap_scores = {}
for key, station in stations.items():
    station_coord = station['coords']
    station_times = station['times']
    station_predicted_times = []
    for i in range(num_debris):
        debris_coord = estimated_positions[i]
        debris_time = estimated_times[i]
        result = geod.Inverse(station_coord[1], station_coord[0], debris_coord[1], debris_coord[0])
        surface_distance = result['s12']
        height_difference = np.abs(station_coord[2] - debris_coord[2])
        total_distance = np.sqrt(surface_distance**2 + height_difference**2)
        time_to_travel = total_distance / speed_of_sound + debris_time
        station_predicted_times.append(time_to_travel)
    
    predicted_times[key] = station_predicted_times
    
    # 计算重合度
    differences = [abs(a - p) for a, p in zip(station_times, station_predicted_times)]
    max_diff = max(differences)
    overlap_score = 1 - (max_diff / max(station_times))
    overlap_scores[key] = overlap_score

# 输出重合度
for key, score in overlap_scores.items():
    print(f"监测站 {key} 的输出重合度: {score:.2f}")

# 计算整体重合度
total_overlap_score = sum(overlap_scores.values()) / len(overlap_scores)
print(f"整体重合度: {total_overlap_score:.2f}")



