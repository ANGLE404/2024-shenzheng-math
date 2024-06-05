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
stations = {
    'A': {'coords': (110.241, 27.204, 824), 'times': [100.767, 164.229, 214.850, 270.065]},
    'B': {'coords': (110.783, 27.456, 727), 'times': [92.453, 112.220, 169.362, 196.583]},
    'C': {'coords': (110.762, 27.785, 742), 'times': [75.560, 110.696, 156.936, 188.020]},
    'D': {'coords': (110.251, 28.025, 850), 'times': [94.653, 141.409, 196.517, 258.985]},
    'E': {'coords': (110.524, 27.617, 786), 'times': [78.600, 86.216, 118.443, 126.669]},
    'F': {'coords': (110.467, 28.081, 678), 'times': [67.274, 166.270, 175.482, 266.871]},
    'G': {'coords': (110.047, 27.521, 575), 'times': [103.738, 163.024, 206.789, 210.306]}
}

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

    # 检查所有残骸之间的地理距离并添加惩罚
    for i in range(num_debris):
        for j in range(i + 1, num_debris):
            lon1, lat1, _ = debris_positions[i]
            lon2, lat2, _ = debris_positions[j]
            distance = geod.Inverse(lat1, lon1, lat2, lon2)['s12']
            if distance < min_distance:
                total_error += min_distance - distance 

    # 正常的误差计算
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
                total_error += time_diff**2 

    # 计算整体重合度作为优化指标
    overlap_data = {
        'scores': [],
        'stagnant_count': 0,
        'last_score': None,
        'previous_total_score': None
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
    if total_overlap_score < 0.95:
        total_error += (0.95 - total_overlap_score) * 900000000  # 当重合度低于0.8时，增加惩罚

    # 更新重合度数据和计数器
    if overlap_data['last_score'] is not None and total_overlap_score == overlap_data['last_score']:
        overlap_data['stagnant_count'] += 1
    else:
        overlap_data['stagnant_count'] = 0  # 重置计数器
    overlap_data['last_score'] = total_overlap_score

    # 当重合度5次不变时，增加极大的惩罚
    if overlap_data['stagnant_count'] >= 5:
        total_error += (0.95 - total_overlap_score) * 100000  # 增加极大的惩罚值

    # 根据重合度的提升给予奖励    if overlap_data['previous_total_score'] is not None:
        score_change = total_overlap_score - overlap_data['previous_total_score']
        if score_change >= 0.01:
            total_error -= score_change * (abs(score_change)**4)  # 奖励值随增长数值增加
        elif score_change < 0:
            total_error += abs(score_change) * (abs(score_change)**2)  # 惩罚值随下降数值增加
    overlap_data['previous_total_score'] = total_overlap_score

    
    
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

num_debris = 4
# 根据stations中的最大和最小时间来设置时间范围
min_time = min(min(station['times']) for station in stations.values())
max_time = max(max(station['times']) for station in stations.values())
bounds = [(109, 112), (26, 29), (4693.82, 88847.59)] * num_debris + [(0, min_time)] * num_debris  # XYZ坐标范围和时间范围

# 约束条件
constraints = [
    NonlinearConstraint(time_difference_constraint, 0, np.inf),
    NonlinearConstraint(time_prior_constraint, 0, np.inf),
    NonlinearConstraint(altitude_constraint, 0, np.inf)
]

# 生成符合边界的随机初始猜测
initial_guess = np.array([np.random.uniform(low, high) for low, high in bounds])

# 定义回调函数用于自适应调整参数
def adaptive_callback(xk, convergence):
    estimated_positions = xk[:num_debris*3].reshape(num_debris, 3)
    estimated_times = xk[num_debris*3:num_debris*4]
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
        differences = [abs(a - p) for a, p in zip(station_times, station_predicted_times)]
        max_diff = max(differences)
        overlap_score = 1 - (max_diff / max(station_times))
        overlap_scores[key] = overlap_score
    total_overlap_score = sum(overlap_scores.values()) / len(overlap_scores)
    print(f"当前整体重合度: {total_overlap_score:.2f}")

    # 动态调整参数
    stagnation_counter=0
    if total_overlap_score < 0.2:
        differential_evolution_args['mutation'] = (0.9, 2.1)
        differential_evolution_args['popsize'] = 20  # 增加种群大小以增加搜索范围
        differential_evolution_args['tol'] = 0.01  # 减小容忍度以提高精度
    elif total_overlap_score < 0.3:
        differential_evolution_args['mutation'] = (1.1, 2.3)
        differential_evolution_args['popsize'] = 30
        differential_evolution_args['tol'] = 0.005
    elif total_overlap_score < 0.4:
        differential_evolution_args['mutation'] = (1.3, 2.5)
        differential_evolution_args['popsize'] = 40
        differential_evolution_args['tol'] = 0.001
    elif total_overlap_score < 0.5:
        differential_evolution_args['mutation'] = (1.5, 2.7)
        differential_evolution_args['popsize'] = 50
        differential_evolution_args['tol'] = 0.0005
    elif total_overlap_score < 0.6:
        differential_evolution_args['mutation'] = (1.7, 2.9)
        differential_evolution_args['popsize'] = 60
        differential_evolution_args['tol'] = 0.0001
    elif total_overlap_score < 0.7:
        differential_evolution_args['mutation'] = (1.9, 3.1)
        differential_evolution_args['popsize'] = 70
        differential_evolution_args['tol'] = 0.00005
    elif total_overlap_score < 0.8:
        differential_evolution_args['mutation'] = (2.1, 3.3)
        differential_evolution_args['popsize'] = 80
        differential_evolution_args['tol'] = 0.00001
    elif total_overlap_score < 0.9:
        differential_evolution_args['mutation'] = (2.3, 3.5)
        differential_evolution_args['popsize'] = 90
        differential_evolution_args['tol'] = 0.000005
    else:
        differential_evolution_args['mutation'] = (2.5, 3.7)
        differential_evolution_args['popsize'] = 100  # 减少种群大小以减少计算量
        differential_evolution_args['tol'] = 0.000001  # 增加容忍度以快速收敛
        # 如果重合度在5轮内一直不变化，增加变异率和种群大小


    differential_evolution_args['recombination'] = 0.90 + 0.05 * (1 - total_overlap_score)
    # 如果达到目标重合度，停止优化
    if total_overlap_score >= 0.99:
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
    constraints=constraints,
    **differential_evolution_args
)

# 存储优化结果
estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
estimated_times = result.x[num_debris*3:num_debris*4]



# 提取结果和输出重合度
if result.success:
    estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
    estimated_times = result.x[num_debris*3:num_debris*4]
    print("优化成功，找到可能的音爆源位置和时间")
    for i in range(num_debris):
        print(f"残骸 {i+1}: 位置 ({estimated_positions[i][0]}, {estimated_positions[i][1]}, {estimated_positions[i][2]}), 时间 {estimated_times[i]}")
else:
    print("优化失败：", result.message)

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