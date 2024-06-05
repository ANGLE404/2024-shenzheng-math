import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # 引入tqdm
# 设置matplotlib的字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pyproj import Transformer
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter
from deap import creator, base, tools, algorithms


def expand_cost_matrix_for_multiple_signals(arrival_times, num_signals_per_device):
    # 每个设备行重复 num_signals_per_device 次
    expanded_matrix = np.tile(arrival_times, (num_signals_per_device, 1))
    return expanded_matrix

def match_signals_to_debris(arrival_times, num_signals_per_device=4):
    # 扩展成本矩阵以允许多对多匹配
    expanded_matrix = expand_cost_matrix_for_multiple_signals(arrival_times, num_signals_per_device)
    
    # 应用匈牙利算法进行匹配
    row_ind, col_ind = linear_sum_assignment(expanded_matrix)
    
    # 将扩展后的行索引转换回原始设备索引
    original_device_indices = row_ind % len(arrival_times)
    return original_device_indices, col_ind

# 设定经纬度和高度的范围
longitude_range = (110, 111)
latitude_range = (27, 28)
altitude_range = (200, 1000)

def generate_random_position():
    lon = np.random.uniform(*longitude_range)
    lat = np.random.uniform(*latitude_range)
    alt = np.random.uniform(*altitude_range)
    #print(f"Generated position: lon={lon}, lat={lat}, alt={alt}")
    return lon, lat, alt

# Utility functions
def geodetic_to_cartesian(lon, lat, alt):
    a = 6378137.0
    f = 1 / 298.257223563
    e_sq = f * (2 - f)
    N = a / np.sqrt(1 - e_sq * np.sin(np.radians(lat))**2)
    X = (N + alt) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    Y = (N + alt) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    Z = ((1 - e_sq) * N + alt) * np.sin(np.radians(lat))
    return (X, Y, Z)

def plot_devices_and_debris(devices, debris_positions, iteration):
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
    ax.set_title(f'第 {iteration+1} 次迭代的设备与残骸位置 3D 视图')
    # 设置坐标轴范围
    ax.set_xlim([110, 111])
    ax.set_ylim([27, 28])
    # 调整图例位置
    ax.legend(loc='upper left', fontsize='small')
    
    # 显示图表
    #plt.show()
    # 保存图表到本地文件
    #fig.savefig(f'设备与残骸位置图_{iteration+1}.png')
    # 关闭图形以释放内存
    plt.close(fig)
def calculate_arrival_times(debris_positions, debris_times, monitors):
    speed_of_sound = 340
    arrival_times = np.zeros((len(debris_positions), len(monitors)))
    monitor_cartesian = np.array([geodetic_to_cartesian(lon, lat, alt) for lon, lat, alt in monitors])
    for i, (debris_pos, debris_time) in enumerate(zip(debris_positions, debris_times)):
        debris_cartesian = geodetic_to_cartesian(*debris_pos)
        distances = np.sqrt(np.sum((monitor_cartesian - debris_cartesian) ** 2, axis=1))
        arrival_times[i, :] = distances / speed_of_sound + debris_time
    return arrival_times

def objective_function(variables, monitor_times, monitors, num_debris, lambda_penalty=0.1):  # 增加lambda_penalty
    variables = np.array(variables)
    predicted_positions = variables[:num_debris * 3].reshape(num_debris, 3)
    predicted_times = variables[num_debris * 3:]
    monitor_cartesian = np.array([geodetic_to_cartesian(lon, lat, alt) for lon, lat, alt in monitors])
    predicted_arrival_times = calculate_arrival_times(predicted_positions, predicted_times, monitor_cartesian)
    original_objective = np.sum((predicted_arrival_times - monitor_times) ** 2)
    reward = lambda_penalty * len(monitors)
    new_objective = original_objective - reward
    return new_objective



def optimize_positions(monitors, initial_guess, monitor_times, num_debris):
    # 定义适应度函数
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # 创建遗传算法所需的工具箱
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1000, 1000)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_debris*4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册遗传算法的操作
    toolbox.register("evaluate", lambda ind: (objective_function(ind, monitor_times, monitors, num_debris),))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 初始化种群
    population = toolbox.population(n=300)

    # 应用遗传算法
    result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)

    # 找到最优解
    best_ind = tools.selBest(population, 1)[0]
    return best_ind.fitness.values[0], best_ind



def cartesian_to_geodetic(x, y, z):
    # 创建从地心地固坐标系(ECEF)转换到地理坐标系(WGS84)的转换器
    transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

    # 使用转换器进行坐标转换
    lon, lat, alt = transformer.transform(x, y, z)
    return lon, lat, alt



def process_single_configuration(monitors, num_debris, debris_positions, debris_times):
    initial_positions = [generate_random_position() for _ in range(num_debris)]
    initial_times = np.random.uniform(100, 200, num_debris)
    initial_guess = np.concatenate((np.ravel(initial_positions), initial_times))

    monitor_times = calculate_arrival_times(debris_positions, debris_times, monitors)
    optimization_result = optimize_positions(monitors, initial_guess, monitor_times, num_debris)

    # 将列表转换为numpy数组后进行reshape
    best_device_positions = np.array(optimization_result[1][:num_debris*3]).reshape(num_debris, 3)
    best_device_positions = [cartesian_to_geodetic(*pos) for pos in best_device_positions]  # 转换回地理坐标
    best_device_count = len(monitors)

    return (optimization_result[0], best_device_count, best_device_positions)




def execute_parallel(num_iterations, monitors_data, num_debris, debris_positions, debris_times):
    with ProcessPoolExecutor() as executor:
        futures = []
        # 第0次迭代选择所有设备
        future = executor.submit(process_single_configuration, monitors_data, num_debris, debris_positions, debris_times)
        futures.append(future)
        
        # 从第1次迭代开始随机选择设备数量
        for i in range(1, num_iterations):
            # 随机选择设备数量，至少选择4个，最多选择全部设备
            num_selected_monitors = np.random.randint(5, len(monitors_data) + 1)
            selected_monitors = monitors_data[np.random.choice(len(monitors_data), num_selected_monitors, replace=False)]
            future = executor.submit(process_single_configuration, selected_monitors, num_debris, debris_positions, debris_times)
            futures.append(future)
        
        # 等待所有任务完成
        results = [future.result() for future in futures]
    return results



def main():
    devices = {
        'A': generate_random_position(),
        'B': generate_random_position(),
        'C': generate_random_position(),
        'D': generate_random_position(),
        'E': generate_random_position(),
        'F': generate_random_position(),
        'G': generate_random_position()
    }
    # 输出生成的随机坐标
    for device, position in devices.items():
        print(f"设备 {device} 的随机坐标: {position}")
    # 将设备的值转换为numpy数组，便于后续处理
    monitors_data = np.array(list(devices.values()))
    # 设置残骸的数量
    num_debris = 4
    # 设置迭代次数，即执行并行处理的次数
    num_iterations = 50
    # 初始化用于存储所有结果的列表
    all_results = []
    best_configurations = []

    # 创建一个进度条
    with tqdm(total=100, desc="总迭代进度") as progress_bar:
        for iteration in range(100):
            # 生成新的残骸位置和时间
            true_debris_positions = np.array([generate_random_position() for _ in range(num_debris)])
            true_debris_times = np.random.uniform(100, 200, num_debris)

            # 计算到达时间
            arrival_times = calculate_arrival_times(true_debris_positions, true_debris_times, monitors_data)
            # 匹配信号到残骸
            device_indices, debris_indices = match_signals_to_debris(arrival_times)

            # 执行并行优化
            results = execute_parallel(1, monitors_data, num_debris, true_debris_positions, true_debris_times)
            all_results.extend(results)

            # 选择最佳配置
            best_result = min(results, key=lambda x: x[0])
            best_configurations.append(best_result[1])

            # 更新进度条
            progress_bar.update(1)

    # 统计每个设备数量出现的次数
    device_counts = Counter(best_configurations)
    total_counts = sum(device_counts.values())
    # 输出统计结果及其占比
    for device_count, frequency in device_counts.items():
        percentage = (frequency / total_counts) * 100
        print(f"监测设备数量 {device_count} 出现了 {frequency} 次，占比 {percentage:.2f}%")
    # 统计完成
    print("设备数量统计及占比输出完成。")


if __name__ == "__main__":
    main()


