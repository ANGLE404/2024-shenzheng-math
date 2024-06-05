***\*深圳杯数学建模挑战赛2024A题\****

***\*多个火箭残骸的准确定位\****

绝大多数火箭为多级火箭，下面级火箭或助推器完成既定任务后，通过级间分离装置分离后坠落。在坠落至地面过程中，残骸会产生跨音速音爆。为了快速回收火箭残骸，在残骸理论落区内布置多台震动波监测设备，以接收不同火箭残骸从空中传来的跨音速音爆，然后根据音爆抵达的时间，定位空中残骸发生音爆时的位置，再采用弹道外推实现残骸落地点的快速精准定位。

***\*附\**** 震动波的传播速度为340 m/s，计算两点间距离时可忽略地面曲率，纬度间每度距离值近似为111.263 km，经度间每度距离值近似为97.304 km。





观测系统时钟0时指的是监测系统中定义的一个参考时间点，通常是开始观测或者某个具体事件的时间。在本题中，音爆抵达时间（相对于观测系统时钟0时）即是指音爆声波到达每个监测设备的时间与这个参考时间点的时间差。

例如，如果观测系统时钟0时是某天的午夜12点，那么一个记录为100.767秒的音爆抵达时间表示，音爆在该天的12点01分40.767秒时抵达了相应的监测设备。

在进行音爆位置的计算时，我们使用这些相对时间来推算音爆发生的确切时间和位置，通过计算音爆从发生点到各个监测设备所需的传播时间。这种方法允许我们从多个时间记录中反向推算出音爆的原始发生时间和位置，即使我们不知道观测系统时钟0时代表的具体绝对时间。

是的，音爆发生时间可以是负数。这个时间值表示音爆发生的相对时间，它相对于观测系统时钟0时的一个时间点。如果这个时间是负数，这意味着音爆实际上是在观测系统定义的0时之前就已经发生了。

音爆发生的时间不必是正数，但关键是它必须满足所有监测设备接收到音爆的时间顺序和逻辑关系。具体来说，音爆发生时间 \( t_0 \) 必须小于或等于所有监测设备记录到的音爆抵达时间。这样，从 \( t_0 \) 到每个设备的抵达时间 \( t \) 才是正数，使得音爆从发生点到达每个设备的计算距离和时间是有意义的。

换句话说，音爆发生时间 \( t_0 \) 是计算所得的所有设备接收到音爆声波的时间差 \( t - \frac{d}{c} \)（其中 \( d \) 是从音爆发生点到设备的距离，\( c \) 是声速）的最大值的负数。这确保了音爆在传播到任何设备之前已经发生。在我们的模型中，我们通过最小化这些时间差的方差来寻找最佳的 \( t_0 \)。

如果音爆发生时间 \( t_0 \) 大于任何一个监测设备记录的音爆抵达时间 \( t \)，这在物理上是不可能的，因为这意味着音爆声波会在产生之前到达监测设备。这违反了因果律——事件的效果（音爆抵达设备）不能在其原因（音爆发生）之前发生。

在建立数学模型和求解问题时，我们必须保证音爆发生的时间 \( t_0 \) 小于或等于所有设备记录到的最早音爆抵达时间。这样，每个设备测量到的音爆抵达时间 \( t \) 与 \( t_0 \) 之间的差，即 \( t - t_0 \)，才是正数，表示音爆从发生点传播到该设备所需的实际时间。

如果在计算或数据输入过程中出现音爆发生时间大于任何一个设备的抵达时间的情况，这通常是数据错误、计算误差或模型设定不当的表现。这种情况下，需要重新检查数据和模型的设置，确保它们符合物理规律和实际观测情况。

# ***\*问题1\**** 建立数学模型，分析如果要精准确定空中单个残骸发生音爆时的位置坐标（经度、纬度、高程）和时间，至少需要布置几台监测设备？

| 设备 | 经度(°) | 纬度(°) | 高程(m) | 音爆抵达时间(s) |
| ---- | ------- | ------- | ------- | --------------- |
| A    | 110.241 | 27.204  | 824     | 100.767         |
| B    | 110.780 | 27.456  | 727     | 112.220         |
| C    | 110.712 | 27.785  | 742     | 188.020         |
| D    | 110.251 | 27.825  | 850     | 258.985         |
| E    | 110.524 | 27.617  | 786     | 118.443         |
| F    | 110.467 | 27.921  | 678     | 266.871         |
| G    | 110.047 | 27.121  | 575     | 163.024         |

要准确确定空中单个残骸发生音爆时的位置坐标（经度、纬度、高程）和时间，我们需要考虑的关键因素包括测量数据的准确性、测量站点的地理分布以及足够的测量数据来构建可靠的数学模型。

可以将每个监测设备接收到的音爆时间与残骸爆炸时间之间的关系表达为一个方程。这个方程基于音爆的传播速度和设备与爆炸源之间的距离。对于每个设备 \(i\)，我们有：
$$
t_i = t_0 + \frac{d_i}{s}
$$
其中：

- \(t_i\) 是设备 \(i\) 接收到音爆的时间。
- \(t_0\) 是残骸发生音爆的时间。
- \(d_i\) 是设备 \(i\) 与音爆源之间的距离。
- \(s\) 是音爆的传播速度，即340 m/s。

设备的地理位置可以转换为笛卡尔坐标系中的点，假设残骸发生音爆的位置为 \( (x, y, z) \)：

- 经度转换为x坐标: \( x = \text{lon} \times 97.304 \times 1000 \)
- 纬度转换为y坐标: \( y = \text{lat} \times 111.263 \times 1000 \)
- 高程为z坐标: \( z = \text{height} \)

因此，每个设备的距离方程可以表示为：
 d_i = \sqrt{(x - x_i)^2 + (y - y_i)^2 + (z - z_i)^2} 
其中 \( (x_i, y_i, z_i) \) 是第i个设备的坐标。

为了能够解决 \( (x, y, z, t_0) \) 这四个未知数，理论上我们至少需要四个独立的方程。这意味着至少需要四台监测设备，且这些设备的位置不能共面，也就是说，它们的位置应该在空间中尽可能分散，以避免退化情况，即所有的设备几乎位于同一平面上。

选取那几个装置更加合适？

```python
import random
import math
from collections import Counter
from tqdm import tqdm  # 引入tqdm库
# 残骸的位置和时间数据
device_coordinates = {
    'A': (110.241, 27.204, 824),
    'B': (110.783, 27.456, 727),
    'C': (110.712, 27.785, 742),
    'D': (110.251, 27.825, 850),
    'E': (110.524, 27.617, 786),
    'F': (110.467, 27.921, 678),
    'G': (110.047, 27.121, 575)
}

device_times = {
    'A': 100.767,
    'B': 112.220,
    'C': 188.020,
    'D': 258.985,
    'E': 118.443,
    'F': 266.871,
    'G': 163.024
}

# 布置的监测设备数量
num_devices = 4

# 蒙特卡洛模拟次数
num_iterations = 1000

# 初始化最佳设备组合统计
best_combination_counts = Counter()

# 运行模拟100次
for run in tqdm(range(100), desc="总模拟进度"):
    min_error = float('inf')
    best_device_combination = None

    # 执行蒙特卡洛模拟
    for _ in tqdm(range(num_iterations), desc=f"模拟进度 {run+1}/100", leave=False):
        # 随机选择设备组合
        selected_devices = random.sample(list(device_coordinates.keys()), num_devices)

        # 计算残骸发生音爆时的位置和时间估计值
        estimated_coordinates = [device_coordinates[device] for device in selected_devices]
        estimated_time = max([device_times[device] for device in selected_devices])

        # 计算估计值与实际值之间的误差
        error = math.sqrt(sum([(estimated_coordinates[i][j] - device_coordinates[selected_devices[i]][j])**2 for i in range(num_devices) for j in range(3)]) + (estimated_time - max([device_times[device] for device in selected_devices]))**2)

        # 更新最小误差和最佳设备组合
        if error < min_error:
            min_error = error
            best_device_combination = tuple(sorted(selected_devices))

    # 更新统计数据
    best_combination_counts[best_device_combination] += 1

# 输出统计结果
print("最佳设备组合出现次数:")
# 对统计结果进行排序，按计数从大到小
sorted_combinations = sorted(best_combination_counts.items(), key=lambda item: item[1], reverse=True)
for combination, count in sorted_combinations:
    print(f"{combination}: {count}次")
    
    
    
    
总模拟进度: 100%|██████████████████████████████████████████████| 100/100 [00:02<00:00, 47.90it/s]
最佳设备组合出现次数:                                                                             
('A', 'B', 'C', 'E'): 7次
('A', 'C', 'D', 'E'): 6次
('A', 'B', 'D', 'F'): 6次
('B', 'C', 'D', 'E'): 5次
('A', 'B', 'C', 'F'): 5次
('A', 'B', 'C', 'G'): 5次
('B', 'C', 'D', 'G'): 4次
('A', 'B', 'E', 'F'): 4次
('B', 'D', 'E', 'G'): 4次
('A', 'C', 'D', 'F'): 3次
('B', 'E', 'F', 'G'): 3次
('A', 'B', 'D', 'E'): 3次
('A', 'C', 'D', 'G'): 3次
('A', 'C', 'E', 'F'): 3次
('A', 'D', 'F', 'G'): 3次
('A', 'E', 'F', 'G'): 3次
('A', 'C', 'F', 'G'): 3次
('C', 'D', 'E', 'G'): 3次
('A', 'D', 'E', 'G'): 3次
('C', 'E', 'F', 'G'): 3次
('C', 'D', 'F', 'G'): 3次
('B', 'C', 'E', 'F'): 2次
('B', 'D', 'E', 'F'): 2次
('A', 'C', 'E', 'G'): 2次
('A', 'B', 'F', 'G'): 2次
('D', 'E', 'F', 'G'): 2次
('C', 'D', 'E', 'F'): 2次
('A', 'B', 'C', 'D'): 1次
('B', 'C', 'F', 'G'): 1次
('A', 'B', 'E', 'G'): 1次
('B', 'C', 'E', 'G'): 1次
('B', 'C', 'D', 'F'): 1次
('A', 'D', 'E', 'F'): 1次
```

### 分析为什么选择 'A', 'B', 'C', 'E'

在您的蒙特卡洛模拟中，组合 'A', 'B', 'C', 'E' 出现次数最多可能有以下原因：

1. **地理位置分布**：设备 'A', 'B', 'C', 'E' 可能在地理上分布得较为均匀，能够较好地覆盖音爆发生区域，从而提供更准确的定位信息。

2. **时间记录的准确性**：这些设备记录的时间可能相对更准确，误差较小，因此在模拟中更频繁地被选为最佳组合。

3. **高程和坐标的适中性**：这些设备的高程和坐标可能位于一个适中的范围内，有助于减少计算误差，提高定位的准确性。

### 确定空中单个残骸发生音爆的位置和时间所需的最少监测设备数量

为了精准确定空中单个残骸发生音爆时的位置坐标（经度、纬度、高程）和时间，理论上至少需要四台监测设备。这是因为：

- **三维空间定位**：确定一个点在三维空间中的位置至少需要三个独立的测量值（在这里是距离），每个设备提供一个距离测量。
- **时间同步和误差**：第四个设备提供额外的数据可以帮助解决时间同步的问题和减少误差，特别是在存在测量噪声和数据不完全同步的情况下。

在实际应用中，可能需要更多的设备来提高系统的冗余性和鲁棒性，尤其是在环境条件复杂或数据质量可能受到影响的情况下。





### 蒙特卡洛方法简介

蒙特卡洛方法是一种基于随机抽样来解决计算问题的统计模拟方法。在多种科学和工程问题中，尤其是在物理和金融领域中，蒙特卡洛方法被广泛应用于那些难以用传统分析方法解决的问题。其核心思想是通过重复随机抽样来计算或模拟概率问题的结果。

### 代码分析

在您的代码中，蒙特卡洛方法被用于估计音爆的位置和时间。通过随机选择不同的设备组合，并计算每种组合下的位置和时间估计误差，代码试图找到误差最小的设备组合。这种方法的优点是它不依赖于问题的解析解，而是通过大量的随机抽样来逼近问题的解。

### 为什么选择 'A', 'B', 'C', 'E'

从输出结果来看，组合 ('A', 'B', 'C', 'E') 出现次数最多，这表明在多次模拟中，这个组合在估计音爆位置和时间时，产生的误差最小。这可能是因为这些设备的地理位置和记录的时间数据相对于其他设备能够更好地代表音爆发生的区域，从而使得通过这些设备计算出的结果更为准确。

### 确定音爆位置和时间所需的最少设备数量

要精确确定空中单个残骸发生音爆的位置坐标（经度、纬度、高程）和时间，理论上至少需要四个监测设备。这是因为要解决一个包含四个未知数（三个空间坐标和一个时间坐标）的问题。每个设备提供一个时间记录，可以用来计算从音爆源到设备的距离。四个独立的距离测量可以帮助解决四个未知数的方程组，从而确定音爆的确切位置和时间。

### 实际应用中的考虑

尽管理论上四个设备足够，但在实际应用中，可能需要更多的设备来提高解的准确性和可靠性。这是因为实际测量中可能存在噪声和误差，多个数据点可以帮助通过统计方法（如最小二乘法）减少这些误差的影响。此外，设备的分布也会影响测量的准确性，理想情况下，这些设备应该均匀地分布在音爆可能发生的区域周围，以最大化覆盖范围并减少盲区。

总结来说，您的代码通过蒙特卡洛方法有效地模拟了不同设备组合对音爆位置和时间估计的影响，并找到了误差最小的设备组合。对于实际操作，建议使用四个以上的设备，并考虑设备的合理布局，以确保估计结果的准确性和可靠性。



在这个项目中，我们使用蒙特卡洛模拟方法来估计音爆的位置和时间。蒙特卡洛方法是一种统计模拟技术，通过重复随机抽样来估计可能的结果。这种方法特别适用于解决复杂的科学和工程问题，其中传统的解析方法可能难以应用或不可行。

### 实验设置

我们设定了一组监测设备，每个设备都记录了音爆发生时的时间，并且我们知道每个设备的具体位置。目标是确定音爆的确切位置和发生时间。

### 模拟过程

在模拟过程中，我们随机选择几个设备组合，并利用这些设备的数据来估计音爆的位置和时间。对于每一种设备组合，我们计算基于这些设备数据得出的音爆位置和时间的估计误差。通过重复这个过程多次，我们可以找到平均误差最小的设备组合，这意味着这个组合提供了对音爆位置和时间最准确的估计。

### 结果分析

在多次模拟后，我们发现某些设备组合比其他组合更频繁地产生较小的误差。例如，设备组合 ('A', 'B', 'C', 'E') 在模拟中显示出最佳性能，这表明这些设备的地理位置和时间记录在估计音爆位置和时间方面相对更为准确和可靠。

### 结论

通过这种方法，我们不仅能够有效地估计音爆的位置和时间，还能识别出最佳的设备组合，为实际操作提供指导。这种基于统计的方法提供了一种强大的工具，用于处理复杂的实际问题，特别是在直接解决方案不明显或难以获得的情况下。

# 假设某火箭一级残骸分离后，在落点附近布置了7台监测设备，各台设备三维坐标（经度、纬度、高程）、音爆抵达时间（相对于观测系统时钟0时）如下表所示：

| 设备 | 经度(°) | 纬度(°) | 高程(m) | 音爆抵达时间(s) |
| ---- | ------- | ------- | ------- | --------------- |
| A    | 110.241 | 27.204  | 824     | 100.767         |
| B    | 110.780 | 27.456  | 727     | 112.220         |
| C    | 110.712 | 27.785  | 742     | 188.020         |
| D    | 110.251 | 27.825  | 850     | 258.985         |
| E    | 110.524 | 27.617  | 786     | 118.443         |
| F    | 110.467 | 27.921  | 678     | 266.871         |
| G    | 110.047 | 27.121  | 575     | 163.024         |

在上述代码中，我们采用了几个数学和优化方法来解决多个监测站点接收音爆时间数据的问题，进而确定音爆的空中位置和时间。以下是详细的数学模型和步骤解析：



在您的问题中，您正在处理声波在空气中的传播，特别是与音爆相关的事件。音爆通常涉及较大的距离和空气作为传播介质，因此声波衰减主要受到以下两个因素的影响：

1. **几何衰减**：由于声波在空间中的扩散，声强会随着距离的增加而减少。对于点源，声强与距离的平方成反比（\( I \propto \frac{1}{r^2} \)）。这是因为声波的能量在三维空间中均匀分布在以声源为中心的球面上，球面的面积与半径的平方成正比。

2. **吸收衰减**：声波在空气中传播时，由于空气的粘性、热传导和分子松弛过程，部分声能会转化为热能，导致声波能量的衰减。这种衰减与频率有关，高频声波比低频声波衰减得更快。衰减可以用指数形式表示（\( I = I_0 e^{-\alpha r} \)），其中 \( \alpha \) 是与介质和频率相关的衰减系数。

### 建议使用的模型：

对于您的应用场景（音爆的位置和时间的确定），建议使用**复合衰减模型**，结合几何衰减和吸收衰减。这种模型可以更准确地描述声波在实际环境中的传播和衰减情况。模型可以表示为：

\[ I = \frac{I_0}{r^2} e^{-\alpha r} \]

其中：

- \( I_0 \) 是初始声强。
- \( r \) 是从声源到接收点的距离。
- \( \alpha \) 是衰减系数，需要根据声波的频率和空气的特性来确定。

### 实现步骤：

1. **确定衰减系数 \( \alpha \)**：这可能需要查阅文献或进行实验来获取，具体取决于声波的频率和环境条件。
2. **计算距离 \( r \)**：根据声源和各监测站的位置计算距离。
3. **应用衰减模型**：使用上述公式计算不同距离下的声强衰减，进而估计声源的位置和发生时间。

如果需要进一步的帮助来实现这个模型，或者有关于如何在Python中编写这部分代码的问题，请随时提出。‘





### 考虑气象条件的声速模型详细讲解

在声学模型中，尤其是在需要精确计算声速传播时间的应用中（如音爆位置和时间的确定），考虑气象条件（如气温、湿度和气压）对声速的影响至关重要。以下是如何将这些因素纳入模型的详细讲解：

#### 1. **声速与气温的关系**

声速 \( c \) 在空气中的速度与温度 \( \theta \)（摄氏度）有直接关系，可以通过以下公式表示：

\[
$$
c = 331.3 + 0.606 \cdot \theta
$$
\]

这个公式基于理想气体的行为，并考虑到气温升高会导致气体分子运动速度增加，从而使声波传播更快。公式中的常数 331.3 m/s 是在 0°C 时的声速，0.606 m/s/°C 是温度对声速影响的系数，反映了温度每升高一度，声速增加的速度。





# 加入音波复合衰减模型和温度影响

### Step 1: 建立问题模型

目标是确定音爆发生的精确位置（经度、纬度、高程）和时间，这需要解决四个未知数：音爆的经度 \(\text{lon}\)，纬度 \(\text{lat}\)，高程 \(z\)，以及音爆发生的时间 \(t_0\)。模型考虑了温度对声速的影响，声波在传播过程中的衰减，包括几何衰减和介质吸收。

### Step 2: 数据准备与声速调整

**Step 1: 建立问题模型**

目标是确定火箭残骸音爆发生的精确位置（经度、纬度、高程）和时间。这需要解决四个未知数：音爆的经度 \(\text{lon}\)，纬度 \(\text{lat}\)，高程 \(z\)，以及音爆发生的时间 \(t_0\)。为了找到这四个参数，我们需要将监测站接收到的音爆时间与从理论发生点计算出的时间进行匹配。

**Step 2: 数据准备**

我们的数据来源是七个监测站的记录，每个站点提供了音爆到达的时间以及站点的地理坐标（经度、纬度和高程）。对于本次建模，选择了四个监测站的数据。这些监测站位于中国湖南省，地理坐标和到达时间如下所示：

- 站点A: 经度110.241, 纬度27.204, 高程824米, 到达时间100.767秒
- 站点B: 经度110.780, 纬度27.456, 高程727米, 到达时间112.220秒
- 站点C: 经度110.712, 纬度27.785, 高程742米, 到达时间188.020秒
- 站点E: 经度110.524, 纬度27.617, 高程786米, 到达时间118.443秒

我们有四个监测站的数据，每个站提供了音爆到达的时间以及站点的地理坐标（经度、纬度和高程）。环境温度 \(T\) 影响声速 \(v\)，根据经验公式：

$$
v = 331.3 + 0.606 \cdot T
$$
给定 \(T = 18^\circ C\)，我们计算得到声速 \(v = 331.3 + 0.606 \cdot 18 = 342.01 \, \text{m/s}\)。

### Step 3: 衰减模型

声波在传播过程中会因为介质（如空气）的阻尼作用而衰减。我们假设声波的衰减遵循复合衰减模型，即几何衰减和吸收衰减相结合：

\[
$$
\text{Attenuation} = \frac{1}{r^2} e^{-\alpha r}
$$
\]

其中 \(r\) 是从音爆源到监测站的距离，\(\alpha\) 是衰减系数，可以根据声波的频率和空气特性确定。

### Step 4: 建立距离和时间关系

首先定义距离函数，即监测站到音爆发生点的欧几里得距离。对于给定的坐标 \((x, y, z)\) 和监测站坐标 \((x_i, y_i, z_i)\)，距离 \(d_i\) 由以下公式给出：

\[
$$
d_i = \sqrt{(x - x_i)^2 + (y - y_i)^2 + (z - z_i)^2}
$$
\]

音爆在每个监测站的理论到达时间 \(t_i\) 可以表示为：

\[
$$
t_i = t_0 + \frac{d_i}{v} \cdot \frac{1}{d_i^2} \cdot e^{-\alpha \cdot d_i}
$$
\]

这里的 \(\frac{d_i}{v}\) 是声波未衰减时从源点到监测站的传播时间，\(\frac{1}{d_i^2} \cdot e^{-\alpha \cdot d_i}\) 是衰减因子，它根据距离调整理论到达时间。

### Step 5: 目标函数和优化

为了估计未知参数 \(\text{lon}, \text{lat}, z, t_0\)，我们需要最小化每个监测站观测到的时间和理论到达时间之间的差异。定义一个目标函数，该函数计算所有监测站的时间差的平方和：

\[
$$
\text{Objective Function} = \sum_{i=1}^n \left( t_i^\text{observed} - \left( t_0 + \frac{d_i}{v} \cdot \frac{1}{d_i^2} \cdot e^{-\alpha \cdot d_i} \right) \right)^2
$$
\]

其中，\(t_i^\text{observed}\) 是第 \(i\) 个监测站观测到的时间。通过优化算法（如差分进化和粒子群优化）最小化目标函数，我们可以估计音爆的位置和时间，达到解决问题的目的。

### Step 7: 使用全局优化后的局部优化进行参数估计

#### 差分进化与粒子群优化（PSO）

为了估计音爆的位置和时间，模型使用了两种优化技术：差分进化和粒子群优化（PSO）。这些方法被选择用于它们在处理复杂、多参数的优化问题中表现出的效果，尤其是在全局寻优方面的优势。

- **差分进化** 是一种基于种群的优化算法，依靠种群中个体间的差异来生成新的候选解，它适用于连续参数优化，并且不需要梯度信息，非常适合于处理非线性、非凸的全局优化问题。
- **粒子群优化（PSO）** 则通过模拟鸟群狩猎行为来寻找最优解，每个“粒子”代表了问题空间中的一个潜在解。粒子根据自身和群体的经验调整飞行方向和速度，逐渐靠近最优解。

这两种方法都在目标函数上实施，该函数度量了监测站实际接收到的音爆时间与模型预测时间之间的误差平方和。



我们首先使用了差分进化算法进行全局优化，以寻找最小化实际与理论音爆时间差的音爆位置和发生时间。接着，为了进一步提高解的精度，我们使用了粒子群优化算法进行局部优化。这种先进行全局搜索，然后在找到的全局最优解周围进行局部搜索的策略，通常能够在更短的时间内找到更优的解决方案。

以下是我们采取的具体步聚：

1.全局优化：

差分进化算法o我们调用了differential_evolution函数，并传入目标函数和参数边界。这一步聚旨在通过差分进化算法在整个参数空间内进行全局搜索，以找到潜在的最优解。我们从差分进化算法的结果中获取最优解xot，它表示差分进化算法找到的全局最优解。

2.局部优化：粒子群优化算法。

我们使用pso函数进行局部优化。为了确保局部搜索的范围不过于狭窄，我们将局部搜索的下界和上界定义为全局最优解xopt的附近范围。
调用pso函数时，我们传入目标函数、局部搜索范围的下界和上界，以及一些其他参数，如粒子群大小、最大迭代次数和终止条件。
最终，pso函数返回局部优化后的最优解xopt和对应的目标函数值fopt。

这种先进行全局优化，然后再进行局部优化的方法，能够更有效地在参数空间中寻找最优解，从而提高了解的精度和稳健性。





# 讲解

您提供的模型是一个综合考虑多种环境因素以及声波衰减的复杂声学模型，它通过精确计算和优化来确定音爆事件的地理位置和时间。以下是这一模型各个步骤的详细讲解：

### Step 1: 建立问题模型

该步骤定义了需要解决的主要问题：确定音爆的地理位置（经度、纬度、高程）和精确发生时间。此问题解决依赖于对多个未知参数的估计，即音爆的地理坐标 \( \text{lon}, \text{lat}, z \) 和时间 \( t_0 \)。模型还考虑了环境因素如温度对声速的影响，以及声波在空气中传播过程中的复合衰减（几何衰减和介质吸收）。

### Step 2: 数据准备与声速调整

在这一步骤中，使用监测站记录的数据，包括音爆到达各监测站的时间和站点的地理坐标。这些数据是模型输入的基础。此外，考虑到环境温度对声速的影响，使用了声速与温度的关系式 \( v = 331.3 + 0.606 \cdot T \) 来调整声速，确保声波传播时间的计算更为准确。

### Step 3: 椭球经纬度与笛卡尔坐标转换

在地理信息系统中，为了确保位置数据的精确表示和处理，经常需要在地球椭球体模型的经纬度坐标和笛卡尔坐标系统之间进行转换。这一步骤对于确保我们可以从监测站的地理坐标准确计算到音爆发生地点的三维距离至关重要。

#### 地理坐标到笛卡尔坐标的转换：

地理坐标系统使用经度(\(\text{lon}\))、纬度(\(\text{lat}\))和高程(\(z\))来定义一个点的位置，其中经度和纬度指定了地球表面上的位置，而高程指定了相对于海平面的垂直距离。转换这些坐标到笛卡尔坐标系（通常是以地球的中心为原点的XYZ坐标系统），可以通过以下步骤完成：

1. **计算辅助参数**：
   - \( N(\phi) = \frac{a}{\sqrt{1-e^2 \sin^2(\phi)}} \)
     其中，\( a \) 是地球椭圆体的赤道半径，\( e \) 是地球椭圆体的偏心率，\(\phi\) 是纬度。

2. **从经纬度转换到XYZ坐标**：
   - \( X = (N(\phi) + z) \cos(\phi) \cos(\lambda) \)
   - \( Y = (N(\phi) + z) \cos(\phi) \sin(\lambda) \)
   - \( Z = \left(\frac{b^2}{a^2}N(\phi) + z\right) \sin(\phi) \)
     其中，\( \lambda \) 是经度，\( b \) 是极半径。

这些公式考虑了地球的非完美球形形状（扁率），提供了从地理坐标到地心地固坐标系的精确映射。

#### 笛卡尔坐标到地理坐标的转换：

反向转换，即从XYZ坐标回到经纬度和高程，是一个更复杂的过程，通常涉及迭代方法来解决非线性方程。转换的基本步骤包括：

1. **估算纬度和经度**：
   - 通过迭代或近似方法求解上述方程反算出\(\phi\)和\(\lambda\)。

2. **计算高程**：
   - 根据求得的\(\phi\)重新计算\(N(\phi)\)和Z的表达式，从而求解高程\(z\)。

这些转换过程对于本项目非常关键，因为它们允许将从监测站接收到的地理坐标转换为适合数学和物理计算的坐标系统，反之亦然。在实际应用中，这些转换通常通过专用的GIS软件或库（如Pyproj）来实现，这些工具已经内置了精确的地球模型和转换算法。

### Step 4: 衰减模型

这一步骤引入了复合衰减模型来描述声波在空气中的衰减行为。模型包括几何衰减 \( \frac{1}{r^2} \) 和介质吸收 \( e^{-\alpha r} \)，其中 \( r \) 是音爆源到监测站的距离，\(\alpha\) 是基于声波频率和空气特性确定的衰减系数。这种衰减模型允许对声波传播中的能量损失进行更精确的量化。

### Step 5: 建立距离和时间关系

该步骤定义了声波传播的基本物理规律，即从音爆源到监测站的距离 \( d_i \) 和声波的传播时间 \( t_i \)。距离使用欧几里得距离公式计算，而传播时间则结合声速、衰减模型来确定。特别地，时间计算公式考虑了声波在传播过程中因距离增加而衰减的影响。

### Step 6: 目标函数和优化

最后一步是定义和最小化目标函数，目标函数是基于监测站观测时间和理论计算时间之间的差异。通过最小化所有监测站时间差的平方和，可以估算出最可能的音爆位置和时间。使用差分进化和粒子群优化等高级算法来求解这一非线性优化问题，以确保找到全局最优解。

整个模型通过科学的方法将理论物理模型与实际应用相结合，不仅考虑了声波的物理传播特性，还融入了环境因素的影响，使得音爆源定位更加精确可靠。

### Step 7: 使用全局优化后的局部优化进行参数估计

#### 差分进化与粒子群优化（PSO）

为了估计音爆的位置和时间，模型使用了两种优化技术：差分进化和粒子群优化（PSO）。这些方法被选择用于它们在处理复杂、多参数的优化问题中表现出的效果，尤其是在全局寻优方面的优势。

- **差分进化** 是一种基于种群的优化算法，依靠种群中个体间的差异来生成新的候选解，它适用于连续参数优化，并且不需要梯度信息，非常适合于处理非线性、非凸的全局优化问题。
- **粒子群优化（PSO）** 则通过模拟鸟群狩猎行为来寻找最优解，每个“粒子”代表了问题空间中的一个潜在解。粒子根据自身和群体的经验调整飞行方向和速度，逐渐靠近最优解。

这两种方法都在目标函数上实施，该函数度量了监测站实际接收到的音爆时间与模型预测时间之间的误差平方和。



我们首先使用了差分进化算法进行全局优化，以寻找最小化实际与理论音爆时间差的音爆位置和发生时间。接着，为了进一步提高解的精度，我们使用了粒子群优化算法进行局部优化。这种先进行全局搜索，然后在找到的全局最优解周围进行局部搜索的策略，通常能够在更短的时间内找到更优的解决方案。

以下是我们采取的具体步聚：

1.全局优化：

差分进化算法o我们调用了differential_evolution函数，并传入目标函数和参数边界。这一步聚旨在通过差分进化算法在整个参数空间内进行全局搜索，以找到潜在的最优解。我们从差分进化算法的结果中获取最优解xot，它表示差分进化算法找到的全局最优解。

2.局部优化：粒子群优化算法。

我们使用pso函数进行局部优化。为了确保局部搜索的范围不过于狭窄，我们将局部搜索的下界和上界定义为全局最优解xopt的附近范围。
调用pso函数时，我们传入目标函数、局部搜索范围的下界和上界，以及一些其他参数，如粒子群大小、最大迭代次数和终止条件。
最终，pso函数返回局部优化后的最优解xopt和对应的目标函数值fopt。

这种先进行全局优化，然后再进行局部优化的方法，能够更有效地在参数空间中寻找最优解，从而提高了解的精度和稳健性。

![image-20240503101932875](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240503101932875.png)

![image-20240503103542519](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240503103542519.png)

不考虑其他影响

```python
from scipy.optimize import differential_evolution
import numpy as np
from pyproj import Transformer
from pyswarm import pso  # 引入pso函数
from geographiclib.geodesic import Geodesic
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot用于绘图
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 给定数据
stations = {
    'A': {'coords': (110.241, 27.204, 824), 'time': 100.767},
    'B': {'coords': (110.780, 27.456, 727), 'time': 112.220},
    'C': {'coords': (110.712, 27.785, 742), 'time': 188.020},
    #'D': {'coords': (110.251, 27.825, 850), 'time': 258.985},
    'E': {'coords': (110.524, 27.617, 786), 'time': 118.443},
    #'F': {'coords': (110.467, 27.921, 678), 'time': 266.871},
    #'G': {'coords': (110.047, 27.121, 575), 'time': 163.024}
}

# 声速 m/s
speed_of_sound = 340

# 创建转换器对象
transformer_geodetic_to_cartesian = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
transformer_cartesian_to_geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

# 转换经纬度到笛卡尔坐标系 (XYZ)
def convert_to_cartesian(stations):
    return {key: {'coords': transformer_geodetic_to_cartesian.transform(*value['coords']), 'time': value['time']} for key, value in stations.items()}

cartesian_stations = convert_to_cartesian(stations)

# 定义目标函数，适应pso
def objective_function_pso(p):
    x0, y0, z0, t0 = p
    total_error = 0
    for station in cartesian_stations.values():
        coord = station['coords']
        arrival_time = station['time']
        distance = np.sqrt((x0 - coord[0])**2 + (y0 - coord[1])**2 + (z0 - coord[2])**2)
        predicted_time = t0 + distance / speed_of_sound
        total_error += (predicted_time - arrival_time) ** 2
    return total_error

# 高程非负约束
def altitude_constraint(p):
    _, _, z0, _ = p
    _, _, alt = transformer_cartesian_to_geodetic.transform(p[0], p[1], p[2])
    return alt  # 高程必须非负


# 约束条件
constraint = NonlinearConstraint(altitude_constraint, 0, np.inf)

# 参数边界
coords_array = np.array([value['coords'] for value in cartesian_stations.values()])
times_array = np.array([value['time'] for value in cartesian_stations.values()])
lb = [np.min(coords_array[:, 0]), np.min(coords_array[:, 1]), np.min(coords_array[:, 2]), np.min(times_array) - 300]
ub = [np.max(coords_array[:, 0]), np.max(coords_array[:, 1]), np.max(coords_array[:, 2]), np.max(times_array) + 300]

# 全局优化：差分进化算法，应用约束
result_global = differential_evolution(objective_function_pso, bounds=list(zip(lb, ub)), constraints=[constraint])


# 局部优化：PSO
result_local_pso, _ = pso(objective_function_pso, lb, ub, swarmsize=100, maxiter=200, phip=0.5, phig=0.8, minfunc=1e-8, minstep=1e-8, debug=True)

# 创建转换器对象，确保坐标系统正确
transformer_cartesian_to_geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

# 使用转换器转换坐标
computed_x, computed_y, computed_z, computed_t0 = result_local_pso
computed_lon, computed_lat, computed_alt = transformer_cartesian_to_geodetic.transform(computed_x, computed_y, computed_z)

# 打印结果，检查高程值
print(f"音爆发生的位置和时间：")
print(f"经度: {computed_lon:.6f}°, 纬度: {computed_lat:.6f}°, 高程: {computed_alt:.2f}米")
print(f"时间: {computed_t0:.2f}秒")


# 创建转换器对象
transformer_cartesian_to_geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

# 输出结果
computed_x, computed_y, computed_z, computed_t0 = result_local_pso
computed_lon, computed_lat, computed_alt = transformer_cartesian_to_geodetic.transform(computed_x, computed_y, computed_z)

def geodesic_distance(coord1, coord2):
    geod = Geodesic.WGS84
    result = geod.Inverse(coord1[1], coord1[0], coord2[1], coord2[0])
    return result['s12']  # 返回两点之间的距离，单位为米

def print_distances_and_times(x, stations):
    total_difference = 0
    predicted_lon, predicted_lat, predicted_alt = transformer_cartesian_to_geodetic.transform(x[0], x[1], x[2])
    for key, station in stations.items():
        coord = station['coords']
        arrival_time = station['time']
        distance = geodesic_distance([predicted_lon, predicted_lat], [coord[0], coord[1]]) + abs(predicted_alt - coord[2])
        theoretical_time = x[3] + distance / speed_of_sound
        time_difference = np.abs(theoretical_time - arrival_time)
        total_difference += time_difference
        print(f"监测站 {key}:")
        print(f"  距离: {distance:.2f} 米")
        print(f"  理论到达时间: {theoretical_time:.2f} 秒")
        print(f"  时间差: {time_difference:.2f} 秒")
    print(f"总时间差: {total_difference:.2f} 秒")

# 输出距离和理论到达时间
print_distances_and_times(result_local_pso, stations)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Transformer

# 创建3D图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制监测站位置
for key, station in stations.items():
    lon, lat, alt = station['coords']
    ax.scatter(lon, lat, alt, color='blue', marker='s', label=f'监测站位置 {key}', s=100, depthshade=True)

# 使用转换器转换坐标
transformer_cartesian_to_geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
computed_lon, computed_lat, computed_alt = transformer_cartesian_to_geodetic.transform(computed_x, computed_y, computed_z)

# 绘制音爆发生的位置
ax.scatter(computed_lon, computed_lat, computed_alt, color='red', marker='o', label='碎片位置', s=100, depthshade=True)

# 连接监测站和音爆位置的虚线
for key, station in stations.items():
    lon, lat, alt = station['coords']
    ax.plot([lon, computed_lon], [lat, computed_lat], [alt, computed_alt], 'gray', linestyle='--')

# 添加标签和图例
ax.set_xlabel('经度')
ax.set_ylabel('纬度')
ax.set_zlabel('高程')
ax.set_title('3D 视图：碎片与监测站位置')

# 显示图形
ax.grid(True)
ax.legend(loc='best')

# 保存图形到本地文件
plt.savefig('3D_view_sonic_boom_location.png')
plt.show()


音爆发生的位置和时间：
经度: 110.501436°, 纬度: 27.304824°, 高程: 947.63米
时间: 17.97秒
监测站 A:
  距离: 28230.52 米
  理论到达时间: 101.00 秒
  时间差: 0.24 秒
监测站 B:
  距离: 32468.23 米
  理论到达时间: 113.47 秒
  时间差: 1.25 秒
监测站 C:
  距离: 57335.15 米
  理论到达时间: 186.61 秒
  时间差: 1.41 秒
监测站 E:
  距离: 34825.98 米
  理论到达时间: 120.40 秒
  时间差: 1.96 秒
总时间差: 4.86 秒
```



加入影响

```
from scipy.optimize import differential_evolution
import numpy as np
from pyproj import Transformer
from pyswarm import pso  # 引入pso函数
from geographiclib.geodesic import Geodesic
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot用于绘图
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 给定数据
stations = {
    'A': {'coords': (110.241, 27.204, 824), 'time': 100.767},
    'B': {'coords': (110.780, 27.456, 727), 'time': 112.220},
    'C': {'coords': (110.712, 27.785, 742), 'time': 188.020},
    #'D': {'coords': (110.251, 27.825, 850), 'time': 258.985},
    'E': {'coords': (110.524, 27.617, 786), 'time': 118.443},
    #'F': {'coords': (110.467, 27.921, 678), 'time': 266.871},
    #'G': {'coords': (110.047, 27.121, 575), 'time': 163.024}
}

# 环境温度 (摄氏度)
temperature = 18  # 假设环境温度为25摄氏度

# 声速 m/s，根据温度调整
speed_of_sound = 331.3 + 0.606 * temperature

# 衰减系数 alpha
alpha = 0.000001  # 此值需要根据实际情况调整

# 创建转换器对象
transformer_geodetic_to_cartesian = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
transformer_cartesian_to_geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

# 转换经纬度到笛卡尔坐标系 (XYZ)
def convert_to_cartesian(stations):
    return {key: {'coords': transformer_geodetic_to_cartesian.transform(*value['coords']), 'time': value['time']} for key, value in stations.items()}

cartesian_stations = convert_to_cartesian(stations)

# 定义目标函数，适应pso
def objective_function_pso(p):
    x0, y0, z0, t0 = p
    total_error = 0
    for station in cartesian_stations.values():
        coord = station['coords']
        arrival_time = station['time']
        distance = np.sqrt((x0 - coord[0])**2 + (y0 - coord[1])**2 + (z0 - coord[2])**2)
        # 计算衰减后的预测时间
        attenuation = np.exp(-alpha * distance)
        predicted_time = t0 + (distance / speed_of_sound) * attenuation
        total_error += (predicted_time - arrival_time) ** 2
    return total_error

# 高程非负约束
def altitude_constraint(p):
    _, _, z0, _ = p
    _, _, alt = transformer_cartesian_to_geodetic.transform(p[0], p[1], p[2])
    return alt  # 高程必须非负

# 约束条件
constraint = NonlinearConstraint(altitude_constraint, 0, np.inf)

# 参数边界
coords_array = np.array([value['coords'] for value in cartesian_stations.values()])
times_array = np.array([value['time'] for value in cartesian_stations.values()])
lb = [np.min(coords_array[:, 0]), np.min(coords_array[:, 1]), np.min(coords_array[:, 2]), np.min(times_array) - 300]
ub = [np.max(coords_array[:, 0]), np.max(coords_array[:, 1]), np.max(coords_array[:, 2]), np.max(times_array) + 300]

# 全局优化：差分进化算法，应用约束
result_global = differential_evolution(objective_function_pso, bounds=list(zip(lb, ub)), constraints=[constraint])

# 局部优化：PSO
result_local_pso, _ = pso(objective_function_pso, lb, ub, swarmsize=1000, maxiter=400, phip=0.6, phig=0.9, minfunc=1e-8, minstep=1e-8, debug=True)

# 使用转换器转换坐标
computed_x, computed_y, computed_z, computed_t0 = result_local_pso
computed_lon, computed_lat, computed_alt = transformer_cartesian_to_geodetic.transform(computed_x, computed_y, computed_z)


# 打印结果，检查高程值
print(f"音爆发生的位置和时间：")
print(f"经度: {computed_lon:.6f}°, 纬度: {computed_lat:.6f}°, 高程: {computed_alt:.2f}米")
print(f"时间: {computed_t0:.2f}秒")

def geodesic_distance(coord1, coord2):
    geod = Geodesic.WGS84
    result = geod.Inverse(coord1[1], coord1[0], coord2[1], coord2[0])
    return result['s12']  # 返回两点之间的距离，单位为米

def print_distances_and_times(x, stations):
    total_difference = 0
    predicted_lon, predicted_lat, predicted_alt = transformer_cartesian_to_geodetic.transform(x[0], x[1], x[2])
    for key, station in stations.items():
        coord = station['coords']
        arrival_time = station['time']
        distance = geodesic_distance([predicted_lon, predicted_lat], [coord[0], coord[1]]) + abs(predicted_alt - coord[2])
        theoretical_time = x[3] + distance / speed_of_sound
        time_difference = np.abs(theoretical_time - arrival_time)
        total_difference += time_difference
        print(f"监测站 {key}:")
        print(f"  距离: {distance:.2f} 米")
        print(f"  理论到达时间: {theoretical_time:.2f} 秒")
        print(f"  时间差: {time_difference:.2f} 秒")
    print(f"总时间差: {total_difference:.2f} 秒")

# 输出距离和理论到达时间
print_distances_and_times(result_local_pso, stations)





import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Transformer

# 创建3D图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制监测站位置
for key, station in stations.items():
    lon, lat, alt = station['coords']
    ax.scatter(lon, lat, alt, color='blue', marker='s', label=f'监测站位置 {key}', s=100, depthshade=True)

# 使用转换器转换坐标
transformer_cartesian_to_geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
computed_lon, computed_lat, computed_alt = transformer_cartesian_to_geodetic.transform(computed_x, computed_y, computed_z)

# 绘制音爆发生的位置
ax.scatter(computed_lon, computed_lat, computed_alt, color='red', marker='o', label='碎片位置', s=100, depthshade=True)

# 连接监测站和音爆位置的虚线
for key, station in stations.items():
    lon, lat, alt = station['coords']
    ax.plot([lon, computed_lon], [lat, computed_lat], [alt, computed_alt], 'gray', linestyle='--')

# 添加标签和图例
ax.set_xlabel('经度')
ax.set_ylabel('纬度')
ax.set_zlabel('高程')
ax.set_title('3D 视图：碎片与监测站位置')

# 显示图形
ax.grid(True)
ax.legend(loc='best')

# 保存图形到本地文件
plt.savefig('3D_view_sonic_boom_location.png')
plt.show()

音爆发生的位置和时间：
经度: 110.503305°, 纬度: 27.293780°, 高程: 926.78米
时间: 21.53秒
监测站 A:
  距离: 27920.03 米
  理论到达时间: 103.12 秒
  时间差: 2.35 秒
监测站 B:
  距离: 32946.17 米
  理论到达时间: 117.80 秒
  时间差: 5.58 秒
监测站 C:
  距离: 58390.51 米
  理论到达时间: 192.16 秒
  时间差: 4.14 秒
监测站 E:
  距离: 36015.42 米
  理论到达时间: 126.77 秒
  时间差: 8.33 秒
总时间差: 20.40 秒
```























# ***\*问题2\**** 火箭残骸除了一级残骸，还有两个或者四个助推器。在多个残骸发生音爆时，监测设备在监测范围内可能会采集到几组音爆数据。假设空中有4个残骸，每个设备按照时间先后顺序收到4组震动波。建立数学模型，分析如何确定监测设备接收到的震动波是来自哪一个残骸？如果要确定4个残骸在空中发生音爆时的位置和时间，至少需要布置多少台监测设备？

![image-20240510224048599](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510224048599.png)

![image-20240510224058567](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510224058567.png)

这个代码示例描述了一个数学模型，用于确定音爆发生的精确位置和时间。该模型采用监测站的位置和声音到达时间数据，通过全局优化和局部优化的组合方法来估计音爆的源位置和发生时间。以下是模型的详细解释与数学公式：

### 数学模型详解

#### 1. 坐标转换

首先，模型需要将监测站的地理坐标（经度、纬度、高程）转换为笛卡尔坐标系（X, Y, Z）。这是因为在笛卡尔坐标系中处理距离和速度计算更为直接和精确。转换使用的是椭球地理坐标系到三维地心坐标系的转换，具体转换公式如下：

- 使用Pyproj的转换器（基于EPSG代码：4326为WGS 84坐标系，4978为WGS 84的地心坐标系）进行坐标转换。

#### 2. 目标函数定义

模型定义了一个目标函数，该函数计算了从假设的音爆源位置到各个监测站的理论到达时间，并将其与实际观测到的到达时间进行比较。目标函数的主要目的是最小化理论到达时间与实际到达时间之间的差的平方和，从而找到最可能的音爆源位置和时间。

$$
f(\mathbf{p}) = \sum \left( t_0 + \frac{\sqrt{(x_0 - x_i)^2 + (y_0 - y_i)^2 + (z_0 - z_i)^2}}{v} - t_i \right)^2
$$

其中:

- \( \mathbf{p} = [x_0, y_0, z_0, t_0] \) 是音爆源的假设位置和时间。
- \( (x_i, y_i, z_i) \) 是第 \(i\) 个监测站的笛卡尔坐标。
- \( t_i \) 是音爆在第 \(i\) 个监测站的实际到达时间。
- \( v \) 是声速，这里取为 340 m/s。

#### 3. 优化算法

模型采用两阶段优化方法：

- **全局优化**：使用差分进化算法，该算法适用于多维且可能非线性、非凸的全局优化问题。它不需要目标函数的梯度信息，特别适用于处理复杂的优化问题。

- **局部优化**：使用非线性最小二乘法细化全局优化的结果。这一步骤旨在通过局部搜索进一步减小目标函数值，提高参数估计的精确度。

#### 4. 参数边界

优化过程中为参数设定了边界，确保搜索过程中参数值的合理性。边界基于监测站的笛卡尔坐标和声音到达时间的范围确定。



# 改进

在您描述的复杂场景中，监测站需要区分并正确归因于来自不同火箭残骸（包括一级残骸和助推器）的音爆。以下是详细的数学模型分析，说明如何通过数学方法和算法解决这个问题。

### 数学模型概述

模型的核心是使用时间差分和位置数据来解决多源定位问题，即同时确定多个音爆源的位置和时间。模型包括数据预处理、成本矩阵构建、信号匹配优化、以及结果的验证和可视化。

### 步骤一：数据预处理和坐标转换

1. **地理坐标到笛卡尔坐标的转换**：
   - 使用椭球模型参数，将每个监测站和残骸的地理坐标（经度、纬度和高程）转换为笛卡尔坐标系统（X, Y, Z）。
   - 公式：
     \[
     X = (N + h) \cos(\phi) \cos(\lambda), \quad Y = (N + h) \cos(\phi) \sin(\lambda), \quad Z = \left(\frac{b^2}{a^2} N + h\right) \sin(\phi)
     \]
     其中 \(N\) 是法线的半径，\(h\) 是高程，\(\phi\) 和 \(\lambda\) 分别是纬度和经度。

### 步骤二：成本矩阵构建和信号匹配

2. **扩展成本矩阵**：
   - 根据每个监测站接收到的多组震动波时间，构建一个成本矩阵，其中矩阵的元素反映了每个残骸到每个监测站的理论到达时间。
   - 成本矩阵扩展，使得每个监测站的每组信号可以与任一残骸的到达时间进行匹配。

3. **匈牙利算法应用**：
   - 使用匈牙利算法（也称为Kuhn-Munkres算法）处理扩展的成本矩阵，找到成本最低的匹配方案，即确定每个监测站接收到的震动波最可能来自哪个残骸。

### 步骤三：优化算法应用

4. **遗传算法优化**：
   - 通过遗传算法优化残骸的预测位置和时间，以最小化所有监测站观测到的时间和通过成本矩阵计算的理论时间之间的差异。
   - 包括适应度函数的设计，遗传操作如交叉、变异和选择的实施。

### 步骤四：结果验证和可视化

5. **结果可视化**：
   - 使用3D图形展示每个监测站和残骸的位置，以及确定的连接线，直观展示哪些监测站的信号被归因于特定的残骸。
   - 可视化帮助验证模型的准确性和解释性。

### 结论和应用

该数学模型通过综合应用坐标转换、成本矩阵构建、匈牙利算法和遗传算法，有效地解决了多源音爆事件的监测和归因问题。该模型不仅提供了一个强大的分析工具来处理复杂的实时数据，还通过可视化增强了结果的解释能力，使得操作者可以准确地识别和响应多源音爆事件。这种方法的成功实施为火箭发射和其他需要精确

监测多个动态源的领域提供了技术支持。



要解决这个问题，我们需要考虑如何使用一对多的匹配策略来确定每个监测设备接收到的震动波是来自哪一个残骸。在这里我改进了匈牙利模型



### 匈牙利模型（Kuhn-Munkres算法）

匈牙利模型，也称为Kuhn-Munkres算法或分配问题，是一种在多项式时间内解决分配问题的经典算法。它主要用于解决一对一的分配问题，即将一组资源分配给一组任务，使得总成本最小化或总效益最大化。

#### 数学模型

假设有 \( n \) 个任务和 \( n \) 个资源，每个任务分配到一个资源有一个成本或效益。成本或效益可以用一个 \( n \times n \) 的矩阵 \( C \) 表示，其中 \( C[i][j] \) 表示第 \( i \) 个任务分配给第 \( j \) 个资源的成本。

目标是找到一个分配方案，使得总成本最小（或总效益最大）。这可以表示为以下优化问题：


$$
\text{minimize} \sum_{i=1}^{n} C[i][\sigma(i)]
$$


其中，\( \sigma \) 是从任务集到资源集的一个双射（一一对应的映射）。

#### 算法步骤

1. **初始化标签**：为每个任务和资源初始化一个标签（潜在变量），通常任务的标签为该行的最小值，资源的标签为0。
2. **寻找可行匹配**：使用标签创建一个等价的任务，寻找最大匹配。
3. **检查完美匹配**：如果找到完美匹配，则算法结束。如果没有，调整标签并重复寻找过程。

## **改进的匈牙利模型**



对于一对多的匹配问题，例如一个设备接收来自多个残骸的信号，传统的匈牙利模型需要调整以适应这种情况。改进的匈牙利模型可以通过以下方式实现：

#### 扩展成本矩阵

为了允许一个资源（如设备）与多个任务（如残骸）匹配，可以通过扩展成本矩阵来实现。具体方法是复制资源行，使得每个资源可以被分配给多个任务。

#### 数学模型

设 \( m \) 为任务的数量，\( n \) 为资源的数量，每个资源可以被分配给 \( k \) 个任务。成本矩阵 \( C \) 扩展为 \( nk \times m \)。目标是：


$$
\text{minimize} \sum_{i=1}^{nk} C[i][\sigma(i)]
$$


其中，\( \sigma \) 是从扩展的资源集到任务集的映射。

#### 算法调整

1. **扩展成本矩阵**：将每个资源的行复制 \( k \) 次。
2. **应用匈牙利算法**：在扩展的成本矩阵上应用标准的匈牙利算法。
3. **解析结果**：将得到的匹配结果映射回原始的资源和任务。

这种改进的匈牙利模型允许更灵活的匹配策略，适用于复杂的实际应用场景，如多源信号处理和资源分配问题。







### 匈牙利模型（Kuhn-Munkres算法）详细介绍

匈牙利模型，也称为Kuhn-Munkres算法，是一种有效解决分配问题的算法，特别是在需要最小化成本或最大化效益的一对一分配问题中。这种算法在计算机科学和运筹学中广泛应用。

#### 数学模型

考虑一个分配问题，其中有 \( n \) 个任务和 \( n \) 个资源。每个任务分配给一个资源有一个特定的成本，这些成本可以通过一个 \( n \times n \) 的成本矩阵 \( C \) 来表示，其中矩阵的元素 \( C[i, j] \) 表示将第 \( i \) 个任务分配给第 \( j \) 个资源的成本。

目标是找到一个完美匹配的分配方案，使得总成本最小化：


\text{minimize} \quad \sum_{i=1}^{n} C[i, \sigma(i)]



其中，\( \sigma \) 是从任务集到资源集的一个双射（即一一对应的映射）。

#### 算法步骤

1. **初始化标签**：
   - 为每个任务 \( i \) 初始化一个标签 \( u_i \) 为该行的最小值，即 \( u_i = \min_j C[i, j] \)。
   - 为每个资源 \( j \) 初始化一个标签 \( v_j = 0 \)。

2. **寻找可行匹配**：
   - 使用这些标签，构建一个等价的成本矩阵 \( C' \)，其中 \( C'[i, j] = C[i, j] - u_i - v_j \)。
   - 在 \( C' \) 中寻找零成本的边，尝试构建最大匹配。

3. **检查完美匹配**：
   - 如果找到完美匹配，算法结束。
   - 如果没有找到，调整标签以创建更多的零成本边，重复寻找过程。

### 改进的匈牙利模型

对于一对多的匹配问题，传统的匈牙利模型需要调整以适应这种情况。这通常通过扩展成本矩阵来实现，允许一个资源与多个任务匹配。

#### 扩展成本矩阵

设 \( m \) 为任务的数量，\( n \) 为资源的数量，每个资源可以被分配给 \( k \) 个任务。我们将成本矩阵 \( C \) 扩展为 \( nk \times m \)，其中每个资源的行被复制 \( k \) 次。

#### 数学模型

目标是找到一个分配方案，使得总成本最小化：


$$
\text{minimize} \quad \sum_{i=1}^{nk} C[i, \sigma(i)]
$$


其中，\( \sigma \) 是从扩展的资源集到任务集的映射。

#### 算法调整

1. **扩展成本矩阵**：
   - 将每个资源的行复制 \( k \) 次，形成新的扩展成本矩阵。

2. **应用匈牙利算法**：
   - 在扩展的成本矩阵上应用标准的匈牙利算法，寻找最小成本的完美匹配。

3. **解析结果**：
   - 将得到的匹配结果映射回原始的资源和任务，以确定每个资源对应的多个任务。

这种改进的匈牙利模型允许更灵活的匹配策略，对应的多个任务。

这种改进的匈牙利模型允许更灵活的匹配策略，适用于复杂的实际应用场景，如多源信号处理和资源分配问题。通过扩展成本矩阵，我们可以模拟一个资源（如监测设备）接收来自多个任务（如不同残骸的信号）的情况。这对于处理如空中多残骸定位等复杂问题特别有用。

### 扩展成本矩阵的数学表达

假设原始成本矩阵 \( C \) 的尺寸为 \( n \times m \)，其中 \( n \) 是资源的数量，\( m \) 是任务的数量。如果每个资源可以处理 \( k \) 个任务，成本矩阵可以扩展为 \( nk \times m \)。扩展后的矩阵 \( C' \) 可以表示为：


$$
C' = \begin{bmatrix}
C \\
C \\
\vdots \\
C
\end{bmatrix}
$$


这里，矩阵 \( C \) 被复制 \( k \) 次。每个资源的每一行代表该资源可以被分配给任何一个任务的成本。

### 匹配过程

在扩展的成本矩阵 \( C' \) 上应用匈牙利算法，我们寻找一个使总成本最小的匹配。这个过程包括：

1. **标签初始化**：对于扩展后的每个资源和每个任务，初始化标签。
2. **寻找可行匹配**：基于当前的标签，寻找成本为零的匹配。
3. **调整标签和更新匹配**：如果未找到完美匹配，调整标签以创建更多的零成本边，然后重复匹配过程。

### 结果解析

匹配完成后，我们需要将扩展矩阵中的结果映射回原始资源和任务。这涉及到解析每个资源对应的多个任务，确保每个资源的分配不超过 \( k \) 个任务。

### 实际应用

这种改进的匈牙利模型特别适合于那些需要资源进行多任务处理的场景，例如：

- **多源信号匹配**：如多个传感器需要同时处理来自多个源的信号。
- **任务调度**：在需要将有限的机器或工作站分配给多个作业的生产环境中。
- **网络流量分配**：在数据中心，将服务器资源分配给处理多个请求。

通过这种方法，可以有效地解决一对多和多对多的匹配问题，提高资源利用效率，优化系统的整体性能。





### 数学模型和解决方案

1. **信号传播模型**：

   - 假设每个残骸在空中的位置为 \( (x_i, y_i, z_i) \) ，发生音爆的时间为 \( t_i \)。

   - 每个监测设备的位置为 \( (x_j', y_j', z_j') \)。

   - 信号从残骸到设备的传播时间 \( \tau
     $$
     \tau_{ij} = \frac{\sqrt{(x_i - x_j')^2 + (y_i - y_j')^2 + (z_i - z_j')^2}}{c}
     $$

   - 设备 \( j \) 在时间 \( t
     $$
     _{ij}' \) 接收到来自残骸 \( i \) 的信号：
     \[
     t_{ij}' = t_i + \tau_{ij}
     $$

2. **确定设备接收信号的来源**：

   - 对于每个设备接收到的信号，计算与所有残骸的理论到达时间，并找出最接近的匹配。
   - 可以使用多目标优化方法，如多目标遗传算法，来同时优化多个设备接收到的信号与所有残骸的匹配。

3. **所需的监测设备数量**：

   - 理论上，为了确定空间中四个点的位置和时间，至少需要四个独立的测量，因为每个点有四个未知数（三个空间坐标和时间）。
   - 实际上，由于信号传播的复杂性和可能的测量误差，可能需要更多的设备来确保足够的覆盖和冗余，从而提高定位的准确性和可靠性。



```python
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



```



在你的代码中，使用了多种技术和算法来解决问题，其中包括经纬度转换、到达时间计算、信号匹配和遗传算法优化。下面将详细解释每一部分的数学模型和对应的计算过程。

### 1. 经纬度到笛卡尔坐标的转换

这一步骤涉及将地球表面的点从地理坐标（经度、纬度和高度）转换为笛卡尔坐标系（X, Y, Z）。数学模型为：


$$
\begin{align*}
a &= 6378137.0 \text{ (地球赤道半径)} \\
f &= \frac{1}{298.257223563} \text{ (扁率)} \\
e^2 &= f \cdot (2 - f) \\
N &= \frac{a}{\sqrt{1 - e^2 \sin^2(\phi)}} \\
X &= (N + h) \cdot \cos(\phi) \cdot \cos(\lambda) \\
Y &= (N + h) \cdot \cos(\phi) \cdot \sin(\lambda) \\
Z &= \left((1 - e^2) N + h\right) \cdot \sin(\phi)
\end{align*}
$$


其中，\( \phi \) 是纬度，\( \lambda \) 是经度，\( h \) 是高度。

### 2. 到达时间的计算

使用了声速在介质中的传播时间来估算每个设备到残骸的到达时间。到达时间 \( t \) 的计算公式为：


t = \frac{d}{v} + t_0



其中，\( d \) 是残骸到设备的距离，\( v \) 是声速（340 m/s），\( t_0 \) 是残骸爆炸发生的时间。距离 \( d \) 通过欧几里得距离公式计算：


d = \sqrt{(X_1 - X_2)^2 + (Y_1 - Y_2)^2 + (Z_1 - Z_2)^2}



### 3. 匹配信号到残骸

为处理多信号与多残骸的匹配，使用了改进的匈牙利算法，其数学模型为最优化分配问题。这个问题的目标是最小化总成本：


$$
\text{minimize} \quad \sum_{i,j} C_{ij} x_{ij}
$$


其中，\( C_{ij} \) 是设备 \( i \) 到残骸 \( j \) 的成本（到达时间差），\( x_{ij} \) 是分配矩阵的元素，表示设备 \( i \) 是否分配到残骸 \( j \)。

在您的问题中，提到了使用匈牙利算法对多信号与多残骸进行匹配，这涉及到一个典型的任务分配问题，其中的目标是最小化总成本。为了应对多个信号可能来自同一监测设备的情况，您采用了一种改进的匈牙利算法，允许每个设备与多个残骸进行匹配。这种方法通常适用于当设备能接收多个信号源时的情况。

### 改进匈牙利算法的原理

改进的匈牙利算法是针对一对多分配问题的解决方案，它能够将单个设备的多个信号与多个残骸进行匹配。这在现实世界问题如多目标跟踪、资源分配等领域有广泛的应用。

### 数学模型

1. **扩展成本矩阵**：首先将成本矩阵 \(C\) 扩展以适应每个设备可分配给多个残骸的情况。如果设备 \(i\) 可以分配给 \(k\) 个不同的残骸，我们就将该设备对应的行在成本矩阵中重复 \(k\) 次。这样做的目的是允许一个设备分配多个信号到不同的残骸，每个信号视作独立的任务。

   扩展后的成本矩阵 \(C_{\text{expanded}}\) 形式如下：


   $$
   C_{\text{expanded}} = \begin{bmatrix}
   C_{11} & C_{12} & \cdots & C_{1n} \\
   C_{11} & C_{12} & \cdots & C_{1n} \\
   \vdots & \vdots & \ddots & \vdots \\
   C_{11} & C_{12} & \cdots & C_{1n} \\
   \vdots & \vdots & \ddots & \vdots \\
   C_{m1} & C_{m2} & \cdots & C_{mn} \\
   C_{m1} & C_{m2} & \cdots & C_{mn} \\
   \end{bmatrix}
   $$

   其中每行对应一个设备的一个信号。

2. **最小化总成本**：在得到扩展后的成本矩阵之后，目标是找到一个分配策略 \(X\)，使得总成本最小化：


   $$
   \text{minimize} \quad \sum_{i,j} C_{ij} x_{ij}
   $$
   {ij} = 1\) 如果设备 \(i\) 的信号分配给残骸 \(j\)，否则 \(x_{ij} = 0\)。

### 实现细节

- **匈牙利算法**：通过应用匈牙利算法或KM算法（Kuhn-Munkres算法）来解决这个分配问题。这些算法能够有效地找到使总成本最小化的最佳匹配。
- **行索引调整**：由于成本矩阵被扩展，每个设备可能对应多行，因此在得到匹配结果后，需要通过模运算将扩展后的行索引转换回原始设备的索引。

### 结果解释

这种方法的优势在于它能够灵活地处理每个设备接收到多个信号的情况，并将这些信号准确地分配到合适的残骸。这在现实中非常有用，特别是在信号可能由于噪声或干扰而不止一个信号对应一个设备的情况下。



### 4. 遗传算法优化

使用遗传算法优化残骸的位置和时间，最小化到达时间的预测误差。目标函数为：


$$
\text{minimize} \quad \sum_{i=1}^{n} \left( t_{\text{predicted},i} - t_{\text{observed},i} \right)^2 - \lambda \cdot \text{reward}
$$


其中，\( t_{\text{predicted},i} \) 是根据模型预测的到达时间，\( t_{\text{observed},i} \) 是观测到的到达时间，\( \lambda \) 是用来调整设备数量对优化结果的影响的惩罚系数。

这个复杂的流程不仅涉及多物理量的转换和计算，还包括了先进的优化算法，能够处理多信号和多目标的情况，为现实世界中类似的问题提供了一种有效的解决方案。







本数学模型专为处理多源音爆定位问题设计，涉及将来自不同火箭残骸的音爆正确归因给各监测站所接收的信号。该模型综合了坐标转换、信号传播时间计算、多信号匹配优化，以及遗传算法等数学方法和算法。

### 1. 坐标转换

首先，所有监测站和音爆源（火箭残骸）的地理坐标（经度、纬度、高程）需要转换为笛卡尔坐标系统（X, Y, Z），以便进行精确的距离和方向计算。转换公式基于以下椭球模型参数：

- **赤道半径** \( a = 6378137.0 \) 米
- **扁率** \( f = 1/298.257223563 \)
- **第一偏心率平方** \( e^2 = f \cdot (2-f) \)

给定纬度 \( \phi \) 和经度 \( \lambda \)，以及高程 \( h \)，转换公式为：

$$
N = \frac{a}{\sqrt{1 - e^2 \sin^2(\phi)}} \\
X = (N + h) \cos(\phi) \cos(\lambda) \\
Y = (N + h) \cos(\phi) \sin(\lambda) \\
Z = ((1 - e^2) N + h) \sin(\phi)
$$

这一步是为了确保所有后续计算的空间一致性和精度。

### 2. 到达时间计算

每个监测站收到来自不同残骸的信号的理论到达时间通过以下方式计算：

$$
t_{arrival} = t_{emission} + \frac{d}{v}
$$

其中，\( d \) 是从残骸到监测站的欧氏距离，\( v \) 是声速（大约为 340 m/s），\( t_{emission} \) 是残骸产生音爆的时间。距离 \( d \) 的计算公式为：

$$
d = \sqrt{(X_1 - X_2)^2 + (Y_1 - Y_2)^2 + (Z_1 - Z_2)^2}
$$

### 3. 多信号匹配优化

为解决每个监测站可能接收到来自不同残骸的多个信号的问题，使用了改进的匈牙利算法进行信号匹配。首先构建成本矩阵 \( C \)，其中 \( C[i, j] \) 表示第 \( i \) 个监测站接收到的信号与第 \( j \) 个残骸理论到达时间的差的平方。

由于每个监测站可能接收到多个信号，成本矩阵扩展以允许每个设备行对应多个任务，使每个设备可以与多个残骸进行匹配：

$$
C_{expanded} = \text{tile}(C, (k, 1))
$$

1. **扩展成本矩阵**：首先将成本矩阵 \(C\) 扩展以适应每个设备可分配给多个残骸的情况。如果设备 \(i\) 可以分配给 \(k\) 个不同的残骸，我们就将该设备对应的行在成本矩阵中重复 \(k\) 次。这样做的目的是允许一个设备分配多个信号到不同的残骸，每个信号视作独立的任务。

   扩展后的成本矩阵 \(C_{\text{expanded}}\) 形式如下：

   

   $$
   C_{\text{expanded}} = \begin{bmatrix}
   C_{11} & C_{12} & \cdots & C_{1n} \\
   C_{11} & C_{12} & \cdots & C_{1n} \\
   \vdots & \vdots & \ddots & \vdots \\
   C_{11} & C_{12} & \cdots & C_{1n} \\
   \vdots & \vdots & \ddots & \vdots \\
   C_{m1} & C_{m2} & \cdots & C_{mn} \\
   C_{m1} & C_{m2} & \cdots & C_{mn} \\
   \end{bmatrix}
   $$

   其中每行对应一个设备的一个信号。

2. **最小化总成本**：在得到扩展后的成本矩阵之后，目标是找到一个分配策略 \(X\)，使得总成本最小化：

   

   $$
   \text{minimize} \quad \sum_{i,j} C_{ij} x_{ij}
   $$
   {ij} = 1\) 如果设备 \(i\) 的信号分配给残骸 \(j\)，否则 \(x_{ij} = 0\)。

使用匈牙利算法找到最小成本匹配，将扩展后的行索引转换回原始设备的索引。

### 4. 遗传算法优化

最后，使用遗传算法进一步优化残骸的预测位置和时间，以最小化所有监测站观测到的时间与模型预测时间之差。适应度函数定义为：

$$
\text{minimize} \quad \sum_{i=1}^{n} \left( t_{\text{predicted},i} - t_{\text{observed},i} \right)^2 - \lambda \cdot \text{number of devices}
$$

这里 \( t_{\text{predicted},i} \) 是根据模型预测的到达时间，\( t_{\text{

observed},i} \) 是实际观测到的到达时间，\( \lambda \) 是正则化参数，用来调节设备数量对总体匹配质量的影响。

### 5. 结果验证和可视化

使用三维图表展示每个监测站和残骸的位置，验证匹配结果的准确性，并通过视觉方法直观展示模型的效果。这不仅有助于验证模型的准确性，也提高了用户对模型输出的信任度和理解。

综上所述，该数学模型通过一系列精细的步骤处理复杂的多源音爆事件，提供了一种高效的技术解决方案，特别适用于需要精确监测和快速响应的情境。



# 分析

从提供的数据和实验结果来看，我们可以进行以下分析和解释：

### 实验设计和结果

1. **不加匈牙利算法的情况**：
   - 实验中监测设备的数量统计结果显示了不同数量设备的优化结果次数：7个设备出现38次，5个设备21次，6个设备16次，而4个设备出现25次。
   - 这种分布表明，没有使用匈牙利算法时，多种配置的设备数量都有可能产生最优结果，显示了问题的解空间较为分散，没有明显的最优设备数量。

2. **加上匈牙利算法的情况**：
   - 当使用匈牙利算法进行优化后，实验中7个设备的配置每次都是最优配置，占比达到了100%。
   - 这表明加入匈牙利算法后，系统优化的效果更加集中且稳定，提升了特定配置（此处为7个设备）下的性能表现。

### 分析

**匈牙利算法的影响**：

- 匈牙利算法是一种著名的最优分配算法，它通过计算最低成本的匹配来优化资源分配。在本实验中，引入匈牙利算法可能帮助系统更有效地将监测设备的信号与特定的音爆源匹配，从而优化整体的定位精度和效率。
- 匈牙利算法的引入减少了解空间的不确定性，使某些设备配置（如7个设备）显示出更高的性能稳定性和优越性。这可能是因为当设备数量足够多时，匈牙利算法能够更全面地利用所有可用数据，从而实现更精确的匹配和定位。

**对操作实践的启示**：

- 在实际操作中，如果条件允许，使用较多的监测设备并结合匈牙利算法进行数据处理，可以显著提高事件响应的准确性和效率。特别是在复杂的多源音爆事件中，这种方法可以显著减少误匹配和提高定位精度。
- 此外，系统设计时应考虑到算法的计算复杂性和实际可操作性，合理选择设备数量和算法策略，以达到成本效益最优化。

### 结论

通过对比实验结果可以看出，匈牙利算法的引入对多源音爆定位问题的解决具有显著影响。它不仅提高了特定配置下的定位成功率，还提升了系统的整体性能表现。因此，在处理类似的复杂定位问题时，应考虑采用适当的数学优化算法，以确保结果的准确性和系统的高效运行。





# ***\*问题3\**** 假设各台监测设备布置的坐标和4个音爆抵达时间分别如下表所示：

| 设备 | 经度(°) | 纬度(°) | 高程(m) | 音爆抵达时间(s) |         |         |         |
| ---- | ------- | ------- | ------- | --------------- | ------- | ------- | ------- |
| A    | 110.241 | 27.204  | 824     | 100.767         | 164.229 | 214.850 | 270.065 |
| B    | 110.783 | 27.456  | 727     | 92.453          | 112.220 | 169.362 | 196.583 |
| C    | 110.762 | 27.785  | 742     | 75.560          | 110.696 | 156.936 | 188.020 |
| D    | 110.251 | 28.025  | 850     | 94.653          | 141.409 | 196.517 | 258.985 |
| E    | 110.524 | 27.617  | 786     | 78.600          | 86.216  | 118.443 | 126.669 |
| F    | 110.467 | 28.081  | 678     | 67.274          | 166.270 | 175.482 | 266.871 |
| G    | 110.047 | 27.521  | 575     | 103.738         | 163.024 | 206.789 | 210.306 |

利用问题2所建立的数学模型，从上表中选取合适的数据，确定4个残骸在空中发生音爆时的位置和时间（4个残骸产生音爆的时间可能不同，但互相差别不超过5 s）。



为了解决问题3，即确定4个残骸在空中发生音爆时的位置和时间，我们可以使用多源定位技术结合数学优化方法。以下是具体的数学模型和步骤描述：

### 步骤 1: 数据准备和坐标转换

首先，需要将给定的监测设备的地理坐标（经度、纬度、高程）转换为笛卡尔坐标系统（X, Y, Z），以便进行空间计算。转换公式基于地球椭球模型，计算如下：

- **椭球模型参数**：
  - 赤道半径 \(a = 6378137.0\) 米
  - 扁率 \(f = 1/298.257223563\)
  - 第一偏心率平方 \(e^2 = f \cdot (2 - f)\)

- **坐标转换公式**：
  \[
  N = \frac{a}{\sqrt{1 - e^2 \sin^2(\phi)}}
  \]
  \[
  X = (N + h) \cos(\phi) \cos(\lambda), \quad Y = (N + h) \cos(\phi) \sin(\lambda), \quad Z = \left(\frac{b^2}{a^2} N + h\right) \sin(\phi)
  \]
  其中，\(\phi\) 和 \(\lambda\) 是纬度和经度的弧度值，\(h\) 是高程。

### 步骤 2: 建立时间和距离模型

使用声速（约340 m/s）计算从每个残骸发生音爆的位置到各个监测站的理论到达时间。计算公式为：

- **距离公式**：
  \[
  d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}
  \]
- **时间公式**：
  \[
  t_{ij} = t_{0i} + \frac{d_{ij}}{v}
  \]
  其中 \(t_{0i}\) 是第 \(i\) 个残骸音爆发生的时间，\(v\) 是声速。

### 步骤 3: 定义优化问题

为了确定四个残骸的位置和音爆时间，需要解决一个优化问题，即最小化所有监测站观测到的音爆到达时间与计算的理论到达时间之间的差异。优化目标函数为：

- **目标函数**：
  \[
  \text{Objective Function} = \sum_{i=1}^{4} \sum_{j=1}^{7} (t_{ij}^\text{observed} - (t_{0i} + \frac{d_{ij}}{v}))^2
  \]

### 步骤 4: 使用优化算法求解

可以使用非线性最小二乘法或全局优化算法如差分进化算法求解该优化问题。这些方法可以帮助我们找到最可能的残骸位置和音爆时间，从而最小化目标函数。

### 实现和结果验证

实现上述模型后，需要通过实际数据验证优化结果的准确性。可以使用仿真数据测试模型，并通过可视化方法（如绘制残骸位置和监测站位置的三维图）来直观地检查优化结果。

通过上述步骤，我们能够有效地使用提供的监测设备数据来确定四个不同火箭残骸音爆发生的精确位置和时间。这种方法不仅适用于本问题，还可以扩展到其他类似的多源定位问题。

```python
import numpy as np
from pyproj import Transformer
from scipy.optimize import NonlinearConstraint, differential_evolution
from geographiclib.geodesic import Geodesic
from tqdm import tqdm
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


# 设定距离的最小阈值（例如：至少每个残骸之间相隔5000米）
min_distance = 5000

# 在全局作用域中定义 geod 对象
geod = Geodesic.WGS84

def objective_function(variables):
    num_debris = 4
    total_error = 0
    debris_positions = variables[:num_debris*3].reshape(num_debris, 3)
    debris_times = variables[num_debris*3:num_debris*4]

    # 检查所有残骸之间的地理距离并添加惩罚
    for i in range(num_debris):
        for j in range(i + 1, num_debris):
            lon1, lat1, _ = debris_positions[i]
            lon2, lat2, _ = debris_positions[j]
            distance = geod.Inverse(lat1, lon1, lat2, lon2)['s12']  # 使用正确的方法和参数顺序
            if distance < min_distance:
                total_error += (min_distance - distance)**2 * 1000  # 添加大的惩罚

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
                total_error += (predicted_time - arrival_time)**2

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


known_positions = np.array([
    [110.43966291532242, 27.670196586093926],
    [110.43966298803502, 27.67019659009026],
    [110.43966353689149, 27.67019660794739],
    [110.43966335115412, 27.670196570602528]
])

def exclusion_constraint(variables):
    num_debris = 4
    # Assuming each debris has three coordinates and one time value
    current_positions_times = variables.reshape(num_debris, 4)  # Reshape to consider longitude, latitude, altitude, and time
    for known in known_positions:
        for current in current_positions_times:
            # Check if the positions are close to known positions
            if np.allclose(current[:2], known, atol=1e-6):  # More strict check, only comparing the first two elements (longitude, latitude)
                return -1000  # If a solution is too close to a known one, return a large negative penalty
    return 0  # If no solutions are too close, return 0



# 参数边界，考虑XYZ坐标和时间
num_debris = 4
# 根据stations中的最大和最小时间来设置时间范围
min_time = min(min(station['times']) for station in stations.values())
max_time = max(max(station['times']) for station in stations.values())
bounds = [(100, 120), (20, 40), (100, 10000)] * num_debris + [(0,min_time)] * num_debris  # XYZ坐标范围和时间范围

# 约束条件
constraints = [
    NonlinearConstraint(time_difference_constraint, 0, np.inf),
    NonlinearConstraint(time_prior_constraint, 0, np.inf),
    NonlinearConstraint(altitude_constraint, 0, np.inf),  # Missing comma here might cause an issue
    NonlinearConstraint(exclusion_constraint, 0, np.inf)
]

# 使用差分进化算法求解
initial_guess = np.random.rand(num_debris * 3 + num_debris) * 1000  # 随机初始猜测


# 初始化进度条
progress_bar = tqdm(total=100, desc='Optimization Progress')

def update_progress(xk, convergence):
    # 更新进度条的当前状态
    progress_bar.update(1)  # 每次调用时进度条增加1
    return False  # 返回False以继续优化
# 执行差分进化算法
result = differential_evolution(
    objective_function,
    bounds=bounds,
    constraints=constraints,
    strategy='best1bin',
    maxiter=200,
    popsize=20,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback=None,
    disp=True,
    polish=True,
    init='random',
    atol=0
)

# 关闭进度条
progress_bar.close()

# 提取结果
if result.success:
    estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
    estimated_times = result.x[num_debris*3:num_debris*4]
    print("优化成功，找到可能的音爆源位置和时间")
    for i in range(num_debris):
        print(f"残骸 {i+1}: 位置 ({estimated_positions[i][0]}, {estimated_positions[i][1]}, {estimated_positions[i][2]}), 时间 {estimated_times[i]}")
else:
    print("优化失败：", result.message)

# 这段代码将设置所有必要的边界和约束，并运行差分进化算法以找到最优解，这里我们假设每个残骸的音爆发生位置和时间可以通过全局优化方法估计。






(base) root@dsw-82132-68df4f4dfb-wx2sq:/mnt/workspace# /opt/conda/bin/python "/mnt/workspace/c31x copy 13.py"
Optimization Progress:   0%|                                                                                | 0/100 [00:00<?, ?it/s]differential_evolution step 1: f(x)= inf
differential_evolution step 2: f(x)= inf
differential_evolution step 3: f(x)= 4.95038e+08
differential_evolution step 4: f(x)= 3.70942e+08
differential_evolution step 5: f(x)= 3.70942e+08
differential_evolution step 6: f(x)= 3.70942e+08
differential_evolution step 7: f(x)= 3.70942e+08
differential_evolution step 8: f(x)= 2.9013e+08
differential_evolution step 9: f(x)= 1.57621e+08
differential_evolution step 10: f(x)= 1.37518e+08
differential_evolution step 11: f(x)= 1.37518e+08
differential_evolution step 12: f(x)= 7.83223e+07
differential_evolution step 13: f(x)= 5.85926e+07
differential_evolution step 14: f(x)= 3.61706e+07
differential_evolution step 15: f(x)= 3.61706e+07
differential_evolution step 16: f(x)= 2.96351e+07
differential_evolution step 17: f(x)= 2.96351e+07
differential_evolution step 18: f(x)= 2.96351e+07
differential_evolution step 19: f(x)= 2.64387e+07
differential_evolution step 20: f(x)= 2.64387e+07
differential_evolution step 21: f(x)= 2.64387e+07
differential_evolution step 22: f(x)= 2.64387e+07
differential_evolution step 23: f(x)= 2.64387e+07
differential_evolution step 24: f(x)= 2.64387e+07
differential_evolution step 25: f(x)= 2.64387e+07
differential_evolution step 26: f(x)= 2.64387e+07
differential_evolution step 27: f(x)= 2.64387e+07
differential_evolution step 28: f(x)= 2.64387e+07
differential_evolution step 29: f(x)= 2.64387e+07
differential_evolution step 30: f(x)= 2.64387e+07
differential_evolution step 31: f(x)= 2.62888e+07
differential_evolution step 32: f(x)= 2.62888e+07
differential_evolution step 33: f(x)= 9.52121e+06
differential_evolution step 34: f(x)= 9.52121e+06
differential_evolution step 35: f(x)= 9.52121e+06
differential_evolution step 36: f(x)= 9.52121e+06
differential_evolution step 37: f(x)= 9.52121e+06
differential_evolution step 38: f(x)= 4.8353e+06
differential_evolution step 39: f(x)= 4.8353e+06
differential_evolution step 40: f(x)= 4.71205e+06
differential_evolution step 41: f(x)= 4.71205e+06
differential_evolution step 42: f(x)= 4.4658e+06
differential_evolution step 43: f(x)= 4.4658e+06
differential_evolution step 44: f(x)= 4.4658e+06
differential_evolution step 45: f(x)= 4.4658e+06
differential_evolution step 46: f(x)= 2.71229e+06
differential_evolution step 47: f(x)= 2.71229e+06
differential_evolution step 48: f(x)= 1.88774e+06
differential_evolution step 49: f(x)= 1.88774e+06
differential_evolution step 50: f(x)= 1.88774e+06
differential_evolution step 51: f(x)= 1.88774e+06
differential_evolution step 52: f(x)= 1.45677e+06
differential_evolution step 53: f(x)= 1.45677e+06
differential_evolution step 54: f(x)= 1.17613e+06
differential_evolution step 55: f(x)= 1.17613e+06
differential_evolution step 56: f(x)= 1.17613e+06
differential_evolution step 57: f(x)= 1.17613e+06
differential_evolution step 58: f(x)= 872574
differential_evolution step 59: f(x)= 872574
differential_evolution step 60: f(x)= 872574
differential_evolution step 61: f(x)= 872574
differential_evolution step 62: f(x)= 872574
differential_evolution step 63: f(x)= 872574
differential_evolution step 64: f(x)= 872574
differential_evolution step 65: f(x)= 872574
differential_evolution step 66: f(x)= 872574
differential_evolution step 67: f(x)= 872574
differential_evolution step 68: f(x)= 565355
differential_evolution step 69: f(x)= 565355
differential_evolution step 70: f(x)= 565355
differential_evolution step 71: f(x)= 565355
differential_evolution step 72: f(x)= 565355
differential_evolution step 73: f(x)= 443885
differential_evolution step 74: f(x)= 443885
differential_evolution step 75: f(x)= 443885
differential_evolution step 76: f(x)= 443885
differential_evolution step 77: f(x)= 443885
differential_evolution step 78: f(x)= 443885
differential_evolution step 79: f(x)= 443885
differential_evolution step 80: f(x)= 443885
differential_evolution step 81: f(x)= 443885
differential_evolution step 82: f(x)= 443885
differential_evolution step 83: f(x)= 443885
differential_evolution step 84: f(x)= 427550
differential_evolution step 85: f(x)= 427550
differential_evolution step 86: f(x)= 427550
differential_evolution step 87: f(x)= 427550
differential_evolution step 88: f(x)= 427550
differential_evolution step 89: f(x)= 390192
differential_evolution step 90: f(x)= 390192
differential_evolution step 91: f(x)= 390192
differential_evolution step 92: f(x)= 390192
differential_evolution step 93: f(x)= 390192
differential_evolution step 94: f(x)= 390192
differential_evolution step 95: f(x)= 390192
differential_evolution step 96: f(x)= 390192
differential_evolution step 97: f(x)= 390192
differential_evolution step 98: f(x)= 390192
differential_evolution step 99: f(x)= 390192
differential_evolution step 100: f(x)= 390192
differential_evolution step 101: f(x)= 390192
differential_evolution step 102: f(x)= 390192
differential_evolution step 103: f(x)= 390192
differential_evolution step 104: f(x)= 390192
differential_evolution step 105: f(x)= 390192
differential_evolution step 106: f(x)= 390192
differential_evolution step 107: f(x)= 390192
differential_evolution step 108: f(x)= 390192
differential_evolution step 109: f(x)= 390192
differential_evolution step 110: f(x)= 390192
differential_evolution step 111: f(x)= 389014
differential_evolution step 112: f(x)= 362791
differential_evolution step 113: f(x)= 362791
differential_evolution step 114: f(x)= 362791
differential_evolution step 115: f(x)= 362791
differential_evolution step 116: f(x)= 362791
differential_evolution step 117: f(x)= 362791
differential_evolution step 118: f(x)= 362791
differential_evolution step 119: f(x)= 362791
differential_evolution step 120: f(x)= 362791
differential_evolution step 121: f(x)= 362791
differential_evolution step 122: f(x)= 362791
differential_evolution step 123: f(x)= 362791
differential_evolution step 124: f(x)= 362791
differential_evolution step 125: f(x)= 362791
differential_evolution step 126: f(x)= 362791
differential_evolution step 127: f(x)= 362791
differential_evolution step 128: f(x)= 362791
differential_evolution step 129: f(x)= 362791
differential_evolution step 130: f(x)= 362791
differential_evolution step 131: f(x)= 362791
differential_evolution step 132: f(x)= 362791
differential_evolution step 133: f(x)= 362791
differential_evolution step 134: f(x)= 362791
differential_evolution step 135: f(x)= 362791
differential_evolution step 136: f(x)= 362791
differential_evolution step 137: f(x)= 362791
differential_evolution step 138: f(x)= 362791
differential_evolution step 139: f(x)= 362791
differential_evolution step 140: f(x)= 362791
differential_evolution step 141: f(x)= 362791
differential_evolution step 142: f(x)= 362791
differential_evolution step 143: f(x)= 362791
differential_evolution step 144: f(x)= 362791
differential_evolution step 145: f(x)= 362791
differential_evolution step 146: f(x)= 362791
differential_evolution step 147: f(x)= 362791
differential_evolution step 148: f(x)= 362791
differential_evolution step 149: f(x)= 362791
differential_evolution step 150: f(x)= 362791
differential_evolution step 151: f(x)= 357680
differential_evolution step 152: f(x)= 357680
differential_evolution step 153: f(x)= 357680
differential_evolution step 154: f(x)= 357680
differential_evolution step 155: f(x)= 357680
differential_evolution step 156: f(x)= 357680
differential_evolution step 157: f(x)= 357680
differential_evolution step 158: f(x)= 357680
differential_evolution step 159: f(x)= 357680
differential_evolution step 160: f(x)= 357680
differential_evolution step 161: f(x)= 357680
differential_evolution step 162: f(x)= 357680
differential_evolution step 163: f(x)= 357680
differential_evolution step 164: f(x)= 356139
differential_evolution step 165: f(x)= 356139
differential_evolution step 166: f(x)= 356139
differential_evolution step 167: f(x)= 356139
differential_evolution step 168: f(x)= 356139
differential_evolution step 169: f(x)= 356139
differential_evolution step 170: f(x)= 356139
differential_evolution step 171: f(x)= 356139
differential_evolution step 172: f(x)= 356139
differential_evolution step 173: f(x)= 356139
differential_evolution step 174: f(x)= 356139
differential_evolution step 175: f(x)= 356139
differential_evolution step 176: f(x)= 356139
differential_evolution step 177: f(x)= 356139
differential_evolution step 178: f(x)= 356139
differential_evolution step 179: f(x)= 356139
differential_evolution step 180: f(x)= 356139
differential_evolution step 181: f(x)= 356139
differential_evolution step 182: f(x)= 356139
differential_evolution step 183: f(x)= 356139
differential_evolution step 184: f(x)= 356139
differential_evolution step 185: f(x)= 356139
differential_evolution step 186: f(x)= 356139
differential_evolution step 187: f(x)= 356139
differential_evolution step 188: f(x)= 356139
differential_evolution step 189: f(x)= 356139
differential_evolution step 190: f(x)= 356139
differential_evolution step 191: f(x)= 356139
differential_evolution step 192: f(x)= 356139
differential_evolution step 193: f(x)= 356139
differential_evolution step 194: f(x)= 356139
differential_evolution step 195: f(x)= 356139
differential_evolution step 196: f(x)= 356139
differential_evolution step 197: f(x)= 356139
differential_evolution step 198: f(x)= 356139
differential_evolution step 199: f(x)= 356139
differential_evolution step 200: f(x)= 356139
differential_evolution step 201: f(x)= 356139
differential_evolution step 202: f(x)= 356139
differential_evolution step 203: f(x)= 356139
differential_evolution step 204: f(x)= 356139
differential_evolution step 205: f(x)= 356139
differential_evolution step 206: f(x)= 356139
differential_evolution step 207: f(x)= 356139
differential_evolution step 208: f(x)= 356139
differential_evolution step 209: f(x)= 356139
differential_evolution step 210: f(x)= 356139
differential_evolution step 211: f(x)= 356139
differential_evolution step 212: f(x)= 356139
differential_evolution step 213: f(x)= 356139
differential_evolution step 214: f(x)= 356139
differential_evolution step 215: f(x)= 356139
differential_evolution step 216: f(x)= 356139
differential_evolution step 217: f(x)= 356139
differential_evolution step 218: f(x)= 356139
differential_evolution step 219: f(x)= 356139
differential_evolution step 220: f(x)= 356139
differential_evolution step 221: f(x)= 356139
differential_evolution step 222: f(x)= 356139
differential_evolution step 223: f(x)= 356139
differential_evolution step 224: f(x)= 354374
differential_evolution step 225: f(x)= 354374
differential_evolution step 226: f(x)= 354374
differential_evolution step 227: f(x)= 352262
differential_evolution step 228: f(x)= 352262
differential_evolution step 229: f(x)= 352262
differential_evolution step 230: f(x)= 352262
differential_evolution step 231: f(x)= 352262
differential_evolution step 232: f(x)= 352262
differential_evolution step 233: f(x)= 352262
differential_evolution step 234: f(x)= 352262
differential_evolution step 235: f(x)= 352262
differential_evolution step 236: f(x)= 352262
differential_evolution step 237: f(x)= 352262
differential_evolution step 238: f(x)= 352262
differential_evolution step 239: f(x)= 352262
differential_evolution step 240: f(x)= 352262
differential_evolution step 241: f(x)= 352262
differential_evolution step 242: f(x)= 352262
differential_evolution step 243: f(x)= 352262
differential_evolution step 244: f(x)= 352262
differential_evolution step 245: f(x)= 352262
differential_evolution step 246: f(x)= 352262
differential_evolution step 247: f(x)= 352262
differential_evolution step 248: f(x)= 352262
differential_evolution step 249: f(x)= 352262
differential_evolution step 250: f(x)= 352262
differential_evolution step 251: f(x)= 352262
differential_evolution step 252: f(x)= 352262
differential_evolution step 253: f(x)= 352262
differential_evolution step 254: f(x)= 352262
differential_evolution step 255: f(x)= 352262
differential_evolution step 256: f(x)= 352262
differential_evolution step 257: f(x)= 352262
differential_evolution step 258: f(x)= 352262
differential_evolution step 259: f(x)= 352262
differential_evolution step 260: f(x)= 352262
differential_evolution step 261: f(x)= 352262
differential_evolution step 262: f(x)= 352262
differential_evolution step 263: f(x)= 352262
differential_evolution step 264: f(x)= 352262
differential_evolution step 265: f(x)= 352262
differential_evolution step 266: f(x)= 352262
differential_evolution step 267: f(x)= 352262
differential_evolution step 268: f(x)= 352262
differential_evolution step 269: f(x)= 352262
differential_evolution step 270: f(x)= 352262
differential_evolution step 271: f(x)= 352262
differential_evolution step 272: f(x)= 352262
differential_evolution step 273: f(x)= 352262
differential_evolution step 274: f(x)= 352262
differential_evolution step 275: f(x)= 352262
differential_evolution step 276: f(x)= 352262
differential_evolution step 277: f(x)= 352262
differential_evolution step 278: f(x)= 352262
differential_evolution step 279: f(x)= 352262
differential_evolution step 280: f(x)= 352262
differential_evolution step 281: f(x)= 352262
differential_evolution step 282: f(x)= 352262
differential_evolution step 283: f(x)= 352262
differential_evolution step 284: f(x)= 352262
differential_evolution step 285: f(x)= 352262
differential_evolution step 286: f(x)= 352262
differential_evolution step 287: f(x)= 352262
differential_evolution step 288: f(x)= 352262
differential_evolution step 289: f(x)= 352262
differential_evolution step 290: f(x)= 352262
differential_evolution step 291: f(x)= 352262
differential_evolution step 292: f(x)= 352262
differential_evolution step 293: f(x)= 352262
differential_evolution step 294: f(x)= 352262
differential_evolution step 295: f(x)= 352262
differential_evolution step 296: f(x)= 352262
differential_evolution step 297: f(x)= 352262
differential_evolution step 298: f(x)= 352262
differential_evolution step 299: f(x)= 352262
differential_evolution step 300: f(x)= 352262
differential_evolution step 301: f(x)= 352262
differential_evolution step 302: f(x)= 352262
differential_evolution step 303: f(x)= 352262
differential_evolution step 304: f(x)= 352262
differential_evolution step 305: f(x)= 352262
differential_evolution step 306: f(x)= 352262
differential_evolution step 307: f(x)= 352262
differential_evolution step 308: f(x)= 352262
differential_evolution step 309: f(x)= 352262
differential_evolution step 310: f(x)= 352262
differential_evolution step 311: f(x)= 348507
differential_evolution step 312: f(x)= 348507
differential_evolution step 313: f(x)= 348507
differential_evolution step 314: f(x)= 348507
differential_evolution step 315: f(x)= 348507
differential_evolution step 316: f(x)= 348507
differential_evolution step 317: f(x)= 348507
differential_evolution step 318: f(x)= 348507
differential_evolution step 319: f(x)= 348507
differential_evolution step 320: f(x)= 348507
differential_evolution step 321: f(x)= 348507
differential_evolution step 322: f(x)= 348507
differential_evolution step 323: f(x)= 348507
differential_evolution step 324: f(x)= 348507
differential_evolution step 325: f(x)= 348507
differential_evolution step 326: f(x)= 348507
differential_evolution step 327: f(x)= 348507
differential_evolution step 328: f(x)= 348507
differential_evolution step 329: f(x)= 348507
differential_evolution step 330: f(x)= 348507
differential_evolution step 331: f(x)= 348507
differential_evolution step 332: f(x)= 348507
differential_evolution step 333: f(x)= 348507
differential_evolution step 334: f(x)= 348507
differential_evolution step 335: f(x)= 348507
differential_evolution step 336: f(x)= 348507
differential_evolution step 337: f(x)= 348507
differential_evolution step 338: f(x)= 348507
differential_evolution step 339: f(x)= 348507
differential_evolution step 340: f(x)= 348507
differential_evolution step 341: f(x)= 348507
differential_evolution step 342: f(x)= 348507
differential_evolution step 343: f(x)= 348507
differential_evolution step 344: f(x)= 348507
differential_evolution step 345: f(x)= 348507
differential_evolution step 346: f(x)= 348507
differential_evolution step 347: f(x)= 348507
differential_evolution step 348: f(x)= 348507
differential_evolution step 349: f(x)= 348507
differential_evolution step 350: f(x)= 348507
differential_evolution step 351: f(x)= 348507
differential_evolution step 352: f(x)= 348507
differential_evolution step 353: f(x)= 348507
differential_evolution step 354: f(x)= 348507
differential_evolution step 355: f(x)= 348507
differential_evolution step 356: f(x)= 348507
differential_evolution step 357: f(x)= 348507
differential_evolution step 358: f(x)= 348507
differential_evolution step 359: f(x)= 348507
differential_evolution step 360: f(x)= 348507
differential_evolution step 361: f(x)= 348507
differential_evolution step 362: f(x)= 348133
differential_evolution step 363: f(x)= 348133
differential_evolution step 364: f(x)= 347380
differential_evolution step 365: f(x)= 347380
differential_evolution step 366: f(x)= 347380
differential_evolution step 367: f(x)= 347380
differential_evolution step 368: f(x)= 347380
differential_evolution step 369: f(x)= 347380
differential_evolution step 370: f(x)= 347380
differential_evolution step 371: f(x)= 347380
differential_evolution step 372: f(x)= 347380
differential_evolution step 373: f(x)= 347380
differential_evolution step 374: f(x)= 347380
differential_evolution step 375: f(x)= 347380
differential_evolution step 376: f(x)= 347380
differential_evolution step 377: f(x)= 347042
differential_evolution step 378: f(x)= 347042
differential_evolution step 379: f(x)= 347042
differential_evolution step 380: f(x)= 347042
differential_evolution step 381: f(x)= 347042
differential_evolution step 382: f(x)= 347042
differential_evolution step 383: f(x)= 347042
differential_evolution step 384: f(x)= 347042
differential_evolution step 385: f(x)= 347042
differential_evolution step 386: f(x)= 346882
differential_evolution step 387: f(x)= 346882
differential_evolution step 388: f(x)= 346574
differential_evolution step 389: f(x)= 346574
differential_evolution step 390: f(x)= 346574
differential_evolution step 391: f(x)= 346574
differential_evolution step 392: f(x)= 346574
differential_evolution step 393: f(x)= 346574
differential_evolution step 394: f(x)= 346574
differential_evolution step 395: f(x)= 346574
differential_evolution step 396: f(x)= 346574
differential_evolution step 397: f(x)= 346574
differential_evolution step 398: f(x)= 346574
differential_evolution step 399: f(x)= 346574
differential_evolution step 400: f(x)= 346057
differential_evolution step 401: f(x)= 346057
differential_evolution step 402: f(x)= 346057
differential_evolution step 403: f(x)= 346057
differential_evolution step 404: f(x)= 346057
differential_evolution step 405: f(x)= 346057
differential_evolution step 406: f(x)= 346057
differential_evolution step 407: f(x)= 346057
differential_evolution step 408: f(x)= 346057
differential_evolution step 409: f(x)= 346057
differential_evolution step 410: f(x)= 346057
differential_evolution step 411: f(x)= 346057
differential_evolution step 412: f(x)= 346057
differential_evolution step 413: f(x)= 346057
differential_evolution step 414: f(x)= 346057
differential_evolution step 415: f(x)= 346057
differential_evolution step 416: f(x)= 346057
differential_evolution step 417: f(x)= 346057
differential_evolution step 418: f(x)= 346057
differential_evolution step 419: f(x)= 346057
Polishing solution with 'trust-constr'
/opt/conda/lib/python3.10/site-packages/scipy/optimize/_hessian_update_strategy.py:182: UserWarning: delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.
  warn('delta_grad == 0.0. Check if the approximated '
/opt/conda/lib/python3.10/site-packages/scipy/optimize/_trustregion_constr/projections.py:181: UserWarning: Singular Jacobian matrix. Using SVD decomposition to perform the factorizations.
  warn('Singular Jacobian matrix. Using SVD decomposition to ' +
Optimization Progress:   0%|                                                                                | 0/100 [11:53<?, ?it/s]
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.46085777227768, 27.62296382401975, 2379.664065387215), 时间 34.746649229343745
残骸 2: 位置 (110.43948617850668, 27.71281409367176, 4225.584788157035), 时间 32.09265785041578
残骸 3: 位置 (110.36481250589891, 27.651724079135423, 3080.6392071334617), 时间 29.755537345963226
残骸 4: 位置 (110.53619370672743, 27.685708989540785, 2850.647555390456), 时间 32.90828596676371
```

```
import numpy as np
from pyproj import Transformer
from scipy.optimize import NonlinearConstraint, differential_evolution
from geographiclib.geodesic import Geodesic
from tqdm import tqdm
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


# 设定距离的最小阈值（例如：至少每个残骸之间相隔5000米）
min_distance = 5000

# 在全局作用域中定义 geod 对象
geod = Geodesic.WGS84

def objective_function(variables):
    num_debris = 4
    total_error = 0
    debris_positions = variables[:num_debris*3].reshape(num_debris, 3)
    debris_times = variables[num_debris*3:num_debris*4]

    # 检查所有残骸之间的地理距离并添加惩罚
    for i in range(num_debris):
        for j in range(i + 1, num_debris):
            lon1, lat1, _ = debris_positions[i]
            lon2, lat2, _ = debris_positions[j]
            distance = geod.Inverse(lat1, lon1, lat2, lon2)['s12']  # 使用正确的方法和参数顺序
            if distance < min_distance:
                total_error += (min_distance - distance)**2 * 1000  # 添加大的惩罚

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
                total_error += (predicted_time - arrival_time)**2

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
bounds = [(100, 120), (20, 40), (100, 10000)] * num_debris + [(0,min_time)] * num_debris  # XYZ坐标范围和时间范围

# 约束条件
constraints = [
    NonlinearConstraint(time_difference_constraint, 0, np.inf),
    NonlinearConstraint(time_prior_constraint, 0, np.inf),
    NonlinearConstraint(altitude_constraint, 0, np.inf)
]

# 使用差分进化算法求解
initial_guess = np.random.rand(num_debris * 3 + num_debris) * 1000  # 随机初始猜测


# 初始化进度条
progress_bar = tqdm(total=100, desc='Optimization Progress')

def update_progress(xk, convergence):
    # 更新进度条的当前状态
    progress_bar.update(1)  # 每次调用时进度条增加1
    return False  # 返回False以继续优化
# 执行差分进化算法
result = differential_evolution(
    objective_function,
    bounds=bounds,
    constraints=constraints,
    strategy='best1bin',
    maxiter=400,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback=None,
    disp=True,
    polish=True,
    init='random',
    atol=0
)

# 关闭进度条
progress_bar.close()

# 提取结果
if result.success:
    estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
    estimated_times = result.x[num_debris*3:num_debris*4]
    print("优化成功，找到可能的音爆源位置和时间")
    for i in range(num_debris):
        print(f"残骸 {i+1}: 位置 ({estimated_positions[i][0]}, {estimated_positions[i][1]}, {estimated_positions[i][2]}), 时间 {estimated_times[i]}")
else:
    print("优化失败：", result.message)

# 这段代码将设置所有必要的边界和约束，并运行差分进化算法以找到最优解，这里我们假设每个残骸的音爆发生位置和时间可以通过全局优化方法估计。

differential_evolution step 238: f(x)= 334109
differential_evolution step 239: f(x)= 334109
Polishing solution with 'trust-constr'
/opt/conda/lib/python3.10/site-packages/scipy/optimize/_hessian_update_strategy.py:182: UserWarning: delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.
  warn('delta_grad == 0.0. Check if the approximated '
/opt/conda/lib/python3.10/site-packages/scipy/optimize/_trustregion_constr/projections.py:181: UserWarning: Singular Jacobian matrix. Using SVD decomposition to perform the factorizations.
  warn('Singular Jacobian matrix. Using SVD decomposition to ' +
Optimization Progress:   0%|                                                                                | 0/100 [10:00<?, ?it/s]
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.47715395384103, 27.691914110851616, 1729.4179055728152), 时间 38.25128937240615
残骸 2: 位置 (110.4265507358363, 27.68907134460038, 543.0625042685751), 时间 34.78283551062484
残骸 3: 位置 (110.45457385861197, 27.651475952266924, 1654.5904500172296), 时间 39.044680435635904
残骸 4: 位置 (110.40388842395184, 27.648655120768865, 1359.334672807638), 时间 34.96287908946682


```

```
import numpy as np
from pyproj import Transformer
from scipy.optimize import NonlinearConstraint, differential_evolution
from geographiclib.geodesic import Geodesic
from tqdm import tqdm
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


# 设定距离的最小阈值（例如：至少每个残骸之间相隔5000米）
min_distance = 5000

# 在全局作用域中定义 geod 对象
geod = Geodesic.WGS84

def objective_function(variables):
    num_debris = 4
    total_error = 0
    debris_positions = variables[:num_debris*3].reshape(num_debris, 3)
    debris_times = variables[num_debris*3:num_debris*4]

    # 检查所有残骸之间的地理距离并添加惩罚
    for i in range(num_debris):
        for j in range(i + 1, num_debris):
            lon1, lat1, _ = debris_positions[i]
            lon2, lat2, _ = debris_positions[j]
            distance = geod.Inverse(lat1, lon1, lat2, lon2)['s12']  # 使用正确的方法和参数顺序
            if distance < min_distance:
                total_error += (min_distance - distance)**2 * 1000  # 添加大的惩罚

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
                total_error += (predicted_time - arrival_time)**2

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
bounds = [(100, 120), (20, 40), (100, 10000)] * num_debris + [(0,min_time)] * num_debris  # XYZ坐标范围和时间范围

# 约束条件
constraints = [
    NonlinearConstraint(time_difference_constraint, 0, np.inf),
    NonlinearConstraint(time_prior_constraint, 0, np.inf),
    NonlinearConstraint(altitude_constraint, 0, np.inf)
]

# 使用差分进化算法求解
initial_guess = np.random.rand(num_debris * 3 + num_debris) * 1000  # 随机初始猜测


# 初始化进度条
progress_bar = tqdm(total=100, desc='Optimization Progress')

def update_progress(xk, convergence):
    # 更新进度条的当前状态
    progress_bar.update(1)  # 每次调用时进度条增加1
    return False  # 返回False以继续优化
# 执行差分进化算法
result = differential_evolution(
    objective_function,
    bounds=bounds,
    constraints=constraints,
    strategy='best1bin',
    maxiter=400,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback=None,
    disp=True,
    polish=True,
    init='random',
    atol=0
)

# 关闭进度条
progress_bar.close()

# 提取结果
if result.success:
    estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
    estimated_times = result.x[num_debris*3:num_debris*4]
    print("优化成功，找到可能的音爆源位置和时间")
    for i in range(num_debris):
        print(f"残骸 {i+1}: 位置 ({estimated_positions[i][0]}, {estimated_positions[i][1]}, {estimated_positions[i][2]}), 时间 {estimated_times[i]}")
else:
    print("优化失败：", result.message)

# 这段代码将设置所有必要的边界和约束，并运行差分进化算法以找到最优解，这里我们假设每个残骸的音爆发生位置和时间可以通过全局优化方法估计。

differential_evolution step 283: f(x)= 333862
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.45664831665903, 27.637860981864236, 8943.766702193967), 时间 15.777785935877453
残骸 2: 位置 (110.51289363160257, 27.690041864390306, 9329.421998094193), 时间 14.951103668898572
残骸 3: 位置 (110.41187378064691, 27.674835558298394, 7353.222887049733), 时间 15.351367619294706
残骸 4: 位置 (110.45844422552285, 27.69731776338262, 9103.078488477482), 时间 12.353665925896763
```

Optimization Progress:   0%|                                                                                | 0/100 [14:49<?, ?it/s]
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.4978342352566, 27.685932790102083, 401.664076332012), 时间 40.253372706332584
残骸 2: 位置 (110.44780409461973, 27.678683604690125, 5611.108269317814), 时间 26.295012805910332
残骸 3: 位置 (110.41393565330762, 27.64512037077913, 8648.772762821558), 时间 16.04595735838143
残骸 4: 位置 (110.39856729481714, 27.689394607966666, 9922.738846138578), 时间 11.441592033046405
(base) root@dsw-82132-68df4f4dfb-wx2sq:/mnt/workspace# 





加入匈牙利改进

```python
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

# 假设的环境条件
temperature = 18  # 摄氏度
humidity = 50  # 相对湿度百分比
pressure = 1013  # 气压百帕

# 计算声速
speed_of_sound = 331.3 + 0.606 * temperature + 0.0124 * humidity - 0.004 * pressure
print("计算得到的声速为：", speed_of_sound)

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

# 设定距离的最小阈值（例如：至少每个残骸之间相隔10000米）
min_distance = 10000

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
            distance = geod.Inverse(lat1, lon1, lat2, lon2)['s12']  # 使用正确的方法和参数顺序
            if distance < min_distance:
                total_error += (min_distance - distance)**2 * 1000  # 添加大的惩罚

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
                total_error += (predicted_time - arrival_time)**2

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
bounds = [(100, 120), (20, 40), (2000, 10000)] * num_debris + [(0,min_time)] * num_debris  # XYZ坐标范围和时间范围

# 约束条件
constraints = [
    NonlinearConstraint(time_difference_constraint, 0, np.inf),
    NonlinearConstraint(time_prior_constraint, 0, np.inf),
    NonlinearConstraint(altitude_constraint, 0, np.inf)
]

# 使用差分进化算法求解
initial_guess = np.random.rand(num_debris * 3 + num_debris) * 1000  # 随机初始猜测


# 初始化进度条
progress_bar = tqdm(total=100, desc='Optimization Progress')

def update_progress(xk, convergence):
    # 更新进度条的当前状态
    progress_bar.update(1)  # 每次调用时进度条增加1
    return False  # 返回False以继续优化
# 执行差分进化算法
result = differential_evolution(
    objective_function,
    bounds=bounds,
    constraints=constraints,
    strategy='best1bin',
    maxiter=2000,
    popsize=15,
    tol=0.005,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback=None,
    disp=True,
    polish=True,
    init='random',
    atol=0,
    workers=-1  # 使用所有可用核心
)

# 关闭进度条
progress_bar.close()

# 提取结果
if result.success:
    estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
    estimated_times = result.x[num_debris*3:num_debris*4]
    print("优化成功，找到可能的音爆源位置和时间")
    for i in range(num_debris):
        print(f"残骸 {i+1}: 位置 ({estimated_positions[i][0]}, {estimated_positions[i][1]}, {estimated_positions[i][2]}), 时间 {estimated_times[i]}")
else:
    print("优化失败：", result.message)
    
    
    

# 这段代码将设置所有必要的边界和约束，并运行差分进化算法以找到最优解，这里我们假设每个残骸的音爆发生位置和时间可以通过全局优化方法估计。




'''
(base) root@dsw-82132-6d57fdb4c8-dtjdv:/mnt/workspace# /opt/conda/bin/python "/mnt/workspace/c31x copy 20.py"
Optimization Progress:   0%|                                                                                | 0/100 [00:00<?, ?it/s]/opt/conda/lib/python3.10/site-packages/scipy/optimize/_differentialevolution.py:387: UserWarning: differential_evolution: the 'workers' keyword has overridden updating='immediate' to updating='deferred'
  with DifferentialEvolutionSolver(func, bounds, args=args,
differential_evolution step 1: f(x)= 4.94099e+08
differential_evolution step 2: f(x)= 4.94099e+08
differential_evolution step 3: f(x)= 4.94099e+08
differential_evolution step 4: f(x)= 4.94099e+08
differential_evolution step 5: f(x)= 3.74443e+08
differential_evolution step 6: f(x)= 3.74443e+08
differential_evolution step 7: f(x)= 2.50279e+08
differential_evolution step 8: f(x)= 1.47613e+08
differential_evolution step 9: f(x)= 1.47613e+08
differential_evolution step 10: f(x)= 1.47613e+08
differential_evolution step 11: f(x)= 1.47613e+08
differential_evolution step 12: f(x)= 9.56752e+07
differential_evolution step 13: f(x)= 9.56752e+07
differential_evolution step 14: f(x)= 9.56752e+07
differential_evolution step 15: f(x)= 9.56752e+07
differential_evolution step 16: f(x)= 9.56752e+07
differential_evolution step 17: f(x)= 9.56752e+07
differential_evolution step 18: f(x)= 9.56752e+07
differential_evolution step 19: f(x)= 5.75074e+07
differential_evolution step 20: f(x)= 5.75074e+07
differential_evolution step 21: f(x)= 3.96147e+07
differential_evolution step 22: f(x)= 3.96147e+07
differential_evolution step 23: f(x)= 3.96147e+07
differential_evolution step 24: f(x)= 3.96147e+07
differential_evolution step 25: f(x)= 3.96147e+07
differential_evolution step 26: f(x)= 3.28365e+07
differential_evolution step 27: f(x)= 3.28365e+07
differential_evolution step 28: f(x)= 3.28365e+07
differential_evolution step 29: f(x)= 3.21678e+07
differential_evolution step 30: f(x)= 3.21678e+07
differential_evolution step 31: f(x)= 3.21678e+07
differential_evolution step 32: f(x)= 3.21678e+07
differential_evolution step 33: f(x)= 3.1654e+07
differential_evolution step 34: f(x)= 1.62501e+07
differential_evolution step 35: f(x)= 1.62501e+07
differential_evolution step 36: f(x)= 1.02968e+07
differential_evolution step 37: f(x)= 1.02968e+07
differential_evolution step 38: f(x)= 1.02968e+07
differential_evolution step 39: f(x)= 1.02968e+07
differential_evolution step 40: f(x)= 1.02968e+07
differential_evolution step 41: f(x)= 1.02968e+07
differential_evolution step 42: f(x)= 1.02968e+07
differential_evolution step 43: f(x)= 1.02968e+07
differential_evolution step 44: f(x)= 1.02968e+07
differential_evolution step 45: f(x)= 1.02968e+07
differential_evolution step 46: f(x)= 1.02968e+07
differential_evolution step 47: f(x)= 1.02968e+07
differential_evolution step 48: f(x)= 9.83157e+06
differential_evolution step 49: f(x)= 9.5297e+06
differential_evolution step 50: f(x)= 5.66768e+06
differential_evolution step 51: f(x)= 5.66768e+06
differential_evolution step 52: f(x)= 5.66768e+06
differential_evolution step 53: f(x)= 5.66768e+06
differential_evolution step 54: f(x)= 5.66768e+06
differential_evolution step 55: f(x)= 5.66768e+06
differential_evolution step 56: f(x)= 5.66768e+06
differential_evolution step 57: f(x)= 4.3961e+06
differential_evolution step 58: f(x)= 4.3961e+06
differential_evolution step 59: f(x)= 4.3961e+06
differential_evolution step 60: f(x)= 4.3961e+06
differential_evolution step 61: f(x)= 4.3961e+06
differential_evolution step 62: f(x)= 4.3961e+06
differential_evolution step 63: f(x)= 4.16942e+06
differential_evolution step 64: f(x)= 4.16942e+06
differential_evolution step 65: f(x)= 4.16942e+06
differential_evolution step 66: f(x)= 4.16942e+06
differential_evolution step 67: f(x)= 4.16942e+06
differential_evolution step 68: f(x)= 1.7189e+06
differential_evolution step 69: f(x)= 1.7189e+06
differential_evolution step 70: f(x)= 1.7189e+06
differential_evolution step 71: f(x)= 1.69562e+06
differential_evolution step 72: f(x)= 1.69562e+06
differential_evolution step 73: f(x)= 1.69562e+06
differential_evolution step 74: f(x)= 1.69562e+06
differential_evolution step 75: f(x)= 1.35554e+06
differential_evolution step 76: f(x)= 1.35554e+06
differential_evolution step 77: f(x)= 1.35554e+06
differential_evolution step 78: f(x)= 1.35554e+06
differential_evolution step 79: f(x)= 901118
differential_evolution step 80: f(x)= 901118
differential_evolution step 81: f(x)= 901118
differential_evolution step 82: f(x)= 901118
differential_evolution step 83: f(x)= 901118
differential_evolution step 84: f(x)= 901118
differential_evolution step 85: f(x)= 901118
differential_evolution step 86: f(x)= 901118
differential_evolution step 87: f(x)= 840738
differential_evolution step 88: f(x)= 840738
differential_evolution step 89: f(x)= 840738
differential_evolution step 90: f(x)= 695217
differential_evolution step 91: f(x)= 695217
differential_evolution step 92: f(x)= 695217
differential_evolution step 93: f(x)= 695217
differential_evolution step 94: f(x)= 695217
differential_evolution step 95: f(x)= 695217
differential_evolution step 96: f(x)= 695217
differential_evolution step 97: f(x)= 695217
differential_evolution step 98: f(x)= 695217
differential_evolution step 99: f(x)= 695217
differential_evolution step 100: f(x)= 695217
differential_evolution step 101: f(x)= 695217
differential_evolution step 102: f(x)= 656318
differential_evolution step 103: f(x)= 643623
differential_evolution step 104: f(x)= 643623
differential_evolution step 105: f(x)= 643623
differential_evolution step 106: f(x)= 528626
differential_evolution step 107: f(x)= 528626
differential_evolution step 108: f(x)= 528626
differential_evolution step 109: f(x)= 528626
differential_evolution step 110: f(x)= 528626
differential_evolution step 111: f(x)= 528626
differential_evolution step 112: f(x)= 528626
differential_evolution step 113: f(x)= 528626
differential_evolution step 114: f(x)= 528626
differential_evolution step 115: f(x)= 528626
differential_evolution step 116: f(x)= 528626
differential_evolution step 117: f(x)= 528626
differential_evolution step 118: f(x)= 518971
differential_evolution step 119: f(x)= 518971
differential_evolution step 120: f(x)= 518971
differential_evolution step 121: f(x)= 518971
differential_evolution step 122: f(x)= 518971
differential_evolution step 123: f(x)= 518971
differential_evolution step 124: f(x)= 498360
differential_evolution step 125: f(x)= 498360
differential_evolution step 126: f(x)= 498360
differential_evolution step 127: f(x)= 498360
differential_evolution step 128: f(x)= 498360
differential_evolution step 129: f(x)= 498360
differential_evolution step 130: f(x)= 498360
differential_evolution step 131: f(x)= 498360
differential_evolution step 132: f(x)= 498360
differential_evolution step 133: f(x)= 484351
differential_evolution step 134: f(x)= 481399
differential_evolution step 135: f(x)= 481399
differential_evolution step 136: f(x)= 481399
differential_evolution step 137: f(x)= 481399
differential_evolution step 138: f(x)= 481399
differential_evolution step 139: f(x)= 481399
differential_evolution step 140: f(x)= 481399
differential_evolution step 141: f(x)= 473634
differential_evolution step 142: f(x)= 465965
differential_evolution step 143: f(x)= 465965
differential_evolution step 144: f(x)= 465965
differential_evolution step 145: f(x)= 465965
differential_evolution step 146: f(x)= 465965
differential_evolution step 147: f(x)= 465965
differential_evolution step 148: f(x)= 465965
differential_evolution step 149: f(x)= 465965
differential_evolution step 150: f(x)= 465965
differential_evolution step 151: f(x)= 464604
differential_evolution step 152: f(x)= 464604
differential_evolution step 153: f(x)= 464604
differential_evolution step 154: f(x)= 464604
differential_evolution step 155: f(x)= 464604
differential_evolution step 156: f(x)= 464604
differential_evolution step 157: f(x)= 464604
differential_evolution step 158: f(x)= 460193
differential_evolution step 159: f(x)= 460193
differential_evolution step 160: f(x)= 460193
differential_evolution step 161: f(x)= 460193
differential_evolution step 162: f(x)= 455983
differential_evolution step 163: f(x)= 433777
differential_evolution step 164: f(x)= 433777
differential_evolution step 165: f(x)= 433777
differential_evolution step 166: f(x)= 433777
differential_evolution step 167: f(x)= 433777
differential_evolution step 168: f(x)= 433777
differential_evolution step 169: f(x)= 433777
differential_evolution step 170: f(x)= 433777
differential_evolution step 171: f(x)= 433777
differential_evolution step 172: f(x)= 433777
differential_evolution step 173: f(x)= 433777
differential_evolution step 174: f(x)= 433777
differential_evolution step 175: f(x)= 433777
differential_evolution step 176: f(x)= 433777
differential_evolution step 177: f(x)= 433777
differential_evolution step 178: f(x)= 433777
differential_evolution step 179: f(x)= 433777
differential_evolution step 180: f(x)= 433777
differential_evolution step 181: f(x)= 433777
differential_evolution step 182: f(x)= 433777
differential_evolution step 183: f(x)= 433777
differential_evolution step 184: f(x)= 426285
differential_evolution step 185: f(x)= 426285
differential_evolution step 186: f(x)= 426285
differential_evolution step 187: f(x)= 426285
differential_evolution step 188: f(x)= 426285
differential_evolution step 189: f(x)= 426285
differential_evolution step 190: f(x)= 426285
differential_evolution step 191: f(x)= 426285
differential_evolution step 192: f(x)= 426285
differential_evolution step 193: f(x)= 426285
differential_evolution step 194: f(x)= 426285
differential_evolution step 195: f(x)= 426285
differential_evolution step 196: f(x)= 426285
differential_evolution step 197: f(x)= 425581
differential_evolution step 198: f(x)= 425581
differential_evolution step 199: f(x)= 412650
differential_evolution step 200: f(x)= 412650
differential_evolution step 201: f(x)= 412650
differential_evolution step 202: f(x)= 412650
differential_evolution step 203: f(x)= 412650
differential_evolution step 204: f(x)= 412650
differential_evolution step 205: f(x)= 412650
differential_evolution step 206: f(x)= 412650
differential_evolution step 207: f(x)= 412650
differential_evolution step 208: f(x)= 412650
differential_evolution step 209: f(x)= 412650
differential_evolution step 210: f(x)= 412650
differential_evolution step 211: f(x)= 412650
differential_evolution step 212: f(x)= 412650
differential_evolution step 213: f(x)= 412650
differential_evolution step 214: f(x)= 412650
differential_evolution step 215: f(x)= 412650
differential_evolution step 216: f(x)= 402951
differential_evolution step 217: f(x)= 402951
differential_evolution step 218: f(x)= 402951
differential_evolution step 219: f(x)= 402951
differential_evolution step 220: f(x)= 402951
differential_evolution step 221: f(x)= 402951
differential_evolution step 222: f(x)= 402951
differential_evolution step 223: f(x)= 402951
differential_evolution step 224: f(x)= 402951
differential_evolution step 225: f(x)= 402951
differential_evolution step 226: f(x)= 402951
differential_evolution step 227: f(x)= 402951
differential_evolution step 228: f(x)= 402951
differential_evolution step 229: f(x)= 402951
differential_evolution step 230: f(x)= 402951
differential_evolution step 231: f(x)= 402951
differential_evolution step 232: f(x)= 402951
differential_evolution step 233: f(x)= 402951
differential_evolution step 234: f(x)= 402951
differential_evolution step 235: f(x)= 402951
differential_evolution step 236: f(x)= 402951
differential_evolution step 237: f(x)= 402951
differential_evolution step 238: f(x)= 402951
differential_evolution step 239: f(x)= 402951
differential_evolution step 240: f(x)= 402951
differential_evolution step 241: f(x)= 402951
differential_evolution step 242: f(x)= 402951
differential_evolution step 243: f(x)= 402951
differential_evolution step 244: f(x)= 402951
differential_evolution step 245: f(x)= 402951
differential_evolution step 246: f(x)= 402951
differential_evolution step 247: f(x)= 402951
differential_evolution step 248: f(x)= 402951
differential_evolution step 249: f(x)= 399632
differential_evolution step 250: f(x)= 399632
differential_evolution step 251: f(x)= 399632
differential_evolution step 252: f(x)= 399632
differential_evolution step 253: f(x)= 399632
differential_evolution step 254: f(x)= 399632
differential_evolution step 255: f(x)= 399632
differential_evolution step 256: f(x)= 399632
differential_evolution step 257: f(x)= 399632
differential_evolution step 258: f(x)= 399632
differential_evolution step 259: f(x)= 399632
differential_evolution step 260: f(x)= 399632
differential_evolution step 261: f(x)= 399632
differential_evolution step 262: f(x)= 399632
differential_evolution step 263: f(x)= 399632
differential_evolution step 264: f(x)= 399632
differential_evolution step 265: f(x)= 399632
differential_evolution step 266: f(x)= 399632
differential_evolution step 267: f(x)= 399632
differential_evolution step 268: f(x)= 399632
differential_evolution step 269: f(x)= 399632
differential_evolution step 270: f(x)= 399632
differential_evolution step 271: f(x)= 399632
differential_evolution step 272: f(x)= 399632
differential_evolution step 273: f(x)= 399632
differential_evolution step 274: f(x)= 399632
differential_evolution step 275: f(x)= 399632
differential_evolution step 276: f(x)= 399632
differential_evolution step 277: f(x)= 399632
differential_evolution step 278: f(x)= 399632
differential_evolution step 279: f(x)= 397720
differential_evolution step 280: f(x)= 397720
differential_evolution step 281: f(x)= 397720
differential_evolution step 282: f(x)= 397720
differential_evolution step 283: f(x)= 397720
differential_evolution step 284: f(x)= 397720
differential_evolution step 285: f(x)= 397720
differential_evolution step 286: f(x)= 397720
differential_evolution step 287: f(x)= 397720
differential_evolution step 288: f(x)= 397720
differential_evolution step 289: f(x)= 397720
differential_evolution step 290: f(x)= 397720
differential_evolution step 291: f(x)= 397720
differential_evolution step 292: f(x)= 397720
differential_evolution step 293: f(x)= 397720
differential_evolution step 294: f(x)= 397720
differential_evolution step 295: f(x)= 397720
differential_evolution step 296: f(x)= 397720
differential_evolution step 297: f(x)= 397720
differential_evolution step 298: f(x)= 397720
differential_evolution step 299: f(x)= 397720
differential_evolution step 300: f(x)= 397720
differential_evolution step 301: f(x)= 397720
differential_evolution step 302: f(x)= 397720
differential_evolution step 303: f(x)= 397720
differential_evolution step 304: f(x)= 397720
differential_evolution step 305: f(x)= 397720
differential_evolution step 306: f(x)= 397720
differential_evolution step 307: f(x)= 397720
differential_evolution step 308: f(x)= 397720
differential_evolution step 309: f(x)= 397720
differential_evolution step 310: f(x)= 397720
differential_evolution step 311: f(x)= 397720
differential_evolution step 312: f(x)= 397720
differential_evolution step 313: f(x)= 397720
differential_evolution step 314: f(x)= 397720
differential_evolution step 315: f(x)= 397720
differential_evolution step 316: f(x)= 397720
differential_evolution step 317: f(x)= 397720
differential_evolution step 318: f(x)= 397720
differential_evolution step 319: f(x)= 397720
differential_evolution step 320: f(x)= 397720
differential_evolution step 321: f(x)= 397720
differential_evolution step 322: f(x)= 397720
differential_evolution step 323: f(x)= 397720
differential_evolution step 324: f(x)= 397720
differential_evolution step 325: f(x)= 397720
differential_evolution step 326: f(x)= 397720
differential_evolution step 327: f(x)= 397720
differential_evolution step 328: f(x)= 397720
differential_evolution step 329: f(x)= 397720
differential_evolution step 330: f(x)= 397720
differential_evolution step 331: f(x)= 397720
differential_evolution step 332: f(x)= 397720
differential_evolution step 333: f(x)= 397720
differential_evolution step 334: f(x)= 397720
differential_evolution step 335: f(x)= 397613
differential_evolution step 336: f(x)= 397613
differential_evolution step 337: f(x)= 397613
differential_evolution step 338: f(x)= 397613
differential_evolution step 339: f(x)= 397613
differential_evolution step 340: f(x)= 397613
differential_evolution step 341: f(x)= 397613
differential_evolution step 342: f(x)= 397613
differential_evolution step 343: f(x)= 397613
differential_evolution step 344: f(x)= 397613
differential_evolution step 345: f(x)= 397613
differential_evolution step 346: f(x)= 397613
differential_evolution step 347: f(x)= 397613
differential_evolution step 348: f(x)= 397613
differential_evolution step 349: f(x)= 397613
differential_evolution step 350: f(x)= 397613
differential_evolution step 351: f(x)= 397613
differential_evolution step 352: f(x)= 397613
differential_evolution step 353: f(x)= 397613
differential_evolution step 354: f(x)= 397613
differential_evolution step 355: f(x)= 397613
differential_evolution step 356: f(x)= 397613
differential_evolution step 357: f(x)= 397613
differential_evolution step 358: f(x)= 397613
differential_evolution step 359: f(x)= 397613
differential_evolution step 360: f(x)= 397613
differential_evolution step 361: f(x)= 397613
differential_evolution step 362: f(x)= 397613
differential_evolution step 363: f(x)= 397613
differential_evolution step 364: f(x)= 397613
differential_evolution step 365: f(x)= 396902
differential_evolution step 366: f(x)= 396902
differential_evolution step 367: f(x)= 396902
differential_evolution step 368: f(x)= 396902
differential_evolution step 369: f(x)= 396902
differential_evolution step 370: f(x)= 396902
differential_evolution step 371: f(x)= 396902
differential_evolution step 372: f(x)= 396902
differential_evolution step 373: f(x)= 396902
differential_evolution step 374: f(x)= 396902
differential_evolution step 375: f(x)= 396902
differential_evolution step 376: f(x)= 396902
differential_evolution step 377: f(x)= 396902
differential_evolution step 378: f(x)= 396902
differential_evolution step 379: f(x)= 396902
differential_evolution step 380: f(x)= 396902
differential_evolution step 381: f(x)= 393850
differential_evolution step 382: f(x)= 393850
differential_evolution step 383: f(x)= 393850
differential_evolution step 384: f(x)= 393850
differential_evolution step 385: f(x)= 393850
differential_evolution step 386: f(x)= 393850
differential_evolution step 387: f(x)= 393850
differential_evolution step 388: f(x)= 393850
differential_evolution step 389: f(x)= 393850
differential_evolution step 390: f(x)= 393850
differential_evolution step 391: f(x)= 393850
differential_evolution step 392: f(x)= 393850
differential_evolution step 393: f(x)= 393850
differential_evolution step 394: f(x)= 393850
differential_evolution step 395: f(x)= 393850
differential_evolution step 396: f(x)= 393850
differential_evolution step 397: f(x)= 393850
differential_evolution step 398: f(x)= 393850
differential_evolution step 399: f(x)= 393850
differential_evolution step 400: f(x)= 393850
differential_evolution step 401: f(x)= 393850
differential_evolution step 402: f(x)= 393850
differential_evolution step 403: f(x)= 393850
differential_evolution step 404: f(x)= 393850
differential_evolution step 405: f(x)= 393850
differential_evolution step 406: f(x)= 393850
differential_evolution step 407: f(x)= 393850
differential_evolution step 408: f(x)= 393850
differential_evolution step 409: f(x)= 393850
differential_evolution step 410: f(x)= 393850
differential_evolution step 411: f(x)= 393850
differential_evolution step 412: f(x)= 393850
differential_evolution step 413: f(x)= 393850
differential_evolution step 414: f(x)= 393850
differential_evolution step 415: f(x)= 393850
differential_evolution step 416: f(x)= 393850
differential_evolution step 417: f(x)= 393850
differential_evolution step 418: f(x)= 393850
differential_evolution step 419: f(x)= 393850
differential_evolution step 420: f(x)= 393374
differential_evolution step 421: f(x)= 393374
differential_evolution step 422: f(x)= 393374
differential_evolution step 423: f(x)= 393374
differential_evolution step 424: f(x)= 393374
differential_evolution step 425: f(x)= 393374
differential_evolution step 426: f(x)= 393374
differential_evolution step 427: f(x)= 393374
differential_evolution step 428: f(x)= 393374
differential_evolution step 429: f(x)= 393374
differential_evolution step 430: f(x)= 393374
differential_evolution step 431: f(x)= 393374
differential_evolution step 432: f(x)= 393374
differential_evolution step 433: f(x)= 393374
differential_evolution step 434: f(x)= 393374
differential_evolution step 435: f(x)= 393374
differential_evolution step 436: f(x)= 393374
differential_evolution step 437: f(x)= 393374
differential_evolution step 438: f(x)= 393374
differential_evolution step 439: f(x)= 393374
differential_evolution step 440: f(x)= 393374
differential_evolution step 441: f(x)= 393374
differential_evolution step 442: f(x)= 393374
differential_evolution step 443: f(x)= 393374
differential_evolution step 444: f(x)= 393374
differential_evolution step 445: f(x)= 393374
differential_evolution step 446: f(x)= 393374
differential_evolution step 447: f(x)= 393374
differential_evolution step 448: f(x)= 393374
differential_evolution step 449: f(x)= 393374
differential_evolution step 450: f(x)= 393374
differential_evolution step 451: f(x)= 393374
differential_evolution step 452: f(x)= 393374
differential_evolution step 453: f(x)= 392517
differential_evolution step 454: f(x)= 392517
differential_evolution step 455: f(x)= 392517
differential_evolution step 456: f(x)= 392517
differential_evolution step 457: f(x)= 392517
differential_evolution step 458: f(x)= 392517
differential_evolution step 459: f(x)= 392517
differential_evolution step 460: f(x)= 392517
differential_evolution step 461: f(x)= 392517
differential_evolution step 462: f(x)= 392517
differential_evolution step 463: f(x)= 392517
differential_evolution step 464: f(x)= 392517
differential_evolution step 465: f(x)= 392517
differential_evolution step 466: f(x)= 392517
differential_evolution step 467: f(x)= 392517
differential_evolution step 468: f(x)= 392517
differential_evolution step 469: f(x)= 392517
differential_evolution step 470: f(x)= 392517
differential_evolution step 471: f(x)= 392517
differential_evolution step 472: f(x)= 392002
differential_evolution step 473: f(x)= 391222
differential_evolution step 474: f(x)= 391222
differential_evolution step 475: f(x)= 391222
differential_evolution step 476: f(x)= 391222
differential_evolution step 477: f(x)= 391222
differential_evolution step 478: f(x)= 391222
differential_evolution step 479: f(x)= 391222
differential_evolution step 480: f(x)= 391222
differential_evolution step 481: f(x)= 391222
differential_evolution step 482: f(x)= 391222
differential_evolution step 483: f(x)= 391222
differential_evolution step 484: f(x)= 391222
differential_evolution step 485: f(x)= 391222
differential_evolution step 486: f(x)= 391222
differential_evolution step 487: f(x)= 391222
differential_evolution step 488: f(x)= 391222
differential_evolution step 489: f(x)= 391222
differential_evolution step 490: f(x)= 391222
differential_evolution step 491: f(x)= 390860
differential_evolution step 492: f(x)= 390860
differential_evolution step 493: f(x)= 390808
differential_evolution step 494: f(x)= 390808
differential_evolution step 495: f(x)= 390808
differential_evolution step 496: f(x)= 390808
differential_evolution step 497: f(x)= 390808
differential_evolution step 498: f(x)= 390808
differential_evolution step 499: f(x)= 390808
differential_evolution step 500: f(x)= 390808
differential_evolution step 501: f(x)= 390808
differential_evolution step 502: f(x)= 390808
differential_evolution step 503: f(x)= 390808
differential_evolution step 504: f(x)= 390808
differential_evolution step 505: f(x)= 390808
differential_evolution step 506: f(x)= 390808
differential_evolution step 507: f(x)= 390808
differential_evolution step 508: f(x)= 389850
differential_evolution step 509: f(x)= 389850
differential_evolution step 510: f(x)= 389850
differential_evolution step 511: f(x)= 389850
differential_evolution step 512: f(x)= 389850
differential_evolution step 513: f(x)= 389475
differential_evolution step 514: f(x)= 389475
differential_evolution step 515: f(x)= 389475
differential_evolution step 516: f(x)= 389106
differential_evolution step 517: f(x)= 389106
differential_evolution step 518: f(x)= 389106
differential_evolution step 519: f(x)= 389106
differential_evolution step 520: f(x)= 389106
differential_evolution step 521: f(x)= 389106
differential_evolution step 522: f(x)= 389106
differential_evolution step 523: f(x)= 389106
differential_evolution step 524: f(x)= 389106
differential_evolution step 525: f(x)= 389106
differential_evolution step 526: f(x)= 389106
differential_evolution step 527: f(x)= 389106
differential_evolution step 528: f(x)= 389106
differential_evolution step 529: f(x)= 389106
differential_evolution step 530: f(x)= 389106
differential_evolution step 531: f(x)= 389106
differential_evolution step 532: f(x)= 389106
differential_evolution step 533: f(x)= 389106
differential_evolution step 534: f(x)= 389106
differential_evolution step 535: f(x)= 389106
differential_evolution step 536: f(x)= 388483
differential_evolution step 537: f(x)= 388483
differential_evolution step 538: f(x)= 388483
differential_evolution step 539: f(x)= 388483
differential_evolution step 540: f(x)= 388483
differential_evolution step 541: f(x)= 388483
differential_evolution step 542: f(x)= 388483
differential_evolution step 543: f(x)= 388483
differential_evolution step 544: f(x)= 388483
differential_evolution step 545: f(x)= 388483
differential_evolution step 546: f(x)= 388483
differential_evolution step 547: f(x)= 388483
differential_evolution step 548: f(x)= 388483
differential_evolution step 549: f(x)= 388483
differential_evolution step 550: f(x)= 388483
differential_evolution step 551: f(x)= 388042
differential_evolution step 552: f(x)= 388042
differential_evolution step 553: f(x)= 388042
differential_evolution step 554: f(x)= 388042
differential_evolution step 555: f(x)= 388042
differential_evolution step 556: f(x)= 388042
differential_evolution step 557: f(x)= 388042
differential_evolution step 558: f(x)= 388042
differential_evolution step 559: f(x)= 388042
differential_evolution step 560: f(x)= 388042
differential_evolution step 561: f(x)= 388042
differential_evolution step 562: f(x)= 388042
differential_evolution step 563: f(x)= 388042
differential_evolution step 564: f(x)= 388042
differential_evolution step 565: f(x)= 388042
differential_evolution step 566: f(x)= 388042
differential_evolution step 567: f(x)= 388042
differential_evolution step 568: f(x)= 387317
differential_evolution step 569: f(x)= 387317
differential_evolution step 570: f(x)= 387317
differential_evolution step 571: f(x)= 387317
differential_evolution step 572: f(x)= 387317
differential_evolution step 573: f(x)= 387317
differential_evolution step 574: f(x)= 387317
differential_evolution step 575: f(x)= 387317
differential_evolution step 576: f(x)= 387317
differential_evolution step 577: f(x)= 387317
differential_evolution step 578: f(x)= 387317
differential_evolution step 579: f(x)= 387317
differential_evolution step 580: f(x)= 387317
differential_evolution step 581: f(x)= 387317
differential_evolution step 582: f(x)= 387317
differential_evolution step 583: f(x)= 387317
differential_evolution step 584: f(x)= 387317
differential_evolution step 585: f(x)= 387317
differential_evolution step 586: f(x)= 387317
differential_evolution step 587: f(x)= 387317
differential_evolution step 588: f(x)= 387208
differential_evolution step 589: f(x)= 387208
differential_evolution step 590: f(x)= 387208
differential_evolution step 591: f(x)= 387208
differential_evolution step 592: f(x)= 387208
differential_evolution step 593: f(x)= 387208
differential_evolution step 594: f(x)= 387208
differential_evolution step 595: f(x)= 386846
differential_evolution step 596: f(x)= 386846
differential_evolution step 597: f(x)= 386846
differential_evolution step 598: f(x)= 386846
differential_evolution step 599: f(x)= 386846
differential_evolution step 600: f(x)= 386846
differential_evolution step 601: f(x)= 386846
differential_evolution step 602: f(x)= 386846
differential_evolution step 603: f(x)= 386846
differential_evolution step 604: f(x)= 386846
differential_evolution step 605: f(x)= 386846
differential_evolution step 606: f(x)= 386846
differential_evolution step 607: f(x)= 386846
differential_evolution step 608: f(x)= 386846
differential_evolution step 609: f(x)= 386846
differential_evolution step 610: f(x)= 386846
differential_evolution step 611: f(x)= 386846
differential_evolution step 612: f(x)= 386846
differential_evolution step 613: f(x)= 386846
differential_evolution step 614: f(x)= 386846
differential_evolution step 615: f(x)= 386846
differential_evolution step 616: f(x)= 386846
differential_evolution step 617: f(x)= 386846
differential_evolution step 618: f(x)= 386846
differential_evolution step 619: f(x)= 386846
differential_evolution step 620: f(x)= 386846
differential_evolution step 621: f(x)= 386846
differential_evolution step 622: f(x)= 386846
differential_evolution step 623: f(x)= 386846
differential_evolution step 624: f(x)= 386846
differential_evolution step 625: f(x)= 386846
differential_evolution step 626: f(x)= 386846
differential_evolution step 627: f(x)= 386846
differential_evolution step 628: f(x)= 386846
differential_evolution step 629: f(x)= 386846
differential_evolution step 630: f(x)= 386846
differential_evolution step 631: f(x)= 386846
differential_evolution step 632: f(x)= 386846
differential_evolution step 633: f(x)= 386706
differential_evolution step 634: f(x)= 386706
differential_evolution step 635: f(x)= 386706
differential_evolution step 636: f(x)= 386706
differential_evolution step 637: f(x)= 386706
differential_evolution step 638: f(x)= 386706
differential_evolution step 639: f(x)= 386611
differential_evolution step 640: f(x)= 386611
differential_evolution step 641: f(x)= 386611
differential_evolution step 642: f(x)= 386611
differential_evolution step 643: f(x)= 386611
differential_evolution step 644: f(x)= 386611
differential_evolution step 645: f(x)= 386611
differential_evolution step 646: f(x)= 386611
differential_evolution step 647: f(x)= 386611
differential_evolution step 648: f(x)= 386611
differential_evolution step 649: f(x)= 386611
Polishing solution with 'trust-constr'
/opt/conda/lib/python3.10/site-packages/scipy/optimize/_hessian_update_strategy.py:182: UserWarning: delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.
  warn('delta_grad == 0.0. Check if the approximated '
Optimization Progress:   0%|                                                                                | 0/100 [06:20<?, ?it/s]
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.36850795953924, 27.66027156553201, 2270.6915182523435), 时间 35.52129918286551
残骸 2: 位置 (110.54038724482075, 27.677137334933494, 2576.9662773374994), 时间 38.58707422465326
残骸 3: 位置 (110.46017840463895, 27.62181594584061, 2096.997412297818), 时间 39.76810629376607
残骸 4: 位置 (110.44863591116885, 27.715572848759617, 2217.7627949433463), 时间 38.964939391858906
(base) root@dsw-82132-6d57fdb4c8-dtjdv:/mnt/workspace# 
'''
```

```python
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
min_distance = 5000

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
            distance = geod.Inverse(lat1, lon1, lat2, lon2)['s12']  # 使用正确的方法和参数顺序
            if distance < min_distance:
                total_error += (min_distance - distance)**2 * 1000  # 添加大的惩罚

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
                total_error += (predicted_time - arrival_time)**2

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
bounds = [(100, 120), (20, 40), (100, 10000)] * num_debris + [(0,min_time)] * num_debris  # XYZ坐标范围和时间范围

# 约束条件
constraints = [
    NonlinearConstraint(time_difference_constraint, 0, np.inf),
    NonlinearConstraint(time_prior_constraint, 0, np.inf),
    NonlinearConstraint(altitude_constraint, 0, np.inf)
]

# 使用差分进化算法求解
initial_guess = np.random.rand(num_debris * 3 + num_debris) * 1000  # 随机初始猜测


# 初始化进度条
progress_bar = tqdm(total=100, desc='Optimization Progress')

def update_progress(xk, convergence):
    # 更新进度条的当前状态
    progress_bar.update(1)  # 每次调用时进度条增加1
    return False  # 返回False以继续优化
# 执行差分进化算法
result = differential_evolution(
    objective_function,
    bounds=bounds,
    constraints=constraints,
    strategy='best1bin',
    maxiter=800,
    popsize=20,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback=None,
    disp=True,
    polish=True,
    init='random',
    atol=0,
    workers=-1  # 使用所有可用核心
)

# 关闭进度条
progress_bar.close()

# 提取结果
if result.success:
    estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
    estimated_times = result.x[num_debris*3:num_debris*4]
    print("优化成功，找到可能的音爆源位置和时间")
    for i in range(num_debris):
        print(f"残骸 {i+1}: 位置 ({estimated_positions[i][0]}, {estimated_positions[i][1]}, {estimated_positions[i][2]}), 时间 {estimated_times[i]}")
else:
    print("优化失败：", result.message)

# 这段代码将设置所有必要的边界和约束，并运行差分进化算法以找到最优解，这里我们假设每个残骸的音爆发生位置和时间可以通过全局优化方法估计。


优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.40097355139117, 27.696932349979967, 1415.2121005714089), 时间 41.61271809256746
残骸 2: 位置 (110.47021173427983, 27.645570263656644, 2327.0439116536245), 时间 40.50839343672057
残骸 3: 位置 (110.46432772895506, 27.69136589386902, 1195.7676474616487), 时间 39.93693555886976
残骸 4: 位置 (110.41589732695459, 27.645418296502125, 1989.7749048631008), 时间 42.55644249090943
(base) root@dsw-82132-6d57fdb4c8-dtjdv:/mnt/workspace# 
```

残骸 1: 位置 (110.36850795953924, 27.66027156553201, 2270.6915182523435), 时间 35.52129918286551
残骸 2: 位置 (110.54038724482075, 27.677137334933494, 2576.9662773374994), 时间 38.58707422465326
残骸 3: 位置 (110.46017840463895, 27.62181594584061, 2096.997412297818), 时间 39.76810629376607
残骸 4: 位置 (110.44863591116885, 27.715572848759617, 2217.7627949433463), 时间 38.964939391858906











优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.44960051041252, 27.62276603378855, 5209.765663071043), 时间 29.79945258654971
残骸 2: 位置 (110.51526473269276, 27.66738928421613, 5041.948243834048), 时间 30.561378011827244
残骸 3: 位置 (110.39104174612847, 27.672709210318395, 4791.150794468814), 时间 29.495153531241286
残骸 4: 位置 (110.45565614181513, 27.716343757306255, 4991.853174327189), 时间 26.733761851952053
整体重合度: 0.63





在提供的Python代码中，使用了一系列的数学模型和优化算法来估计音爆发生的位置和时间。这个数学模型的关键部分涉及地理坐标转换、声波传播时间的计算、以及使用差分进化算法（DEA）进行全局优化。下面详细讲解这些数学公式和模型。

### 数学模型概述

1. **坐标转换**：
   为了方便计算地球表面上两点之间的距离，将地理坐标（经纬度和高程）转换为笛卡尔坐标系。转换公式使用了地理坐标系统到笛卡尔坐标系统的标准转换（EPSG:4326到EPSG:4978）。

2. **声波传播时间计算**：
   声波从残骸到达监测站的时间计算公式为：
   \[
   t_{arrival} = t_{emission} + \frac{d}{v}
   \]
   其中，\( t_{arrival} \) 是声波到达监测站的时间，\( t_{emission} \) 是声波发射时间，\( d \) 是从发射源到监测站的距离，\( v \) 是声速。

   距离 \( d \) 的计算考虑了地面距离和高度差，具体公式为：
   \[
   d = \sqrt{d_{surface}^2 + h^2}
   \]
   其中，\( d_{surface} \) 是地表距离，\( h \) 是高度差。

3. **全局优化算法 - 差分进化算法**：
   使用差分进化算法估计残骸的位置和时间。差分进化算法是一种强大的全局优化算法，适用于处理有约束的优化问题。该算法通过迭代改进候选解，直至找到最优解或满足停止条件。

### 数学公式解析

- **地表距离的计算**：
  地表距离 \( d_{surface} \) 通过地理库计算两点之间的测地线距离。测地线是指在地球表面两点之间的最短路径，其计算公式使用了Vincenty公式或者Karney算法（由`geographiclib`库实现）。

- **声速的影响**：
  声速 \( v \) 受温度影响的公式为：
  \[
  v = 331.3 + 0.606 \times \text{temperature}
  \]
  其中，温度以摄氏度计，此处设定为18°C。

### 优化问题的约束

在使用差分进化算法解决优化问题时，设置了以下约束条件：

1. **时间差约束**：所有残骸的声波发射时间差不应超过5秒。
2. **时间先行约束**：任何残骸的发射时间都不应晚于其到达最近监测站的时间。
3. **高程约束**：所有残骸的高程应为非负值。

### 整体重合度求解过程

整体重合度的计算涉及比较每个监测站测得的声波到达时间和理论上计算得到的声波到达时间。重合度的具体计算公式为：
\[
$$
\text{Overlap Score} = 1 - \frac{\text{max\_diff}}{\max(\text{station\_times})}
$$
\]
其中，`max_diff` 是预计到达时间和实际到达时间之差的最大值，`station_times` 是实际到达时间的集合。

最后，所有监测站的重

合度平均值给出了模型的整体性能指标。此性能指标反映了模型预测准确度的高低，越接近1表示预测越准确。






数学模型




```python
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
                total_error += min_distance - distance  # 使用指数函数增加惩罚

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
                total_error += time_diff**2  # 使用指数函数处理时间差异

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
        total_error += (0.95 - total_overlap_score) * 1000000000  # 当重合度低于0.8时，增加惩罚

    # 更新重合度数据和计数器
    if overlap_data['last_score'] is not None and total_overlap_score == overlap_data['last_score']:
        overlap_data['stagnant_count'] += 1
    else:
        overlap_data['stagnant_count'] = 0  # 重置计数器
    overlap_data['last_score'] = total_overlap_score

    # 当重合度40次不变时，增加极大的惩罚
    if overlap_data['stagnant_count'] >= 20:
        total_error += (0.95 - total_overlap_score) * 10000000  # 增加极大的惩罚值

    # 根据重合度的提升给予奖励
    if overlap_data['previous_total_score'] is not None:
        score_increase = total_overlap_score - overlap_data['previous_total_score']
        if score_increase >= 0.01:
            total_error -= score_increase * 50000  # 奖励值
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
bounds = [(110, 111), (27, 28), (4693.82, 88847.59)] * num_debris + [(0, min_time)] * num_debris  # XYZ坐标范围和时间范围

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
        differential_evolution_args['popsize'] = 100  # 增加种群大小以增加搜索范围
        differential_evolution_args['tol'] = 0.000005  # 减小容忍度以提高精度
    elif total_overlap_score < 0.3:
        differential_evolution_args['mutation'] = (1.1, 2.3)
        differential_evolution_args['popsize'] = 80
        differential_evolution_args['tol'] = 0.000001
    elif total_overlap_score < 0.4:
        differential_evolution_args['mutation'] = (1.3, 2.5)
        differential_evolution_args['popsize'] = 70
        differential_evolution_args['tol'] = 0.00005
    elif total_overlap_score < 0.5:
        differential_evolution_args['mutation'] = (1.5, 2.7)
        differential_evolution_args['popsize'] = 60
        differential_evolution_args['tol'] = 0.00001
    elif total_overlap_score < 0.6:
        differential_evolution_args['mutation'] = (1.7, 2.9)
        differential_evolution_args['popsize'] = 50
        differential_evolution_args['tol'] = 0.0005
    elif total_overlap_score < 0.7:
        differential_evolution_args['mutation'] = (1.9, 3.1)
        differential_evolution_args['popsize'] = 40
        differential_evolution_args['tol'] = 0.0001
    elif total_overlap_score < 0.8:
        differential_evolution_args['mutation'] = (2.1, 3.3)
        differential_evolution_args['popsize'] = 30
        differential_evolution_args['tol'] = 0.005
    elif total_overlap_score < 0.9:
        differential_evolution_args['mutation'] = (2.3, 3.5)
        differential_evolution_args['popsize'] = 20
        differential_evolution_args['tol'] = 0.001
    else:
        differential_evolution_args['mutation'] = (2.5, 3.7)
        differential_evolution_args['popsize'] = 15  # 减少种群大小以减少计算量
        differential_evolution_args['tol'] = 0.001  # 增加容忍度以快速收敛
        # 如果重合度在5轮内一直不变化，增加变异率和种群大小


    differential_evolution_args['recombination'] = 0.90 + 0.05 * (1 - total_overlap_score)
    # 如果达到目标重合度，停止优化
    if total_overlap_score >= 0.95:
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







differential_evolution step 634: f(x)= 1.17367e+08
当前整体重合度: 0.83
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.48892066036824, 27.797832892353675, 4713.227713079905), 时间 0.0069846270530433685
残骸 2: 位置 (110.48030359529935, 27.71160486208126, 23282.908991299388), 时间 4.692970652993509
残骸 3: 位置 (110.49940590928114, 27.61016652934353, 40485.21679209831), 时间 3.542240073680908
残骸 4: 位置 (110.52366806137488, 27.40508996474332, 42493.590924993805), 时间 4.874298828830433
监测站 A 的输出重合度: 0.61
监测站 B 的输出重合度: 0.76
监测站 C 的输出重合度: 0.98
监测站 D 的输出重合度: 0.97
监测站 E 的输出重合度: 0.86
监测站 F 的输出重合度: 0.90
监测站 G 的输出重合度: 0.75
整体重合度: 0.83
```

```
每一次迭代，记录重合度最高的解集，根据重合度最高的解集改变缩小bounds，使得越来越快
from scipy.optimize import minimize
import numpy as np
from geographiclib.geodesic import Geodesic
from random import uniform
overlap_scores = {}
# 假设声速为340 m/s
speed_of_sound = 340
stations = {
    'A': {'coords': (110.241, 27.204, 824), 'times': [100.767, 164.229, 214.850, 270.065]},
    'B': {'coords': (110.783, 27.456, 727), 'times': [92.453, 112.220, 169.362, 196.583]},
    'C': {'coords': (110.762, 27.785, 742), 'times': [75.560, 110.696, 156.936, 188.020]},
    'D': {'coords': (110.251, 28.025, 850), 'times': [94.653, 141.409, 196.517, 258.985]},
    'E': {'coords': (110.524, 27.617, 786), 'times': [78.600, 86.216, 118.443, 126.669]},
    'F': {'coords': (110.467, 28.081, 678), 'times': [67.274, 166.270, 175.482, 266.871]},
    'G': {'coords': (110.047, 27.521, 575), 'times': [103.738, 163.024, 206.789, 210.306]}
}

# 残骸位置和时间
debris_data = [
    {'coords': (110.48699620279264, 27.796804496998107, 4696.420451521799), 'time': 3.022811386609002e-05},
    {'coords': (110.45859694217505, 27.699417573775168, 21702.079942876386), 'time': 2.66763528170297},
    {'coords': (110.45788601098397, 27.548249786156934, 32416.730880741397), 'time': 4.146458911688192},
    {'coords': (110.52598097563148, 27.405970132874376, 42602.40344742619), 'time': 4.976365608831523}
]

num_debris = 4
# 根据stations中的最大和最小时间来设置时间范围
min_time = min(min(station['times']) for station in stations.values())
max_time = max(max(station['times']) for station in stations.values())
bounds = [(109, 112), (26, 29), (4693.82, 88847.59)] * num_debris + [(0, min_time)] * num_debris  # XYZ坐标范围和时间范围
predicted_times = {}
def objective_function(params):
    # 更新残骸数据
    for i in range(num_debris):
        debris_data[i]['coords'] = (params[3*i], params[3*i+1], params[3*i+2])
        debris_data[i]['time'] = params[3*num_debris + i]

    # 重新计算预计到达时间和重合度
    for key, station in stations.items():
        station_coord = station['coords']
        station_predicted_times = []
        for debris in debris_data:
            debris_coord = debris['coords']
            debris_time = debris['time']
            
            geod = Geodesic.WGS84
            result = geod.Inverse(station_coord[1], station_coord[0], debris_coord[1], debris_coord[0])
            surface_distance = result['s12']
            height_difference = np.abs(station_coord[2] - debris_coord[2])
            total_distance = np.sqrt(surface_distance**2 + height_difference**2)
            
            time_to_travel = total_distance / speed_of_sound + debris_time
            station_predicted_times.append(time_to_travel)
            
        predicted_times[key] = station_predicted_times

    # 计算输出重合度
    for key in stations.keys():
        actual = stations[key]['times']  # 使用stations字典中的时间作为实际到达时间
        predicted = predicted_times[key]  # 使用计算得到的预计到达时间
        differences = [abs(a - p) for a, p in zip(actual, predicted)]
        max_diff = max(differences)
        overlap_score = 1 - (max_diff / (max_time - min_time))
        overlap_scores[key] = overlap_score

    # 输出重合度
    for key, score in overlap_scores.items():
        print(f"监测站 {key} 的输出重合度: {score:.2f}")

    # 计算整体重合度
    total_overlap_score = sum(overlap_scores.values()) / len(overlap_scores)
    # 输出整体重合度
    print(f"整体重合度: {total_overlap_score:.2f}")

    # 激励整体重合度接近于1
    return -total_overlap_score  # Minimize negative to maximize overlap

# 初始参数
initial_params = [np.random.uniform(low, high) for bound in bounds for low, high in [bound]]
# 优化过程
result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')

# 输出优化后的残骸位置和时间
if result.success:
    optimized_params = result.x
    print("优化成功！")
    print("优化后的残骸位置和时间:")
    for i in range(num_debris):
        print(f"残骸 {i+1}: 坐标 ({optimized_params[3*i]:.6f}, {optimized_params[3*i+1]:.6f}, {optimized_params[3*i+2]:.2f}), 时间 {optimized_params[3*num_debris + i]:.6f} 秒")
    # 计算整体重合度
    total_overlap_score = sum(overlap_scores.values()) / len(overlap_scores)
    # 输出整体重合度
    print(f"整体重合度: {total_overlap_score:.2f}")
else:
    print("优化失败：", result.message)
```

```python
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

def calculate_total_overlap_score(estimated_positions, estimated_times):
    # 计算每个监测站到每个残骸的距离和时间，并存储预计到达时间
    predicted_times = {}
    for key, station in stations.items():
        station_coord = station['coords']
        station_predicted_times = []
        for i, (debris_coord, debris_time) in enumerate(zip(estimated_positions, estimated_times)):
            geod = Geodesic.WGS84
            result = geod.Inverse(station_coord[1], station_coord[0], debris_coord[1], debris_coord[0])
            surface_distance = result['s12']
            height_difference = np.abs(station_coord[2] - debris_coord[2])
            total_distance = np.sqrt(surface_distance**2 + height_difference**2)
            
            time_to_travel = total_distance / speed_of_sound + debris_time
            station_predicted_times.append(time_to_travel)
        
        predicted_times[key] = station_predicted_times

    # 计算输出重合度
    overlap_scores = {}
    for key in stations.keys():
        actual = stations[key]['times']  # 使用stations字典中的时间作为实际到达时间
        predicted = predicted_times[key]  # 使用计算得到的预计到达时间
        differences = [abs(a - p) for a, p in zip(actual, predicted)]
        max_diff = max(differences)
        overlap_score = 1 - (max_diff / max(actual))
        overlap_scores[key] = overlap_score

    # 计算整体重合度
    total_overlap_score = sum(overlap_scores.values()) / len(overlap_scores)
    return total_overlap_score

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
        total_error += (0.95 - total_overlap_score) * 100000000  # 当重合度低于0.8时，增加惩罚

    # 更新重合度数据和计数器
    if overlap_data['last_score'] is not None and total_overlap_score == overlap_data['last_score']:
        overlap_data['stagnant_count'] += 1
    else:
        overlap_data['stagnant_count'] = 0  # 重置计数器
    overlap_data['last_score'] = total_overlap_score

    # 当重合度40次不变时，增加极大的惩罚
    if overlap_data['stagnant_count'] >= 20:
        total_error += (0.95 - total_overlap_score) * 100000  # 增加极大的惩罚值

    # 根据重合度的提升给予奖励    if overlap_data['previous_total_score'] is not None:
        score_change = total_overlap_score - overlap_data['previous_total_score']
        if score_change >= 0.01:
            total_error -= score_change * (abs(score_change)**2)  # 奖励值随增长数值增加
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
bounds = [(110, 111), (27, 28), (4693.82, 88847.59)] * num_debris + [(0, min_time)] * num_debris  # XYZ坐标范围和时间范围

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
        differential_evolution_args['tol'] = 0.0000005  # 减小容忍度以提高精度
    elif total_overlap_score < 0.3:
        differential_evolution_args['mutation'] = (1.1, 2.3)
        differential_evolution_args['popsize'] = 30
        differential_evolution_args['tol'] = 0.0000001
    elif total_overlap_score < 0.4:
        differential_evolution_args['mutation'] = (1.3, 2.5)
        differential_evolution_args['popsize'] = 40
        differential_evolution_args['tol'] = 0.000005
    elif total_overlap_score < 0.5:
        differential_evolution_args['mutation'] = (1.5, 2.7)
        differential_evolution_args['popsize'] = 50
        differential_evolution_args['tol'] = 0.000001
    elif total_overlap_score < 0.6:
        differential_evolution_args['mutation'] = (1.7, 2.9)
        differential_evolution_args['popsize'] = 60
        differential_evolution_args['tol'] = 0.00005
    elif total_overlap_score < 0.7:
        differential_evolution_args['mutation'] = (1.9, 3.1)
        differential_evolution_args['popsize'] = 70
        differential_evolution_args['tol'] = 0.00001
    elif total_overlap_score < 0.8:
        differential_evolution_args['mutation'] = (2.1, 3.3)
        differential_evolution_args['popsize'] = 80
        differential_evolution_args['tol'] = 0.0005
    elif total_overlap_score < 0.9:
        differential_evolution_args['mutation'] = (2.3, 3.5)
        differential_evolution_args['popsize'] = 90
        differential_evolution_args['tol'] = 0.0001
    else:
        differential_evolution_args['mutation'] = (2.5, 3.7)
        differential_evolution_args['popsize'] = 100  # 减少种群大小以减少计算量
        differential_evolution_args['tol'] = 0.0001  # 增加容忍度以快速收敛
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


differential_evolution step 734: f(x)= 1.2615e+07
当前整体重合度: 0.83
Polishing solution with 'trust-constr'
/opt/conda/lib/python3.10/site-packages/scipy/optimize/_hessian_update_strategy.py:182: UserWarning: delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.
  warn('delta_grad == 0.0. Check if the approximated '
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.48979999816933, 27.797750971937464, 4723.342768475911), 时间 0.0165980163577283
残骸 2: 位置 (110.46123761966241, 27.68792012795248, 21024.1015515943), 时间 3.042742797940989
残骸 3: 位置 (110.45297856826406, 27.54923116635573, 33164.608578547195), 时间 2.8634576535612957
残骸 4: 位置 (110.52284030213089, 27.405504696348554, 42527.613103734344), 时间 4.903649086066846
监测站 A 的输出重合度: 0.61
监测站 B 的输出重合度: 0.76
监测站 C 的输出重合度: 0.98
监测站 D 的输出重合度: 0.97
监测站 E 的输出重合度: 0.86
监测站 F 的输出重合度: 0.90
监测站 G 的输出重合度: 0.75
整体重合度: 0.83
```

# **题目3从这里开始看**

### 方法说明

在本研究中，我们首先定义声速为340米/秒，此值为标准大气条件下的常用估计。监测站的地理位置由经度、纬度和海拔高度构成，并记录了声波到达各站的时间。

对于每个监测站，我们首先计算声波直达该站的距离。利用`Geodesic`库，计算出当前监测站点到所有其他监测站点的水平距离。根据这些距离，我们可以通过以下公式计算出声源可能的高度，其中使用了毕达哥拉斯定理来求解声波到达距离与水平距离之间的高度差：

\[ \text{Height} = \sqrt{(\text{Sound Distance})^2 - (\text{Horizontal Distance})^2} + \text{Reference Elevation} \]

在本分析中，每个监测站均重复执行此计算流程，收集所有基于时间测量的高度估计值。

通过上述方法，我们为每个监测站估算了一个可能的高度范围。通过汇总所有监测站的数据，我们确定了整个监测区域的最低和最高可能高度。这些高度范围的信息对于理解地区的地理和环境特性极为重要，尤其适用于灾害监测和环境保护领域中的应用。

通过声波测量方法估计多个监测站的高度范围。通过测量声波在不同监测站之间传播所需的时间，并结合每个站点的地理坐标，我们能够估算出声源的可能高度。根据声速（340米/秒）和各站点间的测量时间，以下是各监测站的可能高度范围，及其对环境特征理解的贡献。

监测站 A 的可能高度范围: 13234.76m - 83213.02m
监测站 B 的可能高度范围: 4693.82m - 59900.47m
监测站 C 的可能高度范围: 9840.94m - 57257.24m
监测站 D 的可能高度范围: 24049.17m - 85908.24m
监测站 E 的可能高度范围: 26200.44m - 31680.18m
监测站 F 的可能高度范围: 6656.08m - 88847.59m
监测站 G 的可能高度范围: 27995.08m - 60074.09m
监测站 A 的可能高度范围: 13234.76m - 83213.02m
监测站 B 的可能高度范围: 4693.82m - 59900.47m
监测站 C 的可能高度范围: 9840.94m - 57257.24m
监测站 D 的可能高度范围: 24049.17m - 85908.24m
监测站 E 的可能高度范围: 26200.44m - 31680.18m
监测站 F 的可能高度范围: 6656.08m - 88847.59m
监测站 G 的可能高度范围: 27995.08m - 60074.09m
所有监测站的综合可能高度范围: 4693.82m - 88847.59m
![监测站可能高度范围](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\监测站可能高度范围.png)



这组数据为后续我们估计高度提供参考

### 时间预测与实际时间的对比分析

本研究采用声波传播时间计算方法，估算不同监测站接收到来自某些残骸的声波的时间。通过这些计算，我们进一步评估预计到达时间与实际到达时间之间的重合度，用于分析模型的准确性和效率。

#### 时间预测与实际时间对比

对于每个监测站，根据残骸的地理位置和监测站的位置信息，我们计算了声波从残骸到达各个监测站的预计到达时间。这些预计时间随后与监测站记录的实际到达时间进行了对比。此步骤是至关重要的，因为它直接影响模型预测的可靠性和有效性评估。

#### 重合度的计算与应用

重合度是评估预计到达时间与实际到达时间一致性的关键指标。我们定义重合度为：
\[
\text{Overlap Score} = 1 - \frac{\text{Maximum Time Difference}}{\text{Maximum Actual Time}}
\]
其中，“最大时间差异”指的是实际到达时间与预测到达时间之间的最大偏差，而“实际时间中的最大值”指的是记录的到达时间中的最大值。重合度越接近1，表明预测结果与实际观测数据的一致性越高，反映出模型的准确性较好。相反，重合度较低则表明存在较大的预测偏差。

### 重合度计算过程

1. **数据准备**
   - \( A_i \)：第 \( i \) 个事件的实际到达时间。
   - \( P_i \)：第 \( i \) 个事件的预计到达时间。

### 2. **计算时间差异**

   对于每个事件，计算实际到达时间和预计到达时间之间的绝对差异：
   \[
   D_i = |A_i - P_i|
   \]
   其中 \( D_i \) 是第 \( i \) 个事件的时间差异。

### 3. **确定最大时间差异**

   找出所有事件时间差异中的最大值，即：
   \[
   D_{\text{max}} = \max(D_i)
   \]
   \( D_{\text{max}} \) 表示所有事件中最大的时间差异。

### 4. **找出实际时间的最大值**

   在所有实际到达时间中找出最大值：
   \[
   A_{\text{max}} = \max(A_i)
   \]
   \( A_{\text{max}} \) 是实际到达时间中的最大值，用于归一化时间差异。

### 5. **计算重合度**

   最后，使用以下公式计算重合度：
   \[
   O = 1 - \frac{D_{\text{max}}}{A_{\text{max}}}
   \]
   其中 \( O \) 代表重合度，是一个介于0和1之间的值。\( O \) 的值越接近1，表示预测结果与实际观测越一致，准确性越高。

### 解释

通过计算 \( O \)，我们能够量化模型在预测监测站接收到声波时间方面的准确性。高重合度（\( O \) 接近1）表明模型预测与实际数据高度吻合，而低重合度指示预测存在较大偏差，可能需要模型调整或数据质量审核。这个指标对于评估和优化声波传播模型非常有用。






为了更直观地展示模型预测结果与实际数据的匹配程度，我们利用`matplotlib`库生成了重合度的柱状图以及每个监测站预计到达时间的图表。这种可视化方法不仅便于观察每个监测站的表现，也有助于整体评估模型的效果，为进一步的模型优化和应用提供了依据。

通过对声波传播时间的计算及其与实际到达时间的比较，我们能够评估声波传播模型在实际应用中的准确性。重合度的计算进一步提供了一个量化指标，用以衡量预测与实际观测数据之间的一致性。这种方法的有效性不仅体现在提供准确的预测，还在于通过数据可视化使得结果易于理解和分析，为未来的研究方向和应用提供了可能的改进和调整方案。

![image-20240510235618131](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510235618131.png)

![image-20240510235614589](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510235614589.png)

# 使用2题数学模型

### 数学模型的详细描述

本文介绍一个用于估算音爆源位置及其时间的数学模型，模型基于监测站记录的音爆到达时间。通过转换地理坐标到笛卡尔坐标，并利用声速和地理距离的计算，结合全局优化算法和匈牙利算法，我们能够估计音爆的发生位置和时间。

#### 步骤 1: 坐标转换

将地理坐标（经度、纬度、高程）转换为笛卡尔坐标系（X, Y, Z），使用的转换基于WGS84地球椭球模型。坐标转换的数学表达式如下：

- **地球椭球模型**：
  赤道半径 \(a = 6378137.0\) 米，扁率 \(f = 1/298.257223563\)。

- **坐标转换公式**：
  \[
  X = (N + h) \cos(\phi) \cos(\lambda), \quad Y = (N + h) \cos(\phi) \sin(\lambda), \quad Z = \left(\frac{b^2}{a^2} N + h\right) \sin(\phi)
  \]
  其中，\(\phi\) 和 \(\lambda\) 是纬度和经度的弧度值，\(h\) 是高程，\(N\) 是由椭球体的赤道半径 \(a\) 和扁率 \(f\) 计算得到的曲率半径。

#### 步骤 2: 时间和距离的计算

音爆从源点到各监测站的理论到达时间是通过声速和两点之间的地表距离计算得出。声速随温度变化的关系及距离的计算方法如下：

- **声速公式**：
  \[
  v = 331.3 + 0.606 \cdot \text{temperature}
  \]
  其中，\(v\) 是声速，\(\text{temperature}\) 是环境温度。

- **地理距离计算**（使用GeographicLib库的Vincenty公式）：
  \[
  d = \text{Geodesic.WGS84.Inverse}(\phi_1, \lambda_1, \phi_2, \lambda_2)['s12']
  \]

- **到达时间计算**：
  \[
  t = t_0 + \frac{d}{v}
  \]
  其中 \(t_0\) 是音爆发生时间。

#### 步骤 3: 优化目标函数

目标是最小化观测到的音爆到达时间与计算得出的理论到达时间之间的差异。通过构建成本矩阵并应用匈牙利算法进行优化。

- **成本矩阵**：
  \[
  C_{ij} = (t_{ij}^\text{predicted} - \bar{t}_{j}^\text{observed})^2
  \]
  其中 \(C_{ij}\) 是成本矩阵中第 \(i\) 个音爆源位置和第 \(j\) 个监测站之间的成本。

- **匈牙利算法**：
  使用匈牙利算法找到成本矩阵的最小总成本匹配。这个算法保证了每个监测站都匹配到一个唯一的音爆源位置，从而最小化总误差。

#### 步骤 4: 约束和全局优化

为了找到最佳的音爆源位置和时间，模型需要满足多个物理和逻辑约束，并使用差分进化算法进行全局优化。

- **时间约束**：
  \[
  \max(t_{0i}) - \min(t_{0i}) \leq 5 \text{ 秒}


  \]

- **先于约束**：
  \[
  \min_{j}(\min(t_j^\text{observed})) \geq \max(t_{0i})
  \]

- **高程约束**：
  \[
  \min(z_i) \geq 0
  \]

- **全局优化算法**：
  使用差分进化算法解决约束优化问题，找到最小化目标函数的参数集。

通过这些步骤，该数学模型能够有效地预测音爆源的位置和时间，为监测和研究提供了一个强有力的工具。这种方法的准确性依赖于声速的正确估计、坐标转换的精度以及优化算法的效率。





### 数学模型详解：多源音爆定位问题

本研究旨在解决从不同火箭残骸发出的音爆定位问题。通过结合地理坐标转换、信号传播时间计算以及多信号匹配优化技术，该模型能够有效地将监测站接收到的音爆信号正确归因于特定的残骸。下面详细介绍该数学模型的核心组成部分。

#### 1. 坐标转换

模型的第一步是将所有监测站和音爆源的地理坐标（经度、纬度及高程）转换为三维笛卡尔坐标系统（X, Y, Z）。该转换基于以下地球椭球模型参数进行：

- 赤道半径 \( a = 6378137.0 \) 米
- 扁率 \( f = 1/298.257223563 \)
- 第一偏心率平方 \( e^2 = f \cdot (2-f) \)

给定纬度 \( \phi \)，经度 \( \lambda \)，以及高程 \( h \)，转换公式为：

\[
$$
\begin{align*}
N &= \frac{a}{\sqrt{1 - e^2 \sin^2(\phi)}} \\
X &= (N + h) \cos(\phi) \cos(\lambda) \\
Y &= (N + h) \cos(\phi) \sin(\lambda) \\
Z &= ((1 - e^2) N + h) \sin(\phi)
\end{align*}
$$
\]

这些公式确保所有后续计算在空间上一致和精确。

#### 2. 到达时间的计算

音爆从每个残骸到监测站的理论到达时间根据以下公式计算：

\[
$$
t_{arrival} = t_{emission} + \frac{d}{v}
$$
\]

其中 \( d \) 是残骸到监测站的直线距离，\( v \) 是声速（约为 340 m/s），\( t_{emission} \) 是残骸发生音爆的时间。距离 \( d \) 的计算公式为：

\[
$$
d = \sqrt{(X_1 - X_2)^2 + (Y_1 - Y_2)^2 + (Z_1 - Z_2)^2}
$$
\]

#### 3. 多信号匹配优化

使用改进的匈牙利算法进行信号匹配，以解决每个监测站可能接收到来自不同残骸的多个信号的问题。构建成本矩阵 \( C \)，其中 \( C[i, j] \) 表示第 \( i \) 个监测站接收的信号与第 \( j \) 个残骸的理论到达时间的差的平方。然后，该算法寻找成本最小的残骸到监测站的匹配方式。

#### 4. 遗传算法优化

遗传算法用于进一步优化残骸的预测位置和时间，以最小化所有监测站观测到的时间与模型预测时间之间的差异。算法通过模拟自然选择过程中的遗传变异和适应性选择来迭代地优化候选解。

适应度函数定义为：

\[
$$
\text{minimize} \quad \sum_{i=1}^{n} \left( t_{\text{predicted},i} - t_{\text{observed},i} \right)^2 - \lambda \cdot \text{number of devices}
$$
\]

其中，\( t_{\text{predicted},i} \) 是根据模型预测的到达时间，\( t_{\text{observed},i} \) 是实际观测到的到达时间，而 \( \lambda \) 是正则化参数，用以平衡设备数量对总体匹配质量的影响。

5. #### 优化目标

   #### 目标函数

   目标函数 \( E \) 旨在最小化残骸位置和时间预测的总误差。误差由两部分组成：地理距离的惩罚和时间差的平方误差。

   1. **地理距离的惩罚**:
      \[
      E_{\text{distance}} = \sum_{i < j} \max(0, (d_{\text{min}} - d_{ij})^2 \times 1000)
      \]
      其中 \( d_{ij} \) 是残骸 \( i \) 和 \( j \) 之间的地理距离，\( d_{\text{min}} \) 是允许的最小距离。

   2. **时间差的平方误差**:
      \[
      E_{\text{time}} = \sum_{i, j} (t_{\text{pred},ij} - t_{\text{arr},ij})^2
      \]
      其中 \( t_{\text{pred},ij} \) 是从残骸 \( i \) 到监测站 \( j \) 的预测到达时间，\( t_{\text{arr},ij} \) 是实际到达时间。

   总误差 \( E \) 为:
   \[
   E = E_{\text{distance}} + E_{\text{time}}
   \]

   #### 约束条件

   1. **时间差约束**:
      \[
      \text{time\_difference\_constraint} = 5 - (\max(\text{debris\_times}) - \min(\text{debris\_times}))
      \]
      确保所有残骸的到达时间差不超过5秒。

   2. **时间先验约束**:
      \[
      \text{time\_prior\_constraint} = \min(\text{min\_station\_times}) - \max(\text{debris\_times})
      \]
      确保音爆发生时间小于任何监测站记录的最早抵达时间。

   3. **高程约束**:
      \[
      \text{altitude\_constraint} = \min(\text{debris\_altitudes})
      \]
      确保所有残骸的高程非负。

   这些数学表达式和约束条件共同定义了一个优化问题，目标是在满足约束的条件下，通过调整残骸的位置和时间来最小化总误差 \( E \)。

   \[
   $$
   E = \sum_{i < j} \max(0, (d_{\text{min}} - d_{ij})^2 \times 1000) + \sum_{i, j} (t_{\text{pred},ij} - t_{\text{arr},ij})^2
   $$
   约束

   $$
   \text{time\_difference\_constraint} = 5 - (\max(\text{debris\_times}) - \min(\text{debris\_times}))
   $$

   $$
   \text{time\_prior\_constraint} = \min(\text{min\_station\_times}) - \max(\text{debris\_times})
   $$

   $$
   \text{altitude\_constraint} = \min(\text{debris\_altitudes})
   $$

监测站 A 位置: (110.241, 27.204, 824)
  残骸 1: 距离 52127.77 米, 预计到达时间 188.84 秒
  残骸 2: 距离 60232.21 米, 预计到达时间 215.74 秒
  残骸 3: 距离 51136.34 米, 预计到达时间 190.17 秒
  残骸 4: 距离 60305.01 米, 预计到达时间 216.33 秒

监测站 B 位置: (110.783, 27.456, 727)
  残骸 1: 距离 46802.32 米, 预计到达时间 173.18 秒
  残骸 2: 距离 34320.87 米, 预计到达时间 139.53 秒
  残骸 3: 距离 36828.04 米, 预计到达时间 148.09 秒
  残骸 4: 距离 43811.89 米, 预计到达时间 167.82 秒

监测站 C 位置: (110.762, 27.785, 742)
  残骸 1: 距离 41219.77 米, 预计到达时间 156.76 秒
  残骸 2: 距离 24975.02 米, 预计到达时间 112.04 秒
  残骸 3: 距离 34856.95 米, 预计到达时间 142.29 秒
  残骸 4: 距离 31871.59 米, 预计到达时间 132.70 秒

监测站 D 位置: (110.251, 28.025, 850)
  残骸 1: 距离 42066.86 米, 预计到达时间 159.25 秒
  残骸 2: 距离 47973.67 米, 预计到达时间 179.69 秒
  残骸 3: 距离 49219.14 米, 预计到达时间 184.53 秒
  残骸 4: 距离 39452.21 米, 预计到达时间 155.00 秒

监测站 E 位置: (110.524, 27.617, 786)
  残骸 1: 距离 16145.35 米, 预计到达时间 83.01 秒
  残骸 2: 距离 7087.46 米, 预计到达时间 59.43 秒
  残骸 3: 距离 6456.59 米, 预计到达时间 58.76 秒
  残骸 4: 距离 13291.22 米, 预计到达时间 78.06 秒

监测站 F 位置: (110.467, 28.081, 678)
  残骸 1: 距离 47648.86 米, 预计到达时间 175.66 秒
  残骸 2: 距离 45374.48 米, 预计到达时间 172.04 秒
  残骸 3: 距离 50909.64 米, 预计到达时间 189.50 秒
  残骸 4: 距离 40565.41 米, 预计到达时间 158.27 秒

监测站 G 位置: (110.047, 27.521, 575)
  残骸 1: 距离 35336.40 米, 预计到达时间 139.45 秒
  残骸 2: 距离 51729.32 米, 预计到达时间 190.73 秒
  残骸 3: 距离 42329.83 米, 预计到达时间 164.27 秒
  残骸 4: 距离 45157.82 米, 预计到达时间 171.78 秒

监测站 A 的输出重合度: 0.67
监测站 B 的输出重合度: 0.59
监测站 C 的输出重合度: 0.57
监测站 D 的输出重合度: 0.60
监测站 E 的输出重合度: 0.53
监测站 F 的输出重合度: 0.59
监测站 G 的输出重合度: 0.80

![image-20240510235809427](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510235809427.png)

![image-20240511000051912](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240511000051912.png)

![image-20240510235455753](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510235455753.png)

![image-20240510235834371](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510235834371.png)



# 将重合度加入优化，改进模型。



### 数学模型和约束条件的详细解释及公式

#### 1. 坐标转换公式

将地理坐标转换为笛卡尔坐标的数学表达式为：
\[
X = (N(\phi) + h) \cdot \cos(\phi) \cdot \cos(\lambda)
\]
\[
Y = (N(\phi) + h) \cdot \cos(\phi) \cdot \sin(\lambda)
\]
\[
Z = \left(\left(1 - e^2\right) \cdot N(\phi) + h\right) \cdot \sin(\phi)
\]
其中：

- \( N(\phi) = \frac{a}{\sqrt{1 - e^2 \cdot \sin^2(\phi)}} \)
- \( a \) 是赤道半径
- \( e^2 \) 是地球椭球的第一偏心率平方
- \( \phi, \lambda, h \) 分别是纬度、经度和高程

#### 2. 信号传播时间计算

音爆从每个残骸到监测站的传播时间由以下公式给出：
\[
t_{arrival} = t_{emission} + \frac{d}{v}
\]
\[
d = \sqrt{(X_1 - X_2)^2 + (Y_1 - Y_2)^2 + (Z_1 - Z_2)^2}
\]
\[
v = 331.3 + 0.606 \cdot T
\]
其中：

- \( d \) 是残骸到监测站的直线距离
- \( v \) 是声速，依赖于环境温度 \( T \)
- \( t_{emission} \) 是残骸发生音爆的时间

#### 3. 成本矩阵和匹配算法

成本矩阵 \( C \) 用于线性分配算法，其元素定义为：
\[
C_{ij} = (t_{predicted, i} - t_{actual, j})^2
\]
其中 \( t_{predicted, i} \) 是第 \( i \) 个残骸到第 \( j \) 个监测站的预测到达时间，\( t_{actual, j} \) 是实际到达时间。

#### 4. 约束条件

优化问题包含以下约束条件：

- **时间差异约束**：
  \[
  \max(t_{emission}) - \min(t_{emission}) \leq 5 \text{ 秒}
  \]
  确保所有残骸的音爆发生时间差不超过5秒。

- **时间优先性约束**：
  \[
  \min(t_{emission}) \leq \min(t_{arrival})
  \]
  确保所有残骸的音爆发生时间早于任何监测站的接收时间。

- **高程约束**：
  \[
  \min(Z) \geq 0
  \]
  保证所有残骸的高程为非负值。

#### 5. 优化目标

### 1. 目标函数定义

目标函数的目的是最小化总误差 \( E \)，该误差由多个部分组成，包括成本矩阵的匹配误差、地理距离的惩罚、时间差的平方误差以及重合度的惩罚。

### 2. 成本矩阵和匈牙利算法

首先，通过成本矩阵 \( C \) 和匈牙利算法计算最小成本匹配，得到的总误差为：
\[ E = \sum C[row\_ind, col\_ind] \]
其中 \( row\_ind \) 和 \( col\_ind \) 是匈牙利算法返回的行索引和列索引，分别代表最优匹配的残骸和监测站。

### 3. 地理距离惩罚

对于每一对残骸 \( i \) 和 \( j \)，如果它们之间的地理距离 \( d_{ij} \) 小于最小距离 \( d_{min} \)，则增加惩罚：
\[ E += (d_{min} - d_{ij}) \times 10 \]
这里的惩罚是线性的，距离越小，惩罚越大。

### 4. 时间差的平方误差

对于每个残骸 \( i \) 和每个监测站，计算预测时间 \( t_{pred} \) 与实际到达时间 \( t_{arr} \) 的差的平方，并乘以正常误差权重 \( w \)：
\[ E += (t_{pred} - t_{arr})^2 \times w \]
这里 \( t_{pred} \) 是根据声速和距离计算得到的预测到达时间。

### 5. 重合度的计算和惩罚

计算整体重合度 \( S \)，如果重合度小于1，增加惩罚：
\[ E += (1 - S) \times w_{low} \]
其中 \( w_{low} \) 是低重合度的惩罚权重。

### 6. 重合度停滞的处理

如果重合度连续5次未改变，增加额外的惩罚：
\[ E += (1 - S) \times 100000 \]
这是为了避免优化过程中陷入局部最小值。

### 7. 重合度改善的奖励

如果重合度有所改善，根据改善程度 \( \Delta S \) 减少误差：
\[ E -= \Delta S \times (\text{abs}(\Delta S)^4) \]
如果重合度下降，增加误差：
\[ E += \text{abs}(\Delta S) \times (\text{abs}(\Delta S)^2) \]
这是为了鼓励模型向着提高重合度的方向优化。
$$
E = \sum C[row\_ind, col\_ind] + \sum_{i < j} \max(0, (d_{min} - d_{ij}) \times 10) + \sum_{i, j} (t_{pred,ij} - t_{arr,ij})^2 \times w + \max(0, (0.95 - S) \times w_{low})
$$

$$
E += \begin{cases} 
(1 - S) \times 100000 & \text{if stagnation count} \geq 5 \\
-\Delta S \times (\text{abs}(\Delta S)^4) & \text{if } \Delta S \geq 0.01 \\
\text{abs}(\Delta S) \times (\text{abs}(\Delta S)^2) & \text{if } \Delta S < 0
\end{cases}
$$

约束条件

$$
\text{time\_difference\_constraint} = 5 - (\max(\text{debris\_times}) - \min(\text{debris\_times}))
$$

$$
\text{time\_prior\_constraint} = \min(\text{min\_station\_times}) - \max(\text{debris\_times})
$$

$$
\text{altitude\_constraint} = \min(\text{debris\_altitudes})
$$

### 数学模型详细分析：计算总重合度分数

#### 1. 变量定义

- \( \textbf{x} \): 残骸的位置和时间变量数组。
- \( \textbf{p}_i \): 第 \( i \) 个残骸的位置。
- \( t_i \): 第 \( i \) 个残骸的时间。

#### 2. 计算过程

对于每个监测站，计算每个残骸到该监测站的预测到达时间，并与实际到达时间进行比较，以计算重合度分数。

##### a. 预测到达时间计算

对于每个残骸 \( i \) 和监测站 \( j \)，预测到达时间 \( t_{\text{pred},ij} \) 由以下步骤计算：

1. **地表距离**:
   \[
   d_{\text{surface},ij} = \text{geod.Inverse}(lat_{\text{station}_j}, lon_{\text{station}_j}, lat_{p_i}, lon_{p_i})['s12']
   \]

2. **高度差**:
   \[
   \Delta h_{ij} = |z_{\text{station}_j} - z_{p_i}|
   \]

3. **总距离**:
   \[
   d_{\text{total},ij} = \sqrt{d_{\text{surface},ij}^2 + \Delta h_{ij}^2}
   \]

4. **预测时间**:
   \[
   t_{\text{pred},ij} = \frac{d_{\text{total},ij}}{v_{\text{sound}}} + t_i
   \]
   其中 \( v_{\text{sound}} \) 是声速。

##### b. 重合度分数计算

对于每个监测站 \( j \)，计算所有残骸的预测时间与实际时间的差异，然后计算重合度分数：

1. **时间差异**:
   \[
   \Delta t_{ij} = |t_{\text{pred},ij} - t_{\text{arr},ij}|
   \]

2. **最大差异**:
   \[
   \Delta t_{\text{max},j} = \max(\Delta t_{ij})
   \]

3. **平均差异**:
   \[
   \Delta t_{\text{avg},j} = \text{mean}(\Delta t_{ij})
   \]

4. **重合度分数**:
   \[
   S_j = 1 - \min(1, \frac{\Delta t_{\text{avg},j}}{\text{mean}(t_{\text{arr},j})})
   \]

#### 3. 总重合度分数

总重合度分数 \( S_{\text{total}} \) 是所有监测站重合度分数的平均值：
\[
S_{\text{total}} = \frac{\sum S_j}{N}
\]
其中 \( N \) 是监测站的数量。

这个数学模型通过计算每个残骸到每个监测站的预测到达时间，并将其与实际到达时间进行比较，从而评估预测的准确性。总重合度分数 \( S_{\text{total}} \) 反映了整体预测的准确性。







监测站 A 位置: (110.241, 27.204, 824)
  残骸 1: 距离 70345.47 米, 预计到达时间 206.92 秒      
  残骸 2: 距离 61299.01 米, 预计到达时间 183.33 秒      
  残骸 3: 距离 54305.27 米, 预计到达时间 162.58 秒      
  残骸 4: 距离 54918.65 米, 预计到达时间 166.43 秒      

监测站 B 位置: (110.783, 27.456, 727)
  残骸 1: 距离 47828.67 米, 预计到达时间 140.69 秒      
  残骸 2: 距离 45628.79 米, 预计到达时间 137.25 秒      
  残骸 3: 距离 47141.26 米, 预计到达时间 141.51 秒      
  残骸 4: 距离 49399.11 米, 预计到达时间 150.20 秒      

监测站 C 位置: (110.762, 27.785, 742)
  残骸 1: 距离 27156.11 米, 预计到达时间 79.89 秒       
  残骸 2: 距离 37503.70 米, 预计到达时间 113.35 秒      
  残骸 3: 距离 51608.00 米, 预计到达时间 154.65 秒      
  残骸 4: 距离 63812.16 米, 预计到达时间 192.59 秒      

监测站 D 位置: (110.251, 28.025, 850)
  残骸 1: 距离 34667.35 米, 预计到达时间 101.98 秒      
  残骸 2: 距离 47234.50 米, 预计到达时间 141.97 秒      
  残骸 3: 距离 64962.83 米, 预计到达时间 193.93 秒      
  残骸 4: 距离 84666.65 米, 预计到达时间 253.92 秒      

监测站 E 位置: (110.524, 27.617, 786)
  残骸 1: 距离 20689.99 米, 预计到达时间 60.87 秒       
  残骸 2: 距离 22576.50 米, 预计到达时间 69.44 秒       
  残骸 3: 距离 33969.76 米, 预计到达时间 102.77 秒      
  残骸 4: 距离 47870.95 米, 预计到达时间 145.70 秒      

监测站 F 位置: (110.467, 28.081, 678)
  残骸 1: 距离 31728.27 米, 预计到达时间 93.34 秒       
  残骸 2: 距离 48080.91 米, 预计到达时间 144.46 秒      
  残骸 3: 距离 67304.43 米, 预计到达时间 200.82 秒
  残骸 4: 距离 85936.06 米, 预计到达时间 257.66 秒      

监测站 G 位置: (110.047, 27.521, 575)
  残骸 1: 距离 53540.70 米, 预计到达时间 157.49 秒      
  残骸 2: 距离 49320.74 米, 预计到达时间 148.10 秒      
  残骸 3: 距离 51769.76 米, 预计到达时间 155.13 秒      
  残骸 4: 距离 64312.09 米, 预计到达时间 194.06 秒      

监测站 A 的输出重合度: 0.61
监测站 B 的输出重合度: 0.75
监测站 C 的输出重合度: 0.98
监测站 D 的输出重合度: 0.97
监测站 E 的输出重合度: 0.85
监测站 F 的输出重合度: 0.90
监测站 G 的输出重合度: 0.74



![image-20240510223318709](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510223318709.png)



![image-20240510222213252](C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510222213252.png)

# <img src="C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510224821875.png" alt="image-20240510224821875" style="zoom:50%;" />

<img src="C:\Users\ANJIAQI\AppData\Roaming\Typora\typora-user-images\image-20240510224855329.png" alt="image-20240510224855329" style="zoom:50%;" />

```
正常优化
残骸 1: 位置 (110.36850795953924, 27.66027156553201, 2270.6915182523435), 时间 35.52129918286551
残骸 2: 位置 (110.54038724482075, 27.677137334933494, 2576.9662773374994), 时间 38.58707422465326
残骸 3: 位置 (110.46017840463895, 27.62181594584061, 2096.997412297818), 时间 39.76810629376607
残骸 4: 位置 (110.44863591116885, 27.715572848759617, 2217.7627949433463), 时间 38.964939391858906
```



```
改进优化
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.48979999816933, 27.797750971937464, 4723.342768475911), 时间 0.0165980163577283
残骸 2: 位置 (110.46123761966241, 27.68792012795248, 21024.1015515943), 时间 3.042742797940989
残骸 3: 位置 (110.45297856826406, 27.54923116635573, 33164.608578547195), 时间 2.8634576535612957
残骸 4: 位置 (110.52284030213089, 27.405504696348554, 42527.613103734344), 时间 4.903649086066846
监测站 A 的输出重合度: 0.61
监测站 B 的输出重合度: 0.76
监测站 C 的输出重合度: 0.98
监测站 D 的输出重合度: 0.97
监测站 E 的输出重合度: 0.86
监测站 F 的输出重合度: 0.90
监测站 G 的输出重合度: 0.75
```



***\*多个火箭残骸的准确定位\****

绝大多数火箭为多级火箭，下面级火箭或助推器完成既定任务后，通过级间分离装置分离后坠落。在坠落至地面过程中，残骸会产生跨音速音爆。为了快速回收火箭残骸，在残骸理论落区内布置多台震动波监测设备，以接收不同火箭残骸从空中传来的跨音速音爆，然后根据音爆抵达的时间，定位空中残骸发生音爆时的位置，再采用弹道外推实现残骸落地点的快速精准定位。

***\*附\**** 震动波的传播速度为340 m/s，计算两点间距离时可忽略地面曲率，纬度间每度距离值近似为111.263 km，经度间每度距离值近似为97.304 km。

****问题3\**** 假设各台监测设备布置的坐标和4个音爆抵达时间分别如下表所示：

| 设备 | 经度(°) | 纬度(°) | 高程(m) | 音爆抵达时间(s) |         |         |         |
| ---- | ------- | ------- | ------- | --------------- | ------- | ------- | ------- |
| A    | 110.241 | 27.204  | 824     | 100.767         | 164.229 | 214.850 | 270.065 |
| B    | 110.783 | 27.456  | 727     | 92.453          | 112.220 | 169.362 | 196.583 |
| C    | 110.762 | 27.785  | 742     | 75.560          | 110.696 | 156.936 | 188.020 |
| D    | 110.251 | 28.025  | 850     | 94.653          | 141.409 | 196.517 | 258.985 |
| E    | 110.524 | 27.617  | 786     | 78.600          | 86.216  | 118.443 | 126.669 |
| F    | 110.467 | 28.081  | 678     | 67.274          | 166.270 | 175.482 | 266.871 |
| G    | 110.047 | 27.521  | 575     | 103.738         | 163.024 | 206.789 | 210.306 |

# ***\*问题4\****  假设设备记录时间存在0.5 s的随机误差，请修正问题2所建立的模型以较精确地确定4个残骸在空中发生音爆时的位置和时间。通过对问题3表中数据叠加随机误差，给出修正模型的算例，并分析结果误差。

加入误差

```python
c41x81

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
    station['times'] = [time + np.random.normal(0, sigma) for time in station['times']]

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
        total_error += (0.5 - sigma) * 10000000  # 当sigma大于0时，增加惩罚
    if total_overlap_score < 0.95:
        total_error += (0.95 - total_overlap_score) * 10000000  # 当重合度低于0.8时，增加惩罚

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




    # 加入对sigma的惩罚
    penalty = np.log(sigma)
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

num_debris = 4
# 根据stations中的最大和最小时间来设置时间范围
min_time = min(min(station['times']) for station in stations.values())
max_time = max(max(station['times']) for station in stations.values())
#bounds = [(109, 112), (26, 29), (4693.82, 88847.59)] * num_debris + [(0, min_time)] * num_debris  # XYZ坐标范围和时间范围
# 更新变量边界，包括sigma
bounds = [(109, 112), (26, 29), (4693.82, 88847.59)] * num_debris + [(0, min_time)] * num_debris + [(0, 0.5)]  # 假设sigma的范围是0.1到10
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


优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.45727875520993, 27.744701774779813, 14102.267376279178), 时间 0.027017032105688732
残骸 2: 位置 (110.38282731615503, 27.70952260111811, 13656.256333238183), 时间 5.259283750318138
残骸 3: 位置 (110.46779375670582, 27.674229654828753, 32225.844914500343), 时间 4.892696085351794
残骸 4: 位置 (110.50310989131552, 27.53559585451663, 41304.08583035281), 时间 5.289557730318231
监测站 A 的输出重合度: 0.67
监测站 B 的输出重合度: 0.77
监测站 C 的输出重合度: 0.88
监测站 D 的输出重合度: 0.83
监测站 E 的输出重合度: 0.86
监测站 F 的输出重合度: 0.82
监测站 G 的输出重合度: 0.81
整体重合度: 0.81



```



```
初始猜测

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

优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.48990201916722, 27.79764139036915, 4712.82980169974), 时间 0.00324392518606403
残骸 2: 位置 (110.48744049670768, 27.705641858623146, 24915.383921500477), 时间 3.1276517394716525
残骸 3: 位置 (110.49369068104966, 27.640525213738197, 42889.01406889152), 时间 1.552776080287769
残骸 4: 位置 (110.52328640577767, 27.40609389133191, 42587.823602815624), 时间 4.901996569252129
监测站 A 的输出重合度: 0.61
监测站 B 的输出重合度: 0.76
监测站 C 的输出重合度: 0.98
监测站 D 的输出重合度: 0.97
监测站 E 的输出重合度: 0.86
监测站 F 的输出重合度: 0.90
监测站 G 的输出重合度: 0.75
```



 当前 sigma 值为: 0.33

优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.45727875520993, 27.744701774779813, 14102.267376279178), 时间 0.027017032105688732
残骸 2: 位置 (110.38282731615503, 27.70952260111811, 13656.256333238183), 时间 5.259283750318138
残骸 3: 位置 (110.46779375670582, 27.674229654828753, 32225.844914500343), 时间 4.892696085351794
残骸 4: 位置 (110.50310989131552, 27.53559585451663, 41304.08583035281), 时间 5.289557730318231
监测站 A 的输出重合度: 0.67
监测站 B 的输出重合度: 0.77
监测站 C 的输出重合度: 0.88
监测站 D 的输出重合度: 0.83
监测站 E 的输出重合度: 0.86
监测站 F 的输出重合度: 0.82
监测站 G 的输出重合度: 0.81
整体重合度: 0.81





# 如果时间误差无法降低，提供一种解决方案实现残骸空中的精准定位（误差<1km)，并自行根据问题3所计算得到的定位结果模拟所需的监测设备位置和音爆抵达时间数据，验证相关模型。

 

建立深度学习模型
生成数据集

```
import numpy as np
from random import uniform
from geographiclib.geodesic import Geodesic
import pandas as pd

stations = {
    'A': {'coords': (110.241, 27.204, 824), 'times': [100.767, 164.229, 214.850, 270.065]},
    'B': {'coords': (110.783, 27.456, 727), 'times': [92.453, 112.220, 169.362, 196.583]},
    'C': {'coords': (110.762, 27.785, 742), 'times': [75.560, 110.696, 156.936, 188.020]},
    'D': {'coords': (110.251, 28.025, 850), 'times': [94.653, 141.409, 196.517, 258.985]},
    'E': {'coords': (110.524, 27.617, 786), 'times': [78.600, 86.216, 118.443, 126.669]},
    'F': {'coords': (110.467, 28.081, 678), 'times': [67.274, 166.270, 175.482, 266.871]},
    'G': {'coords': (110.047, 27.521, 575), 'times': [103.738, 163.024, 206.789, 210.306]}
}
min_time = min(min(station['times']) for station in stations.values())
max_time = max(max(station['times']) for station in stations.values())
speed_of_sound = 340  # 假设声速为340 m/s

def generate_debris_data(num_debris):
    debris_data = []
    geod = Geodesic.WGS84
    for _ in range(num_debris):
        # 随机生成时间
        time = uniform(0, min_time)   # 时间范围
        # 根据时间计算可能的最大距离
        max_distance = time * speed_of_sound
        
        # 随机选择一个监测站作为参考点
        ref_station = np.random.choice(list(stations.values()))
        ref_coord = ref_station['coords']
        
        # 在最大距离范围内随机生成坐标
        bearing = uniform(0, 360)
        distance = uniform(0, max_distance)
        result = geod.Direct(ref_coord[1], ref_coord[0], bearing, distance)
        lon, lat = result['lon2'], result['lat2']
        alt = uniform(4693.82, 88847.59)  # 高度范围
        
        # 确保生成的坐标在边界内
        lon = max(min(lon, 112), 109)
        lat = max(min(lat, 29), 26)
        alt = max(min(alt, 88847.59), 4693.82)
        time = max(min(time, min_time), 0)
        
        debris = {'coords': (lon, lat, alt), 'time': time}
        
        # 计算重合度
        overlap_scores = {}
        for key, station in stations.items():
            station_coord = station['coords']
            station_times = station['times']
            station_predicted_times = []
            for debris_time in [time]:
                result = geod.Inverse(station_coord[1], station_coord[0], lat, lon)
                surface_distance = result['s12']
                height_difference = np.abs(station_coord[2] - alt)
                total_distance = np.sqrt(surface_distance**2 + height_difference**2)
                time_to_travel = total_distance / speed_of_sound + debris_time
                station_predicted_times.append(time_to_travel)
            differences = [abs(a - p) for a, p in zip(station_times, station_predicted_times)]
            max_diff = max(differences)
            overlap_score = 1 - (max_diff / max(station_times))
            overlap_scores[key] = overlap_score
        
        debris['overlap_scores'] = overlap_scores
        debris_data.append(debris)
    return debris_data

# 生成大量残骸数据
num_debris = 50000  # 指定生成的残骸数量
debris_data = []
for i in range(num_debris):
    progress = (i + 1) / num_debris * 100
    print(f"生成进度: {progress:.2f}%", end='\r')
    debris_data.append(generate_debris_data(1)[0])

# 将生成的残骸数据转换为DataFrame
debris_df = pd.DataFrame(debris_data)

# 将DataFrame中的坐标拆分为独立的列
debris_df[['longitude', 'latitude', 'altitude']] = pd.DataFrame(debris_df['coords'].tolist(), index=debris_df.index)
debris_df.drop(columns=['coords'], inplace=True)

# 将重合度字典展开为独立的列
overlap_scores_df = debris_df['overlap_scores'].apply(pd.Series)
debris_df = pd.concat([debris_df.drop(['overlap_scores'], axis=1), overlap_scores_df], axis=1)

# 存储数据集为CSV文件
debris_df.to_csv('debris_data_svi_dataset2.csv', index=False)

print("\n数据已存储为 'debris_data_svi_dataset2.csv'")

```

****问题3\**** 假设各台监测设备布置的坐标和4个音爆抵达时间分别如下表所示：

| 设备 | 经度(°) | 纬度(°) | 高程(m) | 音爆抵达时间(s) |         |         |         |
| ---- | ------- | ------- | ------- | --------------- | ------- | ------- | ------- |
| A    | 110.241 | 27.204  | 824     | 100.767         | 164.229 | 214.850 | 270.065 |
| B    | 110.783 | 27.456  | 727     | 92.453          | 112.220 | 169.362 | 196.583 |
| C    | 110.762 | 27.785  | 742     | 75.560          | 110.696 | 156.936 | 188.020 |
| D    | 110.251 | 28.025  | 850     | 94.653          | 141.409 | 196.517 | 258.985 |
| E    | 110.524 | 27.617  | 786     | 78.600          | 86.216  | 118.443 | 126.669 |
| F    | 110.467 | 28.081  | 678     | 67.274          | 166.270 | 175.482 | 266.871 |
| G    | 110.047 | 27.521  | 575     | 103.738         | 163.024 | 206.789 | 210.306 |

| time               | longitude          | latitude           | altitude          | A                    | B                    | C                    | D                     | E                    | F                    | G                     |
| ------------------ | ------------------ | ------------------ | ----------------- | -------------------- | -------------------- | -------------------- | --------------------- | -------------------- | -------------------- | --------------------- |
| 57.19389569135152  | 109.87535749252456 | 27.57826469304994  | 86653.84667667755 | 0.050999694325107314 | -0.6898125426801149  | -0.8533619002778658  | -0.05632961696529426  | -1.3201921487241837  | -0.26124784019930325 | -0.008845553023064223 |
| 25.757209624530233 | 110.72676854628882 | 27.85743446977594  | 57701.31594657616 | 0.14697453773324354  | 0.25462023069349116  | 0.363431023172349    | 0.4034275891589203    | -0.11406820225186287 | 0.4147242243051852   | 0.033062567890468575  |
| 55.704653691638775 | 109.87049785934234 | 27.504749192959075 | 77856.58692150193 | 0.16940222416209216  | -0.5903344177273473  | -0.7878182715475186  | -0.022176424011013385 | -1.1711565919473137  | -0.23669275140178025 | 0.1201367206070304    |
| 63.58982598406716  | 110.70759736906857 | 27.32898941039924  | 86755.91745874356 | 0.06446275468038842  | -0.16217200864128567 | -0.49911487205910965 | -0.28742492647570383  | -1.0520671761493499  | -0.33224364432659903 | -0.3504200075103514   |











```python
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
# 打印每个监测站的时间数据以查看随机值
for station, data in stations.items():
    print(f"原始监测站 {station} 的时间数据: {data['times']}")
# 为每个监测站的时间加上随机误差

for station in stations.values():
    station['times'] = [time + np.random.uniform(-0.25, 0.25) for time in station['times']]
# 打印每个监测站的时间数据以查看随机值
for station, data in stations.items():
    print(f"增加误差后监测站 {station} 的时间数据: {data['times']}")

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
min_distance = 5000

# 在全局作用域中定义 geod 对象
geod = Geodesic.WGS84

def objective_function(variables):
    num_debris = 4
    total_error = 0
    sigma = 0.5  # 标准偏差为0.5秒
    sigma = variables[-1]  # 获取 sigma
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
            distance = geod.Inverse(lat1, lon1, lat2, lon2)['s12']  # 使用正确的方法和参数顺序
            if distance < min_distance:
                total_error += (min_distance - distance)**2 * 10  # 添加大的惩罚

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

    # 加入对sigma的惩罚
    penalty = np.log(sigma)
    total_error += penalty
    # 打印sigma的值
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


num_debris = 4
# 根据stations中的最大和最小时间来设置时间范围
min_time = min(min(station['times']) for station in stations.values())
max_time = max(max(station['times']) for station in stations.values())
#bounds = [(109, 112), (26, 29), (4693.82, 88847.59)] * num_debris + [(0, min_time)] * num_debris  # XYZ坐标范围和时间范围
# 更新变量边界，包括sigma
bounds = [(109, 112), (26, 29), (4693.82, 88847.59)] * num_debris + [(0, min_time)] * num_debris + [(0, 0.5)]  # 假设sigma的范围是0.1到10

# 约束条件
constraints = [
    NonlinearConstraint(time_difference_constraint, 0, np.inf),
    NonlinearConstraint(time_prior_constraint, 0, np.inf),
    NonlinearConstraint(altitude_constraint, 0, np.inf)
]

# 使用差分进化算法求解
# 生成符合边界的随机初始猜测
initial_guess = np.array([np.random.uniform(low, high) for low, high in bounds])



# 执行差分进化算法
result = differential_evolution(
    objective_function,
    bounds=bounds,
    constraints=constraints,
    strategy='best1bin',
    maxiter=800,
    popsize=20,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback=None,
    disp=True,
    polish=True,
    init='random',
    atol=0,
    workers=-1  # 使用所有可用核心
)



# 提取结果
if result.success:
    estimated_positions = result.x[:num_debris*3].reshape(num_debris, 3)
    estimated_times = result.x[num_debris*3:num_debris*4]
    print("优化成功，找到可能的音爆源位置和时间")
    for i in range(num_debris):
        print(f"残骸 {i+1}: 位置 ({estimated_positions[i][0]}, {estimated_positions[i][1]}, {estimated_positions[i][2]}), 时间 {estimated_times[i]}")
else:
    print("优化失败：", result.message)

# 这段代码将设置所有必要的边界和约束，并运行差分进化算法以找到最优解，这里我们假设每个残骸的音爆发生位置和时间可以通过全局优化方法估计。

c41v20
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.48090763150937, 27.6696225052871, 6613.027348324766), 时间 22.028912712855472
残骸 2: 位置 (110.41843120252682, 27.690504870148494, 14693.611266179487), 时间 1.300422035898407
残骸 3: 位置 (110.44162659648626, 27.63009275448756, 11607.037477778758), 时间 5.243556356829455
残骸 4: 位置 (110.38058674742611, 27.644869259438245, 13095.323841069825), 时间 3.1394044370325958
```

```

```

```

```

误差小于1公里在经纬度上的表示取决于地球上的具体位置，因为经纬度到实际距离的转换因纬度而异。一般来说，纬度每变化1度，对应的实际距离大约是111公里（地球的平均半径约为6371公里，而地球周长约为40075公里，所以 \( \frac{40075}{360} \approx 111 \) 公里）。经度的变化对应的实际距离则取决于纬度，因为经线间的距离随纬度增加而减小。

### 经度和纬度的误差计算

1. **纬度上的误差**：
   - 纬度上1度的距离大约是111公里，因此1公里在纬度上的误差大约是 \( \frac{1}{111} \) 度，即约0.009度。

2. **经度上的误差**：
   - 经度上的距离变化可以通过以下公式计算：
     \[
     \text{距离} = \cos(\text{纬度}) \times 111 \text{公里}
     \]
   - 假设纬度为27度（以你的数据中的纬度为例），则1度经度上的距离大约是：
     \[
     \cos(27^\circ) \times 111 \approx 0.891 \times 111 \approx 99 \text{公里}
     \]
   - 因此，1公里在经度上的误差大约是 \( \frac{1}{99} \) 度，即约0.0101度。

### 结论

对于误差小于1公里的要求，你可以考虑在经纬度上的误差控制在大约0.009度纬度和0.0101度经度。这些计算提供了一个大致的估计，实际应用时可能需要根据具体地理位置和精确需求进行调整。

























































## ***\*问题4\****  假设设备记录时间存在0.5 s的随机误差，请修正问题2所建立的模型以较精确地确定4个残骸在空中发生音爆时的位置和时间。通过对问题3表中数据叠加随机误差，给出修正模型的算例，并分析结果误差。

在模拟实际观测中的不确定性时，引入时间数据的随机误差是一种常见的做法，旨在更真实地反映现场监测设备在记录数据时可能遇到的随机波动。例如，设备的测量精度限制和环境因素的干扰都可能导致数据的随机误差。在模拟实际观测中的不确定性时，引入时间数据的随机误差是一种有效的方法，旨在更真实地反映监测设备在记录数据时可能遇到的随机波动。这种做法通常涉及在数据中加入一个预定范围内的随机偏差，以模拟如设备精度限制或环境因素干扰等实际情况。

在本案例中，选择的随机误差遵循均匀分布，范围从-0.25秒到+0.25秒。这种均匀分布的误差模拟意味着每个观测值的随机偏差在给定的范围内具有相等的发生概率，无任何倾向性。通过这种方式，每个时间记录被随机增加或减少一段时间，不超过0.25秒。

这种随机误差的引入对于模型的测试非常重要，它不仅增加了数据的不确定性，还提供了一个机会来检验模型在处理不确定信息时的鲁棒性。如果一个模型能在包含这种随机误差的数据中稳定地提供准确的预测，那么它在实际应用中的表现也将更为可靠。此外，这也给优化算法带来了额外的挑战，因为算法需要找到一个在多次实验中都能表现良好的解决方案，从而证明其具有强大的全局搜索能力和高度适应性。

这种选择合理地模拟了实际情况中误差的分布特性，即误差在一个固定范围内随机上下波动，而没有明显的倾向性。通过对每个时间记录随机增加或减少最多0.25秒，您的模型在每次运行时都会处理略有不同的输入数据，从而检验模型在面对数据不确定性时的鲁棒性。

引入随机误差后，数据的不确定性增加，这不仅为模型的鲁棒性测试提供了机会，也增加了优化算法解决问题的挑战。优化算法需要寻找一个能够在多次包含随机误差的数据运算后都能稳定输出准确预测结果的解决方案，这要求算法具有较强的全局搜索能力和适应性。总体而言，这种方法使模型更接近于实际应用场景的需求，同时提高了对模型性能和优化方法的整体要求。









在代码中我加入了误差sigma到我的优化函数内
$$
O b j e c t i v e w i t h P e n a l t y = \frac { 1 } { 2 \sigma ^ { 2 } } \sum _ { i = 1 } ^ { n } ( t _ { i } ^ { o b s e r v e d } - ( t _ { 0 } + \frac { d _ { i } } { v } ) ) ^ { 2 } + \log ( \sigma )
$$
这个公式定义了一个包含惩罚项的目标函数，用于在数学建模中优化火箭残骸的位置和发生时间，同时考虑了观测误差。我们一步步解析这个公式：

### 公式组成

#### 1. 基本误差项

\[ \sum_{i=1}^n (t_{i}^{observed} - (t_{0} + \frac{d_{i}}{v}))^2 \]

这部分是目标函数的核心，它计算所有观测时间 \( t_i^{observed} \) 与理论上从音爆源到监测站的到达时间的平方差。其中：

- \( t_i^{observed} \) 是第 \( i \) 个监测站记录的音爆到达时间。
- \( t_0 \) 是音爆发生的时间。
- \( d_i \) 是从音爆源到第 \( i \) 个监测站的距离。
- \( v \) 是声波在空气中的传播速度。

这一项的目的是找到最佳的 \( t_0 \) 和音爆源位置，使得所有 \( d_i \) 计算得出的理论到达时间与实际观测时间的差异最小。

#### 2. 正则化/惩罚项

\[ \log(\sigma) \]

这部分是正则化项，用于防止过拟合并引入模型的稳定性。其中：

- \( \sigma \) 是表示观测噪声的参数，通常可以理解为误差的标准偏差。

这个惩罚项的作用是平衡模型的复杂性与拟合的好坏。在优化过程中，较大的 \( \sigma \) 值意味着较大的观测噪声，因此通过最小化包含 \( \log(\sigma) \) 的目标函数，我们鼓励模型选择较小的 \( \sigma \) 值，这代表着较高的数据信度。

### 数学意义与应用

整个公式的优化目标是使得总的误差平方和尽可能小，这意味着模型的预测与实际观测之间的偏差最小。同时，通过 \( \log(\sigma) \) 项，模型不会过度适应可能的测量噪声或异常值，从而提高了模型的泛化能力和鲁棒性。

在实际应用中，这个优化问题通常需要通过数值方法解决，如梯度下降、牛顿法或其他全局优化算法，因为解析解可能难以获得。这种类型的误差模型广泛用于声音定位、无线定位和其他需要同步位置和时间信息的领域。

得到了新的数学模型
您的`objective_function`定义了一个用于多目标优化的复杂数学模型，旨在通过音爆时间数据精确定位多个火箭残骸的空中位置和音爆时间。这个函数结合了几个关键的数学概念和技术，下面是对这些主要组成部分的详细解释：

### 1. **成本矩阵与匈牙利算法**

- **成本矩阵**：首先，利用`calculate_cost_matrix`函数计算了一个成本矩阵，该矩阵的每个元素代表预测音爆时间与实际观测时间之间的差异平方。成本矩阵用于衡量每个监测站观测到的音爆时间与由候选音爆位置和时间预测出的音爆到达时间之间的不匹配程度。
- **匈牙利算法**：使用线性分配问题的解决方案（匈牙利算法），为成本矩阵中的每个音爆找到对应的最佳匹配监测站，以最小化总误差。

### 2. **距离和时间差计算**

- 对每个候选音爆位置和每个监测站之间进行距离计算，并基于此距离和声速估计理论上的音爆到达时间。
- 对比理论到达时间和实际观测时间，计算时间差的平方，并根据音爆发生的不确定性（表示为σ）进行加权，加入总误差。

### 3. **几何约束和惩罚**

- **最小距离约束**：对于音爆源位置的候选解，如果两个候选位置之间的距离小于预设的最小值（如8000米），则在总误差中加入惩罚项，以避免物理上不合理的过近解。
- **重合度评分**：计算每个音爆位置预测和监测站数据之间的时间重合度，通过比较最大偏差与监测时间的最大值来衡量。
- **低重合度惩罚**：如果计算得到的重合度分数低于某一阈值（如0.95），则对总误差进行额外的惩罚，以鼓励模型找到更精确匹配监测数据的解。

### 4. **动态误差调整和模型鲁棒性**

- 使用重合度变化和时间数据的一致性动态调整误差估计，如果连续两次迭代的重合度得分未改变，则增加一个“停滞计数”，用于可能的算法收敛判断。
- 根据重合度的变化调整误差值，如果重合度有显著提高，则减少总误差，反之则增加。

### 5. **正则化项**

- 加入了对σ的正则化惩罚，通过\( \log(\sigma) \)，来控制误差项σ的大小，较大的σ值表示更大的不确定性，正则化项有助于避免过拟合和鼓励模型选择合理的误差估计。

数学模型：

在数学模型中，优化函数的定义可以全面用LaTeX来描述，以清晰展示各个数学元素和运算。以下是您代码中定义的优化函数，转换为LaTeX格式的公式：

### 目标函数定义

优化函数的核心是最小化预测的音爆到达时间和实际观测时间之间的误差，同时考虑到监测站间的最小距离约束和数据的重合度。具体公式如下：

#### 总误差计算

总误差由几部分组成，包括时间误差、距离约束惩罚和重合度惩罚：

\[ 
$$
\text{Total Error} = \sum_{i=1}^{n} \left( \frac{(t_i^{\text{obs}} - (t_0 + \frac{d_i}{v}))^2}{2 \sigma^2} \right) + \text{Penalty}_{\text{distance}} + \text{Penalty}_{\text{overlap}} + 10 \log(\sigma)
$$
\]

其中：

- \( t_i^{\text{obs}} \) 是第 \( i \) 个监测站观测到的音爆到达时间。
- \( t_0 \) 是音爆发生时间。
- \( d_i \) 是从音爆发生位置到第 \( i \) 个监测站的距离。
- \( v \) 是声速，根据环境温度调整。
- \( \sigma \) 是表示观测误差的标准偏差。

#### 距离约束惩罚

距离约束惩罚用于确保任意两个残骸之间的距离不小于设定的最小距离 \( \text{min\_distance} \)：

\[
$$
\text{Penalty}_{\text{distance}} = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \max(0, \text{min\_distance} - \text{dist}(x_i, x_j))
$$
\]

其中 \( \text{dist}(x_i, x_j) \) 计算第 \( i \) 和 \( j \) 个残骸之间的距离。

#### 重合度惩罚

重合度惩罚根据预测和观测时间的重合度给出：

\[
$$
\text{Penalty}_{\text{overlap}} = \begin{cases} 
0 & \text{if } \text{overlap\_score} \geq 0.95 \\
10^7 \times (0.95 - \text{overlap\_score}) & \text{otherwise}
\end{cases}
$$
\]

#### 重合度得分

重合度得分是基于预测和实际到达时间之间的差异，用以下公式计算：

\[
$$
\text{overlap\_score} = 1 - \max_{i} \left( \frac{|t_i^{\text{obs}} - t_i^{\text{pred}}|}{\max(t_i^{\text{obs}})} \right)
$$
\]

其中 \( t_i^{\text{pred}} \) 是第 \( i \) 个监测站根据音爆位置和声速计算出的预测到达时间。

通过这些公式，优化函数精确描述了如何结合物理约束、数据误差和预测精度，来寻找最优的音爆位置和时间，以实现在给定观测数据下的最佳音爆源定位。

数学模型的约束条件继续沿用第三题的约束



多维音爆源定位优化算法通过综合考虑多个关键因素来确保模型的准确性和可靠性。首先该函数通过最小化每个监测站观测到的音爆到达时间与从音爆源位置和声速计算出的理论到达时间之间的差的平方和，直接针对预测的准确性进行了优化。这不仅有效地调整了音爆源的位置和发生时间，还显著减少了预测与实际观测之间的偏差。

误差项中引入了以标准偏差 \( \sigma \) 为参数的权重。这种设计不仅调整了各误差的贡献，使得模型在面对数据噪声时保持灵活性，还通过引入正则化项 \( \log(\sigma) \)，促使模型选择一个较小的 \( \sigma \) 值，从而在拟合数据时平衡精度与过拟合的风险。

模型通过引入重合度得分机制进一步提升了预测的相关性和实用性。通过设定一个阈值并对未达到此标准的情况施加惩罚，这种策略不仅鼓励模型在保持低预测误差的同时，也确保了预测结果与实际观测数据在时间上的高度吻合，极大地提高了模型在实际操作中的有效性。

优化函数通过这些精心设计的机制，不仅能够在理论上达到高精度，还能在实际应用中展现出卓越的适应性和可靠性，为解决复杂的实际问题提供了一种有效的多目标优化策略。



输出分析

在本问题研究中，所采用的优化函数显著提高了模型性能，尤其是在减少测量误差和维持预测准确性方面表现突出。模型通过设定一个标准偏差 \( \sigma \) 作为优化目标的其中一个，在不大幅减少重合度的情况下将标准偏差 \( \sigma \) 值降低至0.33，有效地控制了测量误差，显示出对数据的高度拟合能力。这一成就主要归功于三个方面的技术选择和策略实施：首先，采用差分进化算法增强了模型的全局搜索能力，有效避免了陷入局部最优解并显著减少了整体误差；其次，通过引入正则化项 \( \log(\sigma) \)，模型在减少过拟合的同时增强了泛化能力；最后，模型通过精确计算理论与实际到达时间的差异，确保了误差评估的精确性。此外，尽管测量误差得到了显著降低，模型的整体重合度保持在较高水平（0.81），表明在提升数据处理精度的同时，并未牺牲预测的准确性，反映出模型设计的合理性和实用性。



根据重复度的定义，我生成了10万条数据以及他们所对应的重合度，来对比我得重复度，在10万条随机生成的数据中，我检测他们的重合度，虽然有很多残骸的重合度在单个的检测设备上可以达到0.99，但是综合7个重合度后，并没有超过0.8重合度的残骸坐标。

![geographic_distribution](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\geographic_distribution.png)

![time_series](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\time_series.png)

![measurement_distribution](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\measurement_distribution.png)

## 如果时间误差无法降低，提供一种解决方案实现残骸空中的精准定位（误差<1km)，并自行根据问题3所计算得到的定位结果模拟所需的监测设备位置和音爆抵达时间数据，验证相关模型。



在地理坐标系统中，经纬度的变化与实际地表距离之间的关系取决于地球的几何形状和位置的地理纬度。对于经纬度系统，北纬或南纬每变化1度，地表上的实际距离约为111公里，这一估算基于地球平均半径大约为6371公里，而地球的周长大约为40075公里，从而得出每度大约为 \( \frac{40075}{360} \approx 111 \) 公里。

对于纬度的变化，每变化1度对应的地表距离相对固定。因此，1公里在纬度上对应的变化大约为 \( \frac{1}{111} \) 度，即约0.009度。然而，经度的变化对应的实际距离随纬度的增加而减小，这是因为经线在两极处相交，其间距随纬度的增加而缩小。例如，在纬度27度处，1度经度上的实际距离可通过计算 \( \cos(27^\circ) \times 111 \approx 0.891 \times 111 \approx 99 \) 公里来估算。因此，1公里在经度上的变化大约对应 \( \frac{1}{99} \) 度，即约0.0101度。

基于以上分析，在进行精确的地理位置测量和定位时，特别是在误差要求小于1公里的情况下，应当考虑纬度上约0.009度和经度上约0.0101度的误差。这种估算为地理数据处理提供了一种基准，但具体应用中可能需要根据特定地点的地理特性进行适当调整。

但是根据不同的预测边界区间，会有不同对应的残骸坐标和音爆时间，因此我得模型虽然降低了测量误差，但是不能实现残骸空中的精准定位（误差<1km)，因此我修改了第三题中的数学模型，去处了对重合度的优化，加强对误差的优化。在进行优化和模拟实验时，选择恰当的边界（bounds）是确保算法性能和结果可靠性的重要环节。边界的设定直接关系到残骸位置和时间预测的精确度，对优化过程的效率和结果的有效性有深远影响。

### 边界设定的原则与实践

边界设定应以确保新的预测值不会偏离潜在的实际区域为目标，从而保证算法的实用性和结果的可靠性。在实际操作中，经度和纬度的边界是根据已有数据的地理位置进行设定的，通过适当扩展这些边界（例如，将经度和纬度各扩展0.1度），可以为位置偏差和测量误差提供必要的容错空间。此外，高度的边界考虑了可能的飞行高度和地形变化，通过设置一个覆盖最小高度减少100米至最大高度增加100米的范围，为残骸的高度变化提供了足够的适应性。时间边界则基于监测站记录的最早和最晚时间，以确保音爆发生时间的预测不会超出这些观测时间。

### 边界设置对优化的影响

适当设定的边界能有效限制搜索空间，降低计算量，从而提升算法的收敛速度和整体效率。然而，边界设置过宽可能导致搜索效率低下，而设置过窄可能使算法错过最优解。因此，边界设置的合理性直接影响到优化结果的质量和算法的性能。通过精心设计的边界，可以引导优化过程顺利进行，减少不必要的计算，并确保所得结果的实际应用价值。

在数学建模和优化算法中，边界的合理设定是确保模型输出精确可靠的关键。这一过程需要充分考虑问题的物理背景、实际条件及预期应用结果。通过灵活而精确的边界设定，模型不仅能够提供高效的解决方案，还能在不同的应用场景中展现出良好的适应性和可靠性。综上所述，边界设定应视为优化过程中的一个基本而关键的步骤，对于提升模型的实用性和有效性至关重要。



选取好边界后改进的数学模型

在优化残骸定位的问题中，我们构建了一个数学模型来最小化预测位置与实际位置之间的误差，并在此过程中考虑各种约束和惩罚项。以下是详细的数学表达式描述：

### 目标函数定义

目标函数设计为最小化预测的音爆到达时间与实际记录时间的误差的平方和，同时考虑地理距离的约束和对模型精确度的惩罚。具体数学表达式如下：

\[
$$
\text{Total Error} = \sum_{i, j} \left( \frac{(t_{ij}^{\text{predicted}} - t_{ij}^{\text{actual}})^2}{2 \sigma^2} \right) + \sum_{i < j} \text{Penalty for proximity} + \text{Penalty for precision}
$$
\]

其中:

- \( t_{ij}^{\text{predicted}} \) 是从第 \( i \) 个残骸位置到第 \( j \) 个监测站的预测到达时间。
- \( t_{ij}^{\text{actual}} \) 是实际观测到的到达时间。
- \( \sigma \) 是预测误差的标准偏差。

### 惩罚项

1. **距离惩罚**：
   \[
   $$
   \text{Penalty for proximity} = \sum_{i < j} \left\{
     \begin{array}{ll}
       10 \times (\text{min\_distance} - d_{ij})^2 & \text{if } d_{ij} < \text{min\_distance} \\
       0 & \text{otherwise}
     \end{array}
   \right.
   $$
   \]
   其中 \( d_{ij} \) 是第 \( i \) 个和第 \( j \) 个残骸之间的地理距离，\(\text{min\_distance}\) 是设定的最小允许距离。

2. **精确度惩罚**：
   \[
   $$
   \text{Penalty for precision} = \left( \log(\sigma) \right)^2
   $$
   \]
   这一项惩罚过大或过小的 \(\sigma\) 值，鼓励模型选择一个合理的误差标准差，以平衡模型的灵活性和精确度。

### 约束条件

我们还包括了几个约束条件以确保解决方案的可行性：

1. **时间差异约束**：
   \[
   5 \geq \max(t_{\text{debris}}) - \min(t_{\text{debris}})
   \]
   这确保所有预测的音爆发生时间在一个合理的时间窗口内。

2. **时间先行约束**：
   \[
   \min(t_{\text{debris}}) \geq \max(t_{\text{actual}})
   \]
   音爆发生时间应早于或等于所有监测站记录的最晚时间。

3. **高程约束**：
   \[
   \min(z_{\text{debris}}) \geq 0
   \]
   确保所有残骸的高度为非负。

通过这种方式，我们构建的模型不仅能够优化音爆源的位置预测，还能确保预测结果在物理和实际操作上是可行的。这种数学模型的应用，可以有效地指导实际的优化算法实施，提高解决方案的实际应用价值。

根据题目3中不加入重合度优化的数学模型输出的位置确定我们的预测边界区间

110.36775803310434, 27.65017384900437, 4977.249541517314, 27.15493328285458,

  110.44051378222676, 27.71323885268561, 5258.530775804032, 29.39245076502419,

  110.46427165342708, 27.622689873958485, 5218.195503379928, 31.35155219019203,

  110.53700942866655, 27.685533644489574, 5151.379507652564, 29.635226697775398

![4题21 3D视图_音爆源位置与监测站位置](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\4题21 3D视图_音爆源位置与监测站位置.png)

在修改我们的新的数学模型的预测边界区间后，经过加误差后的时间参数

![4题22 3D视图_音爆源位置与监测站位置](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\4题22 3D视图_音爆源位置与监测站位置.png)

110.36775803310434, 27.65017384900437, 4977.249541517314, 27.15493328285458,

  110.44051378222676, 27.71323885268561, 5258.530775804032, 29.39245076502419,

  110.46427165342708, 27.622689873958485, 5218.195503379928, 31.35155219019203,

  110.53700942866655, 27.685533644489574, 5151.379507652564, 29.635226697775398

优化成功，加入误差后，预测的可能的音爆源位置和时间
残骸 1: 位置 (110.37684549207111, 27.653388972700085, 4901.869022796991), 时间 28.501868119846854
残骸 2: 位置 (110.44549918081623, 27.70548355801759, 5084.592869130779), 时间 30.14020104632939
残骸 3: 位置 (110.4606558306885, 27.630896111684596, 5053.35375941329), 时间 31.323526675761613
残骸 4: 位置 (110.52748923066916, 27.685147831281878, 4976.67557846196), 时间 29.84609788552375
新数据残骸 1 与旧数据残骸 1 之间的距离是 1.025 km
新数据残骸 2 与旧数据残骸 2 之间的距离是 0.657 km
新数据残骸 3 与旧数据残骸 3 之间的距离是 0.541 km
新数据残骸 4 与旧数据残骸 4 之间的距离是 1.077 km
平均距离是 0.825 km
平均绝对误差 (MAE): 49.076
均方根误差 (RMSE): 88.332





![4题23 3D视图_音爆源位置与监测站位置](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\4题23 3D视图_音爆源位置与监测站位置.png)



![4题24 新数据残骸与旧数据残骸之间的距离](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\4题24 新数据残骸与旧数据残骸之间的距离.png)

新数据残骸 1 与旧数据残骸 1 之间的距离是 1.025 km
新数据残骸 2 与旧数据残骸 2 之间的距离是 0.657 km
新数据残骸 3 与旧数据残骸 3 之间的距离是 0.541 km
新数据残骸 4 与旧数据残骸 4 之间的距离是 1.077 km
平均距离是 0.825 km
平均绝对误差 (MAE): 49.076
均方根误差 (RMSE): 88.332


重复4组数据，查看效果

1

残骸 1: 位置 (110.45704378248004, 27.623199210719406, 4779.163127337799), 时间 28.24680946471557
残骸 2: 位置 (110.44485032126164, 27.713534350792806, 4758.0532216979045), 时间 28.443594375319602
残骸 3: 位置 (110.53844892140896, 27.67885478511058, 4810.758490988309), 时间 30.99233454255877
残骸 4: 位置 (110.36300872335809, 27.66028218726307, 4710.00821682809), 时间 29.78560604461502

优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.45175640750922, 27.630334592875545, 4790.735029769947), 时间 30.49830838645756
残骸 2: 位置 (110.44757581248732, 27.705397160641624, 4654.534281855), 时间 30.994684700410307
残骸 3: 位置 (110.52877424320475, 27.680700220079625, 4787.668627035571), 时间 30.62681308588005
残骸 4: 位置 (110.37249161762452, 27.662020720721113, 4653.1973997710475), 时间 28.36582556299747
新数据残骸 1 与旧数据残骸 1 之间的距离是 0.653 km
新数据残骸 2 与旧数据残骸 2 之间的距离是 0.452 km
新数据残骸 3 与旧数据残骸 3 之间的距离是 1.083 km
新数据残骸 4 与旧数据残骸 4 之间的距离是 1.062 km
平均距离是 0.812 km
平均绝对误差 (MAE): 16.253
均方根误差 (RMSE): 34.894

2

优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.460076488883, 27.622772784491204, 4818.83604464667), 时间 31.35914037555505
残骸 2: 位置 (110.53906794835201, 27.682314298902014, 4938.010354628853), 时间 30.744717348174838
残骸 3: 位置 (110.44359407309996, 27.712741703713736, 4817.767428227544), 时间 30.483208424357777
残骸 4: 位置 (110.36529645602708, 27.654690879319908, 4707.9417365712025), 时间 26.771203927472214

优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.45544536991486, 27.63053683800406, 4950.596064905634), 时间 31.2040984721963
残骸 2: 位置 (110.52910282643404, 27.6822017769695, 4758.857935195082), 时间 30.64974104878219
残骸 3: 位置 (110.44395072257264, 27.703845142377492, 4727.12630323345), 时间 30.89752849341641
残骸 4: 位置 (110.3743826375412, 27.65786175981871, 4793.929397585791), 时间 28.22048295547266
新数据残骸 1 与旧数据残骸 1 之间的距离是 0.614 km
新数据残骸 2 与旧数据残骸 2 之间的距离是 1.127 km
新数据残骸 3 与旧数据残骸 3 之间的距离是 0.361 km
新数据残骸 4 与旧数据残骸 4 之间的距离是 1.025 km
平均距离是 0.782 km
平均绝对误差 (MAE): 40.632
均方根误差 (RMSE): 73.635

3

优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.36917101959187, 27.654664724246402, 4880.287635085775), 时间 26.341065201988084
残骸 2: 位置 (110.54225657510224, 27.680972346126115, 4783.975685849679), 时间 28.830926566478713
残骸 3: 位置 (110.46420124319684, 27.623262204107068, 5012.360520168654), 时间 30.295366887460936
残骸 4: 位置 (110.44712767858933, 27.712348594508548, 5114.670877940404), 时间 28.336217009446017

优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.37929101858248, 27.65524818124884, 4754.448352749391), 时间 28.293182568896384
残骸 2: 位置 (110.53218089116733, 27.6808252376013, 4886.943163187662), 时间 30.79638351063751
残骸 3: 位置 (110.45761936955915, 27.629725868107073, 5039.532584950864), 时间 30.78883931984181
残骸 4: 位置 (110.44608641179445, 27.70410803316553, 4751.384347179981), 时间 30.62923795328568
新数据残骸 1 与旧数据残骸 1 之间的距离是 1.137 km
新数据残骸 2 与旧数据残骸 2 之间的距离是 1.130 km
新数据残骸 3 与旧数据残骸 3 之间的距离是 0.777 km
新数据残骸 4 与旧数据残骸 4 之间的距离是 0.499 km
平均距离是 0.886 km
平均绝对误差 (MAE): 51.609
均方根误差 (RMSE): 115.164

4

优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.46317568666993, 27.62407055219366, 4978.2544497397585), 时间 31.169758791391562
残骸 2: 位置 (110.36563251168116, 27.648502062912225, 4752.947587261285), 时间 27.192464402120496
残骸 3: 位置 (110.53604868130121, 27.6879637707418, 4802.440780259062), 时间 31.676729265634812
残骸 4: 位置 (110.43819523159837, 27.711580619283176, 4729.851481673799), 时间 29.81713632947713

优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.45629068624983, 27.630542297936916, 4828.616617223717), 时间 30.970088545239406
残骸 2: 位置 (110.37534018704517, 27.650549841715176, 4699.678472060458), 时间 28.20420239418303
残骸 3: 位置 (110.52633343142271, 27.689599752435083, 4634.703574361043), 时间 30.35183531483277
残骸 4: 位置 (110.43831889362556, 27.702883972580143, 4670.589723134612), 时间 30.19256091848779
新数据残骸 1 与旧数据残骸 1 之间的距离是 0.823 km
新数据残骸 2 与旧数据残骸 2 之间的距离是 1.088 km
新数据残骸 3 与旧数据残骸 3 之间的距离是 1.099 km
新数据残骸 4 与旧数据残骸 4 之间的距离是 0.345 km
平均距离是 0.839 km
平均绝对误差 (MAE): 35.829
均方根误差 (RMSE): 68.846

![4-2-4Figure_1](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\4-2-4Figure_1.png)



可以看到，通过比较四组数据来分析优化算法对残骸位置和音爆发生时间预测的表现。结果显示，即便在引入随机误差的情况下，残骸位置的新旧数据之间的平均距离偏差维持在0.782至0.886公里范围内，表明算法保持了较高的预测精度。此外，音爆发生时间的预测虽有波动，但整体变化有限，反映了模型对时间预测的鲁棒性。进一步分析误差指标，发现平均绝对误差（MAE）和均方根误差（RMSE）呈现增加趋势，从数值上揭示了模型在引入误差后稳定性和预测精度的变化。

这些分析结果证实了所使用的优化算法和数学模型在处理带有随机误差的数据时仍能维持良好的稳定性和可靠性。尽管模型已显示出适应性和良好效果，残骸位置的最小和最大距离偏差及其对应的误差指标差异指出，模型在面对不同初始条件和随机误差的影响时，仍需进一步的调整和优化，以提升其在更广泛应用条件下的性能。

针对以上分析，建议进一步优化模型参数，例如调整惩罚权重和标准偏差的估计，以更好地适应数据波动和提升预测准确性。同时，探索和实验不同的优化算法或改进现有算法的策略，可以增强算法的全局搜索能力和收敛速度。此外，加强数据预处理工作，尤其是通过更精细的方法处理数据异常值，也将进一步增强数据的质量和一致性。通过这些措施，可以有效提升模型的实用性和有效性，为未来的研究和实际应用提供坚实的基础。



在加入重合度优化后使用之前的模型和沿用之前预测到的数据所形成的预测边界区间

 当前 sigma 值为: 0.22
优化成功，找到可能的音爆源位置和时间
残骸 1: 位置 (110.46472278376797, 27.630528553017584, 5023.2157189526115), 时间 0.0040844902626275825
残骸 2: 位置 (110.53578368814179, 27.677624101732373, 4896.95139206578), 时间 22.044868890650683
残骸 3: 位置 (110.44749698232485, 27.70905138638477, 4802.831689803336), 时间 44.4925351877494
残骸 4: 位置 (110.37504212103934, 27.65220355683705, 5022.352707456774), 时间 67.27363798491513
新数据残骸 1 与旧数据残骸 1 之间的距离是 0.635 km
新数据残骸 2 与旧数据残骸 2 之间的距离是 0.412 km
新数据残骸 3 与旧数据残骸 3 之间的距离是 0.459 km
新数据残骸 4 与旧数据残骸 4 之间的距离是 1.137 km
平均距离是 0.661 km
平均绝对误差 (MAE): 47.902
均方根误差 (RMSE): 108.986
监测站 A 的输出重合度: 0.81
监测站 B 的输出重合度: 0.92
监测站 C 的输出重合度: 0.87
监测站 D 的输出重合度: 0.75
监测站 E 的输出重合度: 0.55
监测站 F 的输出重合度: 0.70
监测站 G 的输出重合度: 0.82
整体重合度: 0.77

最终得到了  整体重合度: 0.77  sigma 值为: 0.22，并且平均距离是 0.661 km的优秀输出

![音爆源位置与监测站输出重合度可视化](D:\HuaweiMoveData\Users\ANJIAQI\Desktop\hja\音爆源位置与监测站输出重合度可视化.png)
