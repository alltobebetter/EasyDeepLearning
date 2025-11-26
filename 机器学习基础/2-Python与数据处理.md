# 第二章：Python与数据处理基础 

> Copyright 2025 石旭乔 306开放创新实验室

> 工欲善其事，必先利其器：掌握机器学习的必备工具

---

## 2.1 Python环境搭建

### 为什么选择Python？

**比喻：** Python是机器学习界的"普通话"，大家都在用。

**优势：**
- 丰富的库：NumPy、Pandas、Scikit-learn
- 学习曲线平缓：语法简单易懂
- 社区活跃：遇到问题容易找到答案
- 开发效率高：代码简洁

### 安装Anaconda

**什么是Anaconda？**
- **比喻：** Anaconda = Python全家桶套餐
- 包含Python + 常用库 + Jupyter Notebook + 包管理工具

**安装步骤：**

1. **下载Anaconda**
- 访问：https://www.anaconda.com/download
- 选择对应操作系统版本
- 下载Python 3.x版本

2. **安装**
```
Windows: 双击安装包，一路Next
Mac: 双击.pkg文件安装
Linux: bash Anaconda3-xxx.sh
```

3. **验证安装**
```bash
# 打开终端/命令行
python --version
# 应该显示：Python 3.x.x

conda --version
# 应该显示：conda x.x.x
```

### Jupyter Notebook快速上手

**什么是Jupyter Notebook？**
- **比喻：** Jupyter = 可以边写边运行的笔记本
- 适合学习和实验
- 可以混合代码、文字、图表

**启动Jupyter：**
```bash
# 在终端输入
jupyter notebook

# 浏览器会自动打开
# 地址：http://localhost:8888
```

**基本操作：**
```
创建新笔记本：New → Python 3
运行代码：Shift + Enter
添加单元格：点击 + 按钮
切换类型：Code（代码）/ Markdown（文字）
```

**第一个程序：**
```python
# 在Jupyter中输入并运行
print("Hello, Machine Learning!")

# 输出：Hello, Machine Learning!
```

---

## 2.2 NumPy：数值计算基础

### 什么是NumPy？

**全称：** Numerical Python（数值Python）

**比喻：** NumPy = 计算器的超级版，专门处理数组和矩阵运算。

**为什么需要NumPy？**
```python
# Python原生列表：慢
list1 = [1, 2, 3, 4, 5]
list2 = [10, 20, 30, 40, 50]
# 想要对应相加？需要循环

# NumPy数组：快
import numpy as np
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])
result = arr1 + arr2 # 直接相加！
# 结果：[11, 22, 33, 44, 55]
```

### 导入NumPy

```python
import numpy as np # 约定俗成的简写
```

### 创建数组

#### 1. 从列表创建
```python
# 一维数组
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d) # [1 2 3 4 5]

# 二维数组（矩阵）
arr2d = np.array([[1, 2, 3],
[4, 5, 6]])
print(arr2d)
# [[1 2 3]
# [4 5 6]]
```

**比喻：**
- 一维数组 = 一排数字（像一行座位）
- 二维数组 = 表格（像Excel表格）

#### 2. 特殊数组
```python
# 全0数组
zeros = np.zeros((3, 4)) # 3行4列的0
# [[0. 0. 0. 0.]
# [0. 0. 0. 0.]
# [0. 0. 0. 0.]]

# 全1数组
ones = np.ones((2, 3)) # 2行3列的1

# 等差数列
arange = np.arange(0, 10, 2) # 0到10，步长2
# [0 2 4 6 8]

# 等分数列
linspace = np.linspace(0, 1, 5) # 0到1，分成5份
# [0. 0.25 0.5 0.75 1. ]

# 随机数组
random = np.random.rand(3, 3) # 3x3的随机数（0-1之间）
```


### 数组属性

```python
arr = np.array([[1, 2, 3, 4],
[5, 6, 7, 8]])

# 形状（几行几列）
print(arr.shape) # (2, 4) - 2行4列

# 维度
print(arr.ndim) # 2 - 二维

# 元素个数
print(arr.size) # 8 - 共8个元素

# 数据类型
print(arr.dtype) # int64
```

**比喻：**
- shape = 表格的规格（几行几列）
- ndim = 表格的层数（一层、两层、三层...）
- size = 表格里有多少个格子

### ➕ 数组运算

#### 1. 基本运算
```python
arr = np.array([1, 2, 3, 4, 5])

# 加减乘除（每个元素都参与）
print(arr + 10) # [11 12 13 14 15]
print(arr * 2) # [2 4 6 8 10]
print(arr ** 2) # [1 4 9 16 25] - 平方

# 数组之间运算
arr1 = np.array([1, 2, 3])
arr2 = np.array([10, 20, 30])
print(arr1 + arr2) # [11 22 33]
```

**比喻：** 就像给每个学生的分数都加10分。

#### 2. 统计函数
```python
arr = np.array([1, 2, 3, 4, 5])

print(arr.sum()) # 15 - 求和
print(arr.mean()) # 3.0 - 平均值
print(arr.max()) # 5 - 最大值
print(arr.min()) # 1 - 最小值
print(arr.std()) # 1.41... - 标准差
```

**比喻：** 统计班级成绩的总分、平均分、最高分、最低分。

#### 3. 矩阵运算
```python
# 矩阵乘法
A = np.array([[1, 2],
[3, 4]])
B = np.array([[5, 6],
[7, 8]])

# 点乘（矩阵乘法）
C = np.dot(A, B)
# 或者
C = A @ B

# 转置
A_T = A.T
```

### 索引与切片

```python
arr = np.array([10, 20, 30, 40, 50])

# 索引（从0开始）
print(arr[0]) # 10 - 第1个元素
print(arr[-1]) # 50 - 最后一个元素

# 切片
print(arr[1:4]) # [20 30 40] - 第2到第4个
print(arr[:3]) # [10 20 30] - 前3个
print(arr[2:]) # [30 40 50] - 从第3个到最后

# 二维数组索引
arr2d = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])

print(arr2d[0, 0]) # 1 - 第1行第1列
print(arr2d[1, 2]) # 6 - 第2行第3列
print(arr2d[0, :]) # [1 2 3] - 第1行所有列
print(arr2d[:, 0]) # [1 4 7] - 所有行第1列
```

**比喻：**
- 索引 = 找到第几个座位
- 切片 = 选择一段连续的座位

### 布尔索引

```python
arr = np.array([1, 2, 3, 4, 5])

# 找出大于3的元素
mask = arr > 3
print(mask) # [False False False True True]
print(arr[mask]) # [4 5]

# 简写
print(arr[arr > 3]) # [4 5]

# 多条件
print(arr[(arr > 2) & (arr < 5)]) # [3 4]
```

**比喻：** 像筛选器，只留下符合条件的。

---

## 2.3 Pandas：数据分析利器

### 什么是Pandas？

**比喻：** Pandas = Excel的编程版，专门处理表格数据。

**为什么需要Pandas？**
- 处理CSV、Excel等表格数据
- 数据清洗、筛选、分组
- 比NumPy更适合处理真实数据

### 导入Pandas

```python
import pandas as pd # 约定俗成的简写
```

### 两种数据结构

#### 1. Series（一维）

**比喻：** Series = 带标签的一列数据（像Excel的一列）

```python
# 创建Series
s = pd.Series([10, 20, 30, 40, 50])
print(s)
# 0 10
# 1 20
# 2 30
# 3 40
# 4 50

# 自定义索引
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
# a 10
# b 20
# c 30

# 访问元素
print(s['a']) # 10
```

#### 2. DataFrame（二维）

**比喻：** DataFrame = 完整的Excel表格

```python
# 创建DataFrame
data = {
'姓名': ['张三', '李四', '王五'],
'年龄': [25, 30, 35],
'城市': ['北京', '上海', '深圳']
}
df = pd.DataFrame(data)
print(df)
# 姓名 年龄 城市
# 0 张三 25 北京
# 1 李四 30 上海
# 2 王五 35 深圳
```

### 读取数据

```python
# 读取CSV文件
df = pd.read_csv('data.csv')

# 读取Excel文件
df = pd.read_excel('data.xlsx')

# 读取JSON文件
df = pd.read_json('data.json')

# 查看前几行
print(df.head()) # 默认前5行
print(df.head(10)) # 前10行

# 查看后几行
print(df.tail()) # 默认后5行
```

### 数据查看

```python
# 基本信息
print(df.shape) # (行数, 列数)
print(df.columns) # 列名
print(df.info()) # 数据类型、缺失值等
print(df.describe()) # 统计摘要

# 查看某一列
print(df['姓名'])

# 查看多列
print(df[['姓名', '年龄']])

# 查看某一行
print(df.iloc[0]) # 第1行（按位置）
print(df.loc[0]) # 索引为0的行（按标签）
```

### 数据筛选

```python
# 条件筛选
df_filtered = df[df['年龄'] > 25]
# 筛选年龄大于25的

# 多条件
df_filtered = df[(df['年龄'] > 25) & (df['城市'] == '上海')]
# 年龄大于25且城市是上海的

# 使用query方法（更直观）
df_filtered = df.query('年龄 > 25 and 城市 == "上海"')
```

**比喻：** 像Excel的筛选功能，只显示符合条件的行。


### 数据清洗

#### 1. 处理缺失值

```python
# 检查缺失值
print(df.isnull()) # 返回布尔值
print(df.isnull().sum()) # 每列缺失值数量

# 删除缺失值
df_clean = df.dropna() # 删除有缺失值的行
df_clean = df.dropna(axis=1) # 删除有缺失值的列

# 填充缺失值
df_filled = df.fillna(0) # 用0填充
df_filled = df.fillna(df.mean()) # 用平均值填充
df_filled = df.fillna(method='ffill') # 用前一个值填充
```

**比喻：**
- dropna = 把有空格的行删掉
- fillna = 把空格填上

#### 2. 处理重复值

```python
# 检查重复
print(df.duplicated()) # 返回布尔值

# 删除重复
df_unique = df.drop_duplicates()

# 保留最后一次出现的
df_unique = df.drop_duplicates(keep='last')
```

#### 3. 数据类型转换

```python
# 查看数据类型
print(df.dtypes)

# 转换类型
df['年龄'] = df['年龄'].astype(int)
df['日期'] = pd.to_datetime(df['日期'])
```

### 数据分组与聚合

```python
# 按城市分组，计算平均年龄
grouped = df.groupby('城市')['年龄'].mean()

# 多种聚合
grouped = df.groupby('城市').agg({
'年龄': ['mean', 'max', 'min'],
'工资': 'sum'
})
```

**比喻：** 像Excel的数据透视表，按类别统计。

### 数据合并

```python
# 纵向拼接（上下拼接）
df_combined = pd.concat([df1, df2])

# 横向拼接（左右拼接）
df_combined = pd.concat([df1, df2], axis=1)

# 类似SQL的JOIN
df_merged = pd.merge(df1, df2, on='ID', how='inner')
# how: 'inner', 'outer', 'left', 'right'
```

**比喻：**
- concat = 把两张表格拼在一起
- merge = 根据共同列合并（像拼图）

---

## 2.4 Matplotlib：数据可视化

### 什么是Matplotlib？

**比喻：** Matplotlib = 画笔和画布，把数据画成图表。

**为什么需要可视化？**
- 一图胜千言
- 快速发现数据规律
- 展示分析结果

### 导入Matplotlib

```python
import matplotlib.pyplot as plt
# 在Jupyter中显示图表
%matplotlib inline
```

### 折线图

```python
# 数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 绘制
plt.plot(x, y)
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('折线图示例')
plt.show()
```

**用途：** 展示趋势变化（股票走势、温度变化）

### 柱状图

```python
# 数据
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 55]

# 绘制
plt.bar(categories, values)
plt.xlabel('类别')
plt.ylabel('数值')
plt.title('柱状图示例')
plt.show()
```

**用途：** 对比不同类别（销售额对比、成绩对比）

### 散点图

```python
# 数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# 绘制
plt.scatter(x, y)
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('散点图示例')
plt.show()
```

**用途：** 展示两个变量的关系（身高vs体重）

### 直方图

```python
# 数据
data = [1, 2, 2, 3, 3, 3, 4, 4, 5]

# 绘制
plt.hist(data, bins=5)
plt.xlabel('数值')
plt.ylabel('频数')
plt.title('直方图示例')
plt.show()
```

**用途：** 展示数据分布（成绩分布、年龄分布）

### 饼图

```python
# 数据
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]

# 绘制
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('饼图示例')
plt.show()
```

**用途：** 展示占比（市场份额、预算分配）

### 图表美化

```python
# 设置样式
plt.style.use('seaborn') # 使用seaborn风格

# 设置图表大小
plt.figure(figsize=(10, 6))

# 设置颜色
plt.plot(x, y, color='red', linewidth=2)

# 添加网格
plt.grid(True)

# 添加图例
plt.plot(x, y1, label='线1')
plt.plot(x, y2, label='线2')
plt.legend()

# 保存图片
plt.savefig('my_plot.png', dpi=300)
```

---

## 2.5 数据预处理

### 为什么需要数据预处理？

**比喻：** 数据预处理 = 做菜前的准备工作（洗菜、切菜）

**原因：**
- 原始数据通常很"脏"（缺失、异常、格式不统一）
- 不同特征的量纲不同（身高用cm，体重用kg）
- 机器学习算法对数据格式有要求

###  特征缩放

#### 1. 标准化 (Standardization)

**公式：** `z = (x - mean) / std`

**比喻：** 把所有数据转换成"标准分"（均值0，标准差1）

```python
from sklearn.preprocessing import StandardScaler

# 原始数据
data = [[1, 2], [3, 4], [5, 6]]

# 标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print(data_scaled)
# [[-1.22 -1.22]
# [ 0. 0. ]
# [ 1.22 1.22]]
```

**何时使用：**
- 数据符合正态分布
- 算法对数据分布敏感（SVM、神经网络）

#### 2. 归一化 (Normalization)

**公式：** `x_new = (x - min) / (max - min)`

**比喻：** 把所有数据压缩到0-1之间

```python
from sklearn.preprocessing import MinMaxScaler

# 原始数据
data = [[1, 2], [3, 4], [5, 6]]

# 归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

print(data_scaled)
# [[0. 0. ]
# [0.5 0.5]
# [1. 1. ]]
```

**何时使用：**
- 数据有明确的上下界
- 需要保持数据的分布形状

**对比：**
```
原始数据：身高[150, 160, 170, 180]cm, 体重[50, 60, 70, 80]kg

标准化后：身高[-1.34, -0.45, 0.45, 1.34], 体重[-1.34, -0.45, 0.45, 1.34]
归一化后：身高[0, 0.33, 0.67, 1.0], 体重[0, 0.33, 0.67, 1.0]
```

###  特征编码

#### 1. 标签编码 (Label Encoding)

**比喻：** 把文字转成数字（红→0, 绿→1, 蓝→2）

```python
from sklearn.preprocessing import LabelEncoder

# 原始数据
colors = ['红', '绿', '蓝', '红', '绿']

# 编码
encoder = LabelEncoder()
colors_encoded = encoder.fit_transform(colors)

print(colors_encoded) # [2 1 0 2 1]
```

**适用场景：** 有序类别（低、中、高）

#### 2. 独热编码 (One-Hot Encoding)

**比喻：** 把一个类别变成多个0/1列

```python
from sklearn.preprocessing import OneHotEncoder

# 原始数据
colors = [['红'], ['绿'], ['蓝'], ['红']]

# 编码
encoder = OneHotEncoder(sparse=False)
colors_encoded = encoder.fit_transform(colors)

print(colors_encoded)
# [[0. 0. 1.] # 红
# [0. 1. 0.] # 绿
# [1. 0. 0.] # 蓝
# [0. 0. 1.]] # 红
```

**适用场景：** 无序类别（颜色、城市）

**为什么需要独热编码？**
```
错误：红=0, 绿=1, 蓝=2
→ 模型会认为 蓝 > 绿 > 红（但颜色没有大小关系！）

正确：独热编码
→ 红=[0,0,1], 绿=[0,1,0], 蓝=[1,0,0]
→ 没有大小关系
```

### 特征选择

**目的：** 选择最重要的特征，去掉无用特征

**比喻：** 买房只看价格、地段、面积，不看门牌号

**方法：**

1. **过滤法：** 根据统计指标选择
```python
from sklearn.feature_selection import SelectKBest, f_classif

# 选择最好的K个特征
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
```

2. **包装法：** 根据模型性能选择
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 递归特征消除
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_selected = rfe.fit_transform(X, y)
```

3. **嵌入法：** 模型自带特征重要性
```python
from sklearn.ensemble import RandomForestClassifier

# 随机森林的特征重要性
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
```

---

## 2.6 实战练习

### 练习1：NumPy数组操作

```python
import numpy as np

# 创建一个5x5的随机矩阵
matrix = np.random.rand(5, 5)

# 任务：
# 1. 找出最大值和最小值
# 2. 计算每行的平均值
# 3. 将所有大于0.5的值替换为1，小于等于0.5的替换为0
```

### 练习2：Pandas数据处理

```python
import pandas as pd

# 创建数据
data = {
'姓名': ['张三', '李四', '王五', '赵六', '张三'],
'年龄': [25, 30, None, 35, 25],
'城市': ['北京', '上海', '深圳', '北京', '北京'],
'工资': [8000, 12000, 15000, 10000, 8000]
}
df = pd.DataFrame(data)

# 任务：
# 1. 处理缺失值（用平均年龄填充）
# 2. 删除重复行
# 3. 按城市分组，计算平均工资
# 4. 筛选出工资大于10000的记录
```

### 练习3：数据可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据
months = ['1月', '2月', '3月', '4月', '5月', '6月']
sales = [120, 150, 180, 160, 200, 220]

# 任务：
# 1. 绘制销售额折线图
# 2. 添加标题、坐标轴标签
# 3. 添加网格
# 4. 保存图片
```

---

## 2.7 本章总结

### 核心要点

1. **NumPy**
- 高效的数组运算库
- 创建数组、数组运算、索引切片
- 是机器学习的基础

2. **Pandas**
- 表格数据处理工具
- 读取数据、数据清洗、数据筛选
- 80%的时间在用Pandas处理数据

3. **Matplotlib**
- 数据可视化工具
- 折线图、柱状图、散点图、直方图
- 一图胜千言

4. **数据预处理**
- 特征缩放：标准化、归一化
- 特征编码：标签编码、独热编码
- 特征选择：选择重要特征

### 下一步

学完第二章，你应该：
- 会用NumPy处理数组
- 会用Pandas处理表格数据
- 会用Matplotlib画图
- 理解数据预处理的重要性

**接下来：**
→ 第三章：学习经典机器学习算法
→ KNN、线性回归、决策树等
→ 真正开始机器学习之旅！

---

**恭喜你完成第二章！**

现在你已经掌握了机器学习的必备工具，准备好学习算法了吗？
