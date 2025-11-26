# 第六章：Scikit-learn实战精通

> Copyright 2025 石旭乔 306开放创新实验室

> 掌握Scikit-learn的高级技巧，成为机器学习实战高手

---

## 6.1 Scikit-learn简介

### 什么是Scikit-learn？

**比喻：** Scikit-learn = 机器学习的瑞士军刀

**特点：**
- 算法丰富：分类、回归、聚类、降维等
- API统一：所有算法接口一致
- 文档完善：有大量示例和教程
- 性能优秀：底层用C/Cython优化
- 社区活跃：持续更新维护

### 核心模块

```python
from sklearn import (
    datasets,          # 数据集
    preprocessing,     # 数据预处理
    model_selection,   # 模型选择
    linear_model,      # 线性模型
    tree,             # 决策树
    ensemble,         # 集成方法
    svm,              # 支持向量机
    neighbors,        # 近邻算法
    naive_bayes,      # 朴素贝叶斯
    cluster,          # 聚类
    decomposition,    # 降维
    metrics,          # 评估指标
    pipeline          # 管道
)
```

### 统一的API设计

**所有模型都遵循相同的接口：**

```python
# 1. 创建模型
model = SomeModel(param1=value1, param2=value2)

# 2. 训练模型
model.fit(X_train, y_train)

# 3. 预测
y_pred = model.predict(X_test)

# 4. 评估
score = model.score(X_test, y_test)
```

**比喻：** 就像所有汽车都有方向盘、油门、刹车，操作方式一样

---

## 6.2 数据预处理模块

### 特征缩放

#### 1. StandardScaler（标准化）

```python
from sklearn.preprocessing import StandardScaler

# 创建缩放器
scaler = StandardScaler()

# 训练集：fit + transform
X_train_scaled = scaler.fit_transform(X_train)

# 测试集：只transform（使用训练集的参数）
X_test_scaled = scaler.transform(X_test)

# 查看参数
print('均值:', scaler.mean_)
print('标准差:', scaler.scale_)
```

**重要：** 测试集不能fit，只能transform！

```python
# 错误做法
X_test_scaled = scaler.fit_transform(X_test)  # 会导致数据泄露！

# 正确做法
X_test_scaled = scaler.transform(X_test)
```

#### 2. MinMaxScaler（归一化）

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# 自定义范围
scaler = MinMaxScaler(feature_range=(-1, 1))
```

#### 3. RobustScaler（鲁棒缩放）

**特点：** 对异常值不敏感

```python
from sklearn.preprocessing import RobustScaler

# 使用中位数和四分位数
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

**何时使用：**
- StandardScaler：数据符合正态分布
- MinMaxScaler：需要固定范围
- RobustScaler：数据有异常值

### 🏷️ 特征编码

#### 1. LabelEncoder（标签编码）

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# 编码
colors = ['红', '绿', '蓝', '红', '绿']
colors_encoded = le.fit_transform(colors)
print(colors_encoded)  # [2 1 0 2 1]

# 解码
colors_decoded = le.inverse_transform(colors_encoded)
print(colors_decoded)  # ['红' '绿' '蓝' '红' '绿']

# 查看类别
print(le.classes_)  # ['蓝' '绿' '红']
```

#### 2. OneHotEncoder（独热编码）

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

# 编码
colors = [['红'], ['绿'], ['蓝'], ['红']]
colors_encoded = ohe.fit_transform(colors)
print(colors_encoded)
# [[0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]

# 查看类别
print(ohe.categories_)
```

#### 3. OrdinalEncoder（序数编码）

**用于有序类别：**

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['低', '中', '高']])

education = [['低'], ['高'], ['中'], ['低']]
education_encoded = oe.fit_transform(education)
print(education_encoded)  # [[0.] [2.] [1.] [0.]]
```

### 数据转换

#### 1. PolynomialFeatures（多项式特征）

```python
from sklearn.preprocessing import PolynomialFeatures

# 创建多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)

X = [[1, 2], [3, 4]]
X_poly = poly.fit_transform(X)
print(X_poly)
# [[1. 2. 1. 2. 4.]  # [x1, x2, x1^2, x1*x2, x2^2]
#  [3. 4. 9. 12. 16.]]

# 查看特征名称
print(poly.get_feature_names_out(['x1', 'x2']))
```

**应用：** 线性模型拟合非线性关系

#### 2. FunctionTransformer（自定义转换）

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# 对数转换
log_transformer = FunctionTransformer(np.log1p)
X_log = log_transformer.fit_transform(X)

# 自定义函数
def custom_transform(X):
    return X ** 2

square_transformer = FunctionTransformer(custom_transform)
```

### 处理缺失值

```python
from sklearn.impute import SimpleImputer

# 用均值填充
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X)

# 其他策略
imputer = SimpleImputer(strategy='median')   # 中位数
imputer = SimpleImputer(strategy='most_frequent')  # 众数
imputer = SimpleImputer(strategy='constant', fill_value=0)  # 常数
```


---

## 6.3 模型选择模块

### 数据集划分

```python
from sklearn.model_selection import train_test_split

# 基本划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 测试集比例
    random_state=42,    # 随机种子
    stratify=y,         # 分层采样
    shuffle=True        # 是否打乱
)

# 三次划分（训练/验证/测试）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

### 交叉验证

#### 1. cross_val_score（快速交叉验证）

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

# 5折交叉验证
scores = cross_val_score(model, X, y, cv=5)
print(f'各折分数: {scores}')
print(f'平均分数: {scores.mean():.3f} (+/- {scores.std():.3f})')

# 指定评分指标
scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
```

#### 2. cross_validate（详细交叉验证）

```python
from sklearn.model_selection import cross_validate

# 多个评分指标
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

results = cross_validate(
    model, X, y, 
    cv=5, 
    scoring=scoring,
    return_train_score=True  # 返回训练集分数
)

print('测试集准确率:', results['test_accuracy'].mean())
print('测试集F1:', results['test_f1_weighted'].mean())
print('训练时间:', results['fit_time'].mean())
```

#### 3. 不同的交叉验证策略

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold,
    TimeSeriesSplit, LeaveOneOut
)

# K折
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 分层K折（保持类别比例）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 留一法
loo = LeaveOneOut()

# 使用
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # 训练和评估
```

### 超参数搜索

#### 1. GridSearchCV（网格搜索）

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# 创建网格搜索
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,          # 使用所有CPU
    verbose=2,          # 显示进度
    return_train_score=True
)

# 执行搜索
grid_search.fit(X_train, y_train)

# 查看结果
print('最佳参数:', grid_search.best_params_)
print('最佳分数:', grid_search.best_score_)
print('最佳模型:', grid_search.best_estimator_)

# 查看所有结果
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df[['params', 'mean_test_score', 'rank_test_score']].head())

# 使用最佳模型预测
y_pred = grid_search.predict(X_test)
```

#### 2. RandomizedSearchCV（随机搜索）

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 定义参数分布
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# 创建随机搜索
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=100,         # 尝试100次
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)
```

#### 3. HalvingGridSearchCV（快速网格搜索）

**原理：** 逐步淘汰表现差的参数组合

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

halving_search = HalvingGridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    factor=3,           # 每轮保留1/3
    cv=5,
    random_state=42
)

halving_search.fit(X_train, y_train)
```

---

## 6.4 Pipeline管道

### 为什么需要Pipeline？

**问题：** 数据预处理步骤繁琐，容易出错

```python
# 传统方式：容易出错
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 容易忘记只transform

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

**解决：** 使用Pipeline把步骤串联起来

```python
# Pipeline方式：简洁不易错
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### 创建Pipeline

#### 1. 基本Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# 方法1：使用Pipeline类
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])

# 方法2：使用make_pipeline（自动命名）
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    LogisticRegression()
)

# 训练
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)

# 评估
score = pipeline.score(X_test, y_test)
```

#### 2. 访问Pipeline中的步骤

```python
# 访问某个步骤
scaler = pipeline.named_steps['scaler']
pca = pipeline.named_steps['pca']

# 或者用索引
scaler = pipeline.steps[0][1]

# 查看PCA的解释方差
print(pca.explained_variance_ratio_)
```

#### 3. Pipeline与GridSearchCV结合

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格（注意命名格式：步骤名__参数名）
param_grid = {
    'pca__n_components': [5, 10, 20],
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print('最佳参数:', grid_search.best_params_)
```

### ColumnTransformer（列转换器）

**用途：** 对不同列应用不同的转换

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 假设数据有数值列和类别列
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'gender']

# 创建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# 结合Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

### 完整示例

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# 创建示例数据
data = pd.DataFrame({
    'age': [25, 30, None, 35, 40],
    'income': [50000, 60000, 70000, None, 90000],
    'city': ['北京', '上海', '北京', '深圳', '上海'],
    'gender': ['男', '女', '男', '女', '男'],
    'target': [0, 1, 0, 1, 1]
})

X = data.drop('target', axis=1)
y = data['target']

# 定义数值列和类别列
numeric_features = ['age', 'income']
categorical_features = ['city', 'gender']

# 数值列处理：填充缺失值 + 标准化
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 类别列处理：填充缺失值 + 独热编码
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 组合转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 完整Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 超参数搜索
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(X, y)

print('最佳参数:', grid_search.best_params_)
print('最佳分数:', grid_search.best_score_)
```

---

## 6.5 模型持久化

### 保存和加载模型

#### 1. 使用joblib（推荐）

```python
from joblib import dump, load

# 保存模型
dump(model, 'model.joblib')

# 加载模型
model_loaded = load('model.joblib')

# 使用加载的模型
y_pred = model_loaded.predict(X_test)
```

#### 2. 使用pickle

```python
import pickle

# 保存
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 加载
with open('model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)
```

#### 3. 保存Pipeline

```python
# Pipeline也可以直接保存
dump(pipeline, 'pipeline.joblib')

# 加载后直接使用
pipeline_loaded = load('pipeline.joblib')
y_pred = pipeline_loaded.predict(X_new)
```

### 保存预处理器

```python
# 保存scaler
dump(scaler, 'scaler.joblib')

# 在生产环境加载
scaler = load('scaler.joblib')
X_new_scaled = scaler.transform(X_new)
```

---

## 6.6 实用技巧

### 1. 快速原型开发

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 一行代码评估模型
X, y = load_iris(return_X_y=True)
scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
print(f'准确率: {scores.mean():.3f} (+/- {scores.std():.3f})')
```

### 2. 模型对比

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f'{name}: {scores.mean():.3f} (+/- {scores.std():.3f})')
```

### 3. 特征重要性分析

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 获取特征重要性
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importances)

# 可视化
import matplotlib.pyplot as plt
plt.barh(importances['feature'][:10], importances['importance'][:10])
plt.xlabel('重要性')
plt.title('Top 10 重要特征')
plt.gca().invert_yaxis()
plt.show()
```

### 4. 学习曲线

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    n_jobs=-1
)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='训练集')
plt.plot(train_sizes, val_scores.mean(axis=1), label='验证集')
plt.xlabel('训练样本数')
plt.ylabel('准确率')
plt.title('学习曲线')
plt.legend()
plt.grid(True)
plt.show()
```

### 5. 验证曲线

```python
from sklearn.model_selection import validation_curve

param_range = [1, 3, 5, 7, 10, 15, 20]
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X, y,
    param_name='max_depth',
    param_range=param_range,
    cv=5
)

# 绘制验证曲线
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_scores.mean(axis=1), label='训练集')
plt.plot(param_range, val_scores.mean(axis=1), label='验证集')
plt.xlabel('max_depth')
plt.ylabel('准确率')
plt.title('验证曲线')
plt.legend()
plt.grid(True)
plt.show()
```


---

## 6.7 常见问题与解决方案

### 问题1：数据泄露

**错误示例：**
```python
# 在划分数据前进行缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 用了全部数据！
X_train, X_test = train_test_split(X_scaled, y)
```

**正确做法：**
```python
# 先划分，再缩放
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 只transform
```

### 问题2：类别不平衡

**解决方案1：调整类别权重**
```python
from sklearn.ensemble import RandomForestClassifier

# 自动平衡
model = RandomForestClassifier(class_weight='balanced')

# 手动设置
model = RandomForestClassifier(class_weight={0: 1, 1: 10})
```

**解决方案2：重采样**
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 过采样（增加少数类）
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 欠采样（减少多数类）
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

### 问题3：特征量纲不同

**解决方案：**
```python
# 使用StandardScaler或MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 问题4：类别特征处理

**解决方案：**
```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 对类别列进行独热编码
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_columns),
    ('cat', OneHotEncoder(), categorical_columns)
])
```

### 问题5：过拟合

**解决方案：**
```python
# 1. 增加正则化
from sklearn.linear_model import Ridge, Lasso
model = Ridge(alpha=1.0)  # L2正则化
model = Lasso(alpha=0.1)  # L1正则化

# 2. 减少模型复杂度
model = DecisionTreeClassifier(max_depth=5)

# 3. 使用交叉验证
scores = cross_val_score(model, X, y, cv=5)

# 4. 增加训练数据
# 5. 使用Dropout（神经网络）
```

### 问题6：训练太慢

**解决方案：**
```python
# 1. 使用n_jobs参数
model = RandomForestClassifier(n_jobs=-1)  # 使用所有CPU

# 2. 减少数据量
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1)

# 3. 使用更简单的模型
# 4. 特征选择，减少特征数
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)
```

---

## 6.8 最佳实践

### 1. 代码组织

```python
# 推荐的项目结构
"""
project/
├── data/
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── notebooks/            # Jupyter notebooks
├── src/
│   ├── data/            # 数据处理
│   ├── features/        # 特征工程
│   ├── models/          # 模型训练
│   └── visualization/   # 可视化
├── models/              # 保存的模型
├── requirements.txt     # 依赖
└── README.md
"""
```

### 2. 配置管理

```python
# config.py
CONFIG = {
    'data': {
        'train_path': 'data/train.csv',
        'test_path': 'data/test.csv',
        'random_state': 42
    },
    'model': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    },
    'training': {
        'test_size': 0.2,
        'cv_folds': 5
    }
}
```

### 3. 实验追踪

```python
import mlflow

# 记录实验
with mlflow.start_run():
    # 记录参数
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('max_depth', 10)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 记录指标
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric('accuracy', accuracy)
    
    # 保存模型
    mlflow.sklearn.log_model(model, 'model')
```

### 4. 单元测试

```python
import unittest

class TestModel(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.model = RandomForestClassifier()
    
    def test_model_fit(self):
        """测试模型能否正常训练"""
        self.model.fit(self.X, self.y)
        self.assertTrue(hasattr(self.model, 'feature_importances_'))
    
    def test_model_predict(self):
        """测试模型能否正常预测"""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_model_score(self):
        """测试模型准确率"""
        self.model.fit(self.X, self.y)
        score = self.model.score(self.X, self.y)
        self.assertGreater(score, 0.9)

if __name__ == '__main__':
    unittest.main()
```

### 5. 文档编写

```python
def train_model(X_train, y_train, model_params=None):
    """
    训练机器学习模型
    
    Parameters:
    -----------
    X_train : array-like, shape (n_samples, n_features)
        训练特征
    y_train : array-like, shape (n_samples,)
        训练标签
    model_params : dict, optional
        模型参数
    
    Returns:
    --------
    model : sklearn estimator
        训练好的模型
    
    Examples:
    ---------
    >>> X_train, y_train = load_data()
    >>> model = train_model(X_train, y_train, {'n_estimators': 100})
    >>> y_pred = model.predict(X_test)
    """
    if model_params is None:
        model_params = {'n_estimators': 100, 'random_state': 42}
    
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    
    return model
```

---

## 6.9 完整项目模板

```python
"""
机器学习项目完整模板
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==================== 1. 数据加载 ====================
def load_data(filepath):
    """加载数据"""
    df = pd.read_csv(filepath)
    return df

# ==================== 2. 数据探索 ====================
def explore_data(df):
    """数据探索"""
    print('=== 数据基本信息 ===')
    print(df.info())
    print('\n=== 数据统计 ===')
    print(df.describe())
    print('\n=== 缺失值 ===')
    print(df.isnull().sum())
    
    # 可视化
    # ...

# ==================== 3. 数据预处理 ====================
def preprocess_data(df):
    """数据预处理"""
    # 处理缺失值
    # 处理异常值
    # 特征工程
    return df

# ==================== 4. 创建Pipeline ====================
def create_pipeline(numeric_features, categorical_features):
    """创建预处理和模型Pipeline"""
    
    # 数值特征处理
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # 类别特征处理
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 组合
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # 完整Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    
    return pipeline

# ==================== 5. 训练模型 ====================
def train_model(X_train, y_train, pipeline):
    """训练模型"""
    
    # 定义参数网格
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # 网格搜索
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print('最佳参数:', grid_search.best_params_)
    print('最佳分数:', grid_search.best_score_)
    
    return grid_search.best_estimator_

# ==================== 6. 评估模型 ====================
def evaluate_model(model, X_test, y_test):
    """评估模型"""
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 分类报告
    print('\n=== 分类报告 ===')
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('混淆矩阵')
    plt.show()
    
    return y_pred

# ==================== 7. 保存模型 ====================
def save_model(model, filepath):
    """保存模型"""
    dump(model, filepath)
    print(f'模型已保存到: {filepath}')

# ==================== 8. 主函数 ====================
def main():
    """主函数"""
    
    # 1. 加载数据
    df = load_data('data/train.csv')
    
    # 2. 数据探索
    explore_data(df)
    
    # 3. 数据预处理
    df = preprocess_data(df)
    
    # 4. 准备数据
    X = df.drop('target', axis=1)
    y = df['target']
    
    numeric_features = ['age', 'income']
    categorical_features = ['city', 'gender']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 5. 创建Pipeline
    pipeline = create_pipeline(numeric_features, categorical_features)
    
    # 6. 训练模型
    model = train_model(X_train, y_train, pipeline)
    
    # 7. 评估模型
    y_pred = evaluate_model(model, X_test, y_test)
    
    # 8. 保存模型
    save_model(model, 'models/model.joblib')
    
    print('\n项目完成！')

if __name__ == '__main__':
    main()
```

---

## 6.10 本章总结

### 核心要点

1. **Scikit-learn核心模块**
   - datasets：数据集
   - preprocessing：数据预处理
   - model_selection：模型选择
   - metrics：评估指标
   - pipeline：管道

2. **数据预处理**
   - StandardScaler：标准化
   - MinMaxScaler：归一化
   - OneHotEncoder：独热编码
   - SimpleImputer：填充缺失值

3. **模型选择**
   - train_test_split：划分数据
   - cross_val_score：交叉验证
   - GridSearchCV：网格搜索
   - RandomizedSearchCV：随机搜索

4. **Pipeline**
   - 串联多个步骤
   - 避免数据泄露
   - 简化代码
   - 便于调参

5. **最佳实践**
   - 代码组织
   - 配置管理
   - 实验追踪
   - 单元测试
   - 文档编写

### 你已经掌握

- Scikit-learn的核心功能
- 数据预处理的各种方法
- 模型选择和调优技巧
- Pipeline的使用
- 完整项目的开发流程
- 机器学习的最佳实践

### 接下来

**恭喜你完成机器学习基础课程！**

现在你可以：
1. 独立完成机器学习项目
2. 继续学习深度学习
3. 参加Kaggle竞赛
4. 阅读机器学习论文
5. 为开源项目贡献代码

**推荐学习路径：**
```
机器学习基础（已完成）
    ↓
深度学习入门
    ↓
计算机视觉 / 自然语言处理
    ↓
实战项目 / 竞赛
    ↓
论文阅读 / 前沿技术
```

---

## 推荐资源

### 书籍
- 《机器学习实战》
- 《统计学习方法》- 李航
- 《Python机器学习基础教程》
- 《Hands-On Machine Learning》

### 在线课程
- 吴恩达《机器学习》- Coursera
- Fast.ai - 实战导向
- Kaggle Learn - 免费教程

### 实践平台
- Kaggle - 数据竞赛
- 天池 - 阿里云竞赛平台
- Google Colab - 免费GPU

### 社区
- Stack Overflow
- GitHub
- Reddit r/MachineLearning
- 知乎机器学习话题

---

**恭喜你完成第六章！**

你已经成为Scikit-learn实战高手，准备好迎接深度学习的挑战了吗？

**记住：** 机器学习是一个不断学习和实践的过程，保持好奇心，持续进步！
