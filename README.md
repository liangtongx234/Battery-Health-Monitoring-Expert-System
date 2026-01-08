# Battery Health Monitoring Expert System
# 电池健康监测专家系统

基于 CBAM-CNN-Transformer 的可解释性 SOH 预测系统

## 系统功能

### 1. 批量训练模式 (推荐)
- 运行 `批量训练模型.bat` 自动训练所有电池类型
- 自动识别 `data/` 目录下的所有电池类型
- 每种类型使用前N-1个文件训练，最后1个文件测试
- 模型自动保存到 `saved_models/`

### 2. Web界面训练模式
- 上传CSV文件进行训练
- 设置**额定容量**用于SOH计算
- 实时显示训练进度

### 3. 预测模式
- 选择已保存的模型
- 上传测试数据进行预测
- 显示SOH预测结果和性能指标

### 4. SHAP可解释性分析
- 特征重要性排序
- 可选择任意循环查看对应的SHAP瀑布图
- 退化机理雷达图分析

## 快速启动

### 方法1：批量训练所有模型
```bash
# 1. 将数据文件放入 data/ 目录
# 2. 双击运行
批量训练模型.bat
# 3. 训练完成后启动Web界面
启动系统.bat
```

### 方法2：命令行
```bash
# 批量训练
python train_all_models.py

# 启动Web界面
streamlit run app.py
```

## 数据目录结构

```
项目目录/
├── data/                          # 原始数据（用户放入）
│   ├── 2C_battery-1.csv
│   ├── 2C_battery-2.csv
│   ├── ...
│   ├── 2C_battery-8.csv
│   ├── 3C_battery-1.csv
│   ├── ...
│   ├── 3C_battery-15.csv
│   ├── R2.5_battery-1.csv
│   ├── ...
│   └── R3_battery-6.csv
├── saved_models/                  # 训练好的模型（自动生成）
│   ├── model_2C.pth
│   ├── model_3C.pth
│   ├── model_R2.5.pth
│   ├── model_R3.pth
│   └── training_summary.csv
├── app.py                         # Web应用主程序
├── train_all_models.py            # 批量训练脚本
├── 启动系统.bat                   # 启动Web界面
├── 批量训练模型.bat               # 批量训练启动
└── requirements.txt               # 依赖
```

## 数据格式

CSV文件需包含以下列：
- `CC Q` - 恒流充电容量
- `CV Q` - 恒压充电容量
- `CC charge time` - 恒流充电时间
- `CV charge time` - 恒压充电时间
- `slope 2-3`, `slope 3-4` - 电压曲线斜率
- `entropy`, `skewness`, `kurtosis` - 统计特征
- `capacity` - 容量值（目标列）

## SOH计算说明

**SOH = 实际容量 / 额定容量**

默认额定容量为 2.0 Ah，可在 `train_all_models.py` 中修改：
```python
RATED_CAPACITY = 2.0  # 修改此值
```

## 训练配置

在 `train_all_models.py` 中可修改训练参数：
```python
CONFIG = {
    'seq_length': 12,      # 序列长度
    'num_epochs': 100,     # 训练轮数
    'batch_size': 64,      # 批次大小
    'learning_rate': 0.0005,  # 学习率
    'num_heads': 8,        # 注意力头数
    'num_layers': 4,       # Transformer层数
    'dropout': 0.3,        # Dropout率
    'patience': 15         # Early stopping耐心值
}
```

## 模型文件格式

每个 .pth 文件包含：
- `model_state_dict` - 模型权重
- `scaler_X` - 特征标准化器
- `scaler_y` - 标签标准化器
- `feature_names` - 特征名称列表
- `seq_length` - 序列长度
- `input_dim` - 输入维度
- `config` - 训练配置
- `rated_capacity` - 额定容量
- `battery_type` - 电池类型
- `metrics` - 评估指标 (MAE, RMSE, R2, MAPE)

## 使用步骤

### 第一步：准备数据
将CSV文件按命名规则放入 `data/` 目录：
- `{类型}_battery-{编号}.csv`
- 例如: `2C_battery-1.csv`, `R2.5_battery-3.csv`

### 第二步：批量训练
```bash
双击 "批量训练模型.bat"
```
或
```bash
python train_all_models.py
```

### 第三步：启动Web界面
```bash
双击 "启动系统.bat"
```
或
```bash
streamlit run app.py
```

### 第四步：使用系统
1. 选择"加载模型预测"模式
2. 从下拉菜单选择已训练的模型
3. 上传测试数据
4. 查看预测结果和SHAP分析

## 部署到GitHub

训练完成后，将以下文件上传到GitHub：
```
├── app.py
├── train_all_models.py
├── requirements.txt
├── README.md
├── 启动系统.bat
├── 批量训练模型.bat
├── data/                    # 示例数据（可选）
└── saved_models/            # 训练好的模型
    ├── model_2C.pth
    ├── model_3C.pth
    ├── model_R2.5.pth
    ├── model_R3.pth
    └── training_summary.csv
```

## Streamlit Cloud部署

1. 上传到GitHub仓库
2. 访问 share.streamlit.io
3. 连接仓库，选择 app.py
4. 部署完成后获得公开链接

## 论文引用

> Section 4.5. Prototype System Implementation
> 
> 为了验证提出框架的工程可行性，我们开发了一个基于Web的原型系统。
> 该系统集成了CCT-Net推理引擎和SHAP可视化模块。
> 用户可以上传充电数据获得SOH估计和可解释诊断报告。
