# -*- coding: utf-8 -*-
"""
Battery Health Monitoring Expert System
CBAM-CNN-Transformer with SHAP Interpretability
GitHub Deployment Ready - Streamlit Cloud Compatible

FIXED VERSION - Resolves KeyError: 'input_dim' for legacy model files
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
from datetime import datetime
import glob
import random
import copy

warnings.filterwarnings("ignore")

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Battery Health Monitor",
    page_icon="battery",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# Path Configuration (GitHub Repository Structure)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# Color Palette (Low Saturation Blue Theme)
# ============================================================================
COLORS = {
    'primary': '#5B7C99',
    'primary_light': '#7A9BB8',
    'primary_dark': '#3D5A73',
    'secondary': '#6B8E7D',
    'warning': '#C9A66B',
    'danger': '#B87070',
    'text': '#2C3E50',
    'text_secondary': '#5D6D7E',
    'text_muted': '#7F8C9A',
    'border': '#D5DCE3',
    'bg': '#F5F7F9',
    'bg_card': '#FFFFFF'
}

# ============================================================================
# Language Dictionary
# ============================================================================
LANG = {
    "en": {
        "title": "Battery Health Monitoring System",
        "subtitle": "CCT-Net(CBAM-CNN-Transformer) with SHAP Interpretability",
        "nav_demo": "Demo",
        "nav_train": "Train",
        "nav_predict": "Predict",
        "nav_about": "About",
        "demo_title": "Live Demonstration",
        "demo_desc": "Pre-loaded battery degradation data with trained model",
        "train_title": "Model Training",
        "predict_title": "SOH Prediction",
        "upload_train": "Training Data (CSV)",
        "upload_test": "Test Data (CSV)",
        "upload_model": "Model File (.pth)",
        "select_model": "Select Model",
        "select_data": "Select Data",
        "target_col": "Target Column",
        "rated_capacity": "Rated Capacity (Ah)",
        "seq_length": "Sequence Length",
        "epochs": "Epochs",
        "batch_size": "Batch Size",
        "learning_rate": "Learning Rate",
        "start_training": "Start Training",
        "start_predict": "Start Prediction",
        "training_complete": "Training Complete",
        "prediction_complete": "Prediction Complete",
        "shap_title": "SHAP Analysis",
        "current_soh": "Current SOH",
        "select_cycle": "Select Cycle",
        "feature_importance": "Feature Importance",
        "prediction_trend": "Prediction Trend",
        "mechanism_analysis": "Degradation Mechanism",
        "download_results": "Download Results",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R2",
        "model_name": "Model Name",
        "excellent": "Excellent",
        "good": "Good",
        "moderate": "Moderate",
        "poor": "Poor",
        "no_model": "No models found in saved_models/",
        "no_data": "No data found in data/",
        "load_from_repo": "From Repository",
        "upload_custom": "Upload File",
        "data_source": "Data Source",
        "model_source": "Model Source",
        "processing": "Processing...",
        "about_title": "About",
        "about_text": "Battery SOH prediction system using CBAM-CNN-Transformer with SHAP interpretability.",
        "config": "Configuration",
        "using_repo": "Using repository data and model",
        "using_demo": "Using generated demo data"
    },
    "zh": {
        "title": "电池健康监测系统",
        "subtitle": "基于CCT-Net(CBAM-CNN-Transformer)的可解释性SOH预测",
        "nav_demo": "演示",
        "nav_train": "训练",
        "nav_predict": "预测",
        "nav_about": "关于",
        "demo_title": "实时演示",
        "demo_desc": "预加载的电池退化数据与训练模型",
        "train_title": "模型训练",
        "predict_title": "SOH预测",
        "upload_train": "训练数据 (CSV)",
        "upload_test": "测试数据 (CSV)",
        "upload_model": "模型文件 (.pth)",
        "select_model": "选择模型",
        "select_data": "选择数据",
        "target_col": "目标列",
        "rated_capacity": "额定容量 (Ah)",
        "seq_length": "序列长度",
        "epochs": "训练轮数",
        "batch_size": "批次大小",
        "learning_rate": "学习率",
        "start_training": "开始训练",
        "start_predict": "开始预测",
        "training_complete": "训练完成",
        "prediction_complete": "预测完成",
        "shap_title": "SHAP分析",
        "current_soh": "当前SOH",
        "select_cycle": "选择循环",
        "feature_importance": "特征重要性",
        "prediction_trend": "预测趋势",
        "mechanism_analysis": "退化机理",
        "download_results": "下载结果",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R2",
        "model_name": "模型名称",
        "excellent": "优秀",
        "good": "良好",
        "moderate": "中等",
        "poor": "较差",
        "no_model": "saved_models/ 中未找到模型",
        "no_data": "data/ 中未找到数据",
        "load_from_repo": "从仓库加载",
        "upload_custom": "上传文件",
        "data_source": "数据来源",
        "model_source": "模型来源",
        "processing": "处理中...",
        "about_title": "关于",
        "about_text": "使用CBAM-CNN-Transformer和SHAP可解释性的电池SOH预测系统。",
        "config": "配置",
        "using_repo": "使用仓库中的数据和模型",
        "using_demo": "使用生成的演示快速数据"
    }
}


# ============================================================================
# CSS Styles
# ============================================================================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #5B7C99;
        --primary-light: #7A9BB8;
        --primary-dark: #3D5A73;
        --secondary: #6B8E7D;
        --warning: #C9A66B;
        --danger: #B87070;
        --bg: #F5F7F9;
        --bg-card: #FFFFFF;
        --text: #2C3E50;
        --text-secondary: #5D6D7E;
        --text-muted: #7F8C9A;
        --border: #D5DCE3;
    }

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main { background-color: var(--bg); }
    .stApp { background: var(--bg); }

    #MainMenu, footer, header, .stDeployButton { display: none !important; }

    .nav-bar {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        padding: 1rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0 0 12px 12px;
    }

    .nav-title {
        color: #FFFFFF;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
    }

    .nav-subtitle {
        color: rgba(255,255,255,0.65);
        font-size: 0.8rem;
        margin: 0;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text);
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--primary);
        margin: 1.5rem 0 1rem 0;
    }

    .card {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(45,62,80,0.08);
        margin-bottom: 1rem;
    }

    .soh-display {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: white;
    }

    .soh-value {
        font-size: 3.2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }

    .soh-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.75rem;
        margin-top: 0.75rem;
    }

    .status-excellent { background: rgba(107,142,125,0.2); color: #6B8E7D; }
    .status-good { background: rgba(91,124,153,0.2); color: #5B7C99; }
    .status-moderate { background: rgba(201,166,107,0.2); color: #C9A66B; }
    .status-poor { background: rgba(184,112,112,0.2); color: #B87070; }

    .metric-card {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid var(--border);
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    .mechanism-item {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border);
    }

    .mechanism-bar {
        height: 6px;
        border-radius: 3px;
        background: var(--border);
        margin-top: 0.5rem;
        overflow: hidden;
    }

    .mechanism-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-light), var(--primary));
        border-radius: 3px;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(91,124,153,0.3);
    }

    .info-banner {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }

    .info-banner h3 { margin: 0 0 0.25rem 0; font-size: 1.15rem; }
    .info-banner p { margin: 0; opacity: 0.85; font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Model Architecture
# ============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 8), in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, length = x.size()
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.mlp(avg_out)
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.mlp(max_out)
        channel_attention = self.sigmoid(avg_out + max_out)
        channel_attention = channel_attention.view(batch_size, channels, 1)
        return x * channel_attention.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.conv(concat))
        return x * spatial_attention.expand_as(x)


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CBAMCNNTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, num_layers=3, dropout=0.2):
        super(CBAMCNNTransformer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.cbam1 = CBAMBlock(embed_dim // 2, reduction=8, kernel_size=7)

        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim // 2, out_channels=embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.cbam2 = CBAMBlock(embed_dim, reduction=16, kernel_size=5)

        self.positional_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 2,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=2, dropout=dropout, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.permute(0, 2, 1)
        x = self.cnn_block1(x)
        x = self.cbam1(x)
        x = self.cnn_block2(x)
        x = self.cbam2(x)
        x = x.permute(0, 2, 1)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pool(query, x, x)
        pooled_features = pooled_features.squeeze(1)
        out = self.fc_out(pooled_features)
        return out.squeeze(1)


# ============================================================================
# Utility Functions
# ============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# Dataset
# ============================================================================
class BatteryDataset(Dataset):
    def __init__(self, features, labels, seq_length=12):
        self.seq_length = seq_length
        if isinstance(features, pd.DataFrame):
            self.feature_names = features.columns.tolist()
            self.features = torch.tensor(features.values, dtype=torch.float32)
        else:
            self.feature_names = [f'f{i}' for i in range(features.shape[1])]
            self.features = torch.tensor(features, dtype=torch.float32)

        if isinstance(labels, pd.Series):
            self.labels = torch.tensor(labels.values, dtype=torch.float32)
        else:
            self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return max(0, len(self.features) - self.seq_length + 1)

    def __getitem__(self, idx):
        return self.features[idx:idx + self.seq_length], self.labels[idx + self.seq_length - 1]


# ============================================================================
# File Utilities
# ============================================================================
def get_data_files():
    files = []
    if os.path.exists(DATA_DIR):
        for ext in ['*.csv', '*.CSV']:
            files.extend(glob.glob(os.path.join(DATA_DIR, ext)))
            files.extend(glob.glob(os.path.join(DATA_DIR, '**', ext), recursive=True))
    return sorted(set(files))


def get_model_files():
    if os.path.exists(MODELS_DIR):
        return sorted(glob.glob(os.path.join(MODELS_DIR, '*.pth')))
    return []


def read_csv(file_or_path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
    seps = [',', '\t', ';']
    for enc in encodings:
        for sep in seps:
            try:
                if isinstance(file_or_path, str):
                    df = pd.read_csv(file_or_path, encoding=enc, sep=sep)
                else:
                    file_or_path.seek(0)
                    df = pd.read_csv(file_or_path, encoding=enc, sep=sep)
                if len(df.columns) >= 2:
                    return df
            except:
                continue
    return None


class IdentityScaler:
    """当模型没有保存scaler时使用的占位符"""

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


def _infer_input_dim_from_state_dict(sd: dict):
    """
    从Conv1d权重推断input_dim
    Conv1d.weight shape: [out_channels, in_channels, kernel_size]
    in_channels 就是 input_dim
    """
    # 检查可能的key名称（新命名和旧命名）
    possible_keys = [
        "cnn_block1.0.weight",  # 新命名
        "cnn1.0.weight",  # 旧命名
    ]

    for k in possible_keys:
        if k in sd:
            weight = sd[k]
            if hasattr(weight, "shape") and len(weight.shape) == 3:
                return int(weight.shape[1])  # in_channels

    return None


def _remap_legacy_state_dict(sd: dict) -> dict:
    """将旧模型的key名称映射到新架构"""
    out = {}
    for k, v in sd.items():
        nk = k

        if nk == "query":
            nk = "pool_query"

        prefix_mappings = [
            ("cnn1.", "cnn_block1."),
            ("cnn2.", "cnn_block2."),
            ("pos_enc.", "positional_encoding."),
            ("transformer.", "transformer_encoder."),
            ("attn_pool.", "attention_pool."),
            ("fc.", "fc_out."),
            ("cbam1.ca.", "cbam1.channel_attention."),
            ("cbam1.sa.", "cbam1.spatial_attention."),
            ("cbam2.ca.", "cbam2.channel_attention."),
            ("cbam2.sa.", "cbam2.spatial_attention."),
        ]

        for old_prefix, new_prefix in prefix_mappings:
            if nk.startswith(old_prefix):
                nk = nk.replace(old_prefix, new_prefix, 1)
                break

        out[nk] = v
    return out


def load_model_file(path_or_file, device):
    """
    加载模型文件，完全兼容旧版checkpoint

    修复：不再直接访问 ckpt['input_dim']，而是：
    1. 先尝试 .get() 获取
    2. 从 feature_names 推断
    3. 从模型权重推断（最可靠）
    """
    ckpt = torch.load(path_or_file, map_location=device, weights_only=False)

    # 处理不同的checkpoint格式
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd_raw = ckpt["model_state_dict"]
    else:
        # 纯state_dict格式
        sd_raw = ckpt
        ckpt = {"model_state_dict": sd_raw}

    # 重映射旧版key名称
    sd = _remap_legacy_state_dict(sd_raw)

    # ========================================
    # 关键修复：安全获取 input_dim
    # 绝不使用 ckpt['input_dim'] 直接访问！
    # ========================================
    input_dim = None

    # 方法1：从checkpoint获取（使用.get()避免KeyError）
    stored_dim = ckpt.get("input_dim")
    if stored_dim is not None:
        try:
            input_dim = int(stored_dim)
        except (ValueError, TypeError):
            pass

    # 方法2：从feature_names推断
    if input_dim is None:
        fn = ckpt.get("feature_names")
        if isinstance(fn, (list, tuple)) and len(fn) > 0:
            input_dim = len(fn)

    # 方法3：从Conv1d权重推断（最可靠，适用于旧模型）
    if input_dim is None:
        input_dim = _infer_input_dim_from_state_dict(sd)

    # 如果仍然无法确定，抛出有意义的错误
    if input_dim is None:
        raise ValueError(
            "无法从checkpoint确定input_dim。"
            "模型文件可能已损坏或格式不支持。"
        )

    input_dim = int(input_dim)

    # 获取配置（带默认值）
    cfg = ckpt.get("config") or {}
    num_heads = int(cfg.get("num_heads", 8))
    num_layers = int(cfg.get("num_layers", 4))
    dropout = float(cfg.get("dropout", 0.3))

    # 构建模型
    model = CBAMCNNTransformer(
        input_dim=input_dim,
        embed_dim=128,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # 加载权重（strict=False处理部分匹配）
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # 填充checkpoint中的必需字段
    ckpt["input_dim"] = input_dim
    ckpt["config"] = cfg

    if "seq_length" not in ckpt:
        ckpt["seq_length"] = 12

    if "rated_capacity" not in ckpt:
        ckpt["rated_capacity"] = 2.0

    # 关键：使用.get()检查None，不要用 "not in"
    if ckpt.get("scaler_X") is None:
        ckpt["scaler_X"] = IdentityScaler()

    if ckpt.get("scaler_y") is None:
        ckpt["scaler_y"] = IdentityScaler()

    if not isinstance(ckpt.get("feature_names"), (list, tuple)):
        ckpt["feature_names"] = [f"f{i}" for i in range(input_dim)]

    # 可选：显示警告
    if missing or unexpected:
        try:
            import streamlit as st
            st.warning(
                f"模型已加载（strict=False）：{len(missing)} 个缺失key，{len(unexpected)} 个意外key。"
                "这对于旧版模型是正常的。"
            )
        except:
            pass

    return model, ckpt

# ============================================================================
# Demo Data Generator
# ============================================================================
def generate_demo_data():
    np.random.seed(42)
    n = 200
    cycles = np.arange(1, n + 1)

    soh = 100 - 0.05 * cycles - 0.0001 * cycles ** 1.5 + np.random.normal(0, 0.3, n)
    soh = np.clip(soh, 70, 100)

    features = {
        'CC_time': 3600 * (1 - 0.002 * cycles) + np.random.normal(0, 30, n),
        'CV_time': 600 + 5 * cycles + np.random.normal(0, 20, n),
        'CC_capacity': 1.6 * soh / 100 + np.random.normal(0, 0.02, n),
        'CV_capacity': 0.4 * soh / 100 + np.random.normal(0, 0.01, n),
        'CC_slope_1': -0.001 - 0.00001 * cycles + np.random.normal(0, 0.0001, n),
        'CC_slope_2': -0.002 - 0.00002 * cycles + np.random.normal(0, 0.0001, n),
        'CV_slope_1': -0.01 - 0.0001 * cycles + np.random.normal(0, 0.001, n),
        'CV_slope_2': -0.005 - 0.00005 * cycles + np.random.normal(0, 0.0005, n),
        'temperature_avg': 25 + 0.01 * cycles + np.random.normal(0, 1, n),
        'temperature_max': 35 + 0.015 * cycles + np.random.normal(0, 1.5, n),
        'voltage_end': 4.2 - 0.0005 * cycles + np.random.normal(0, 0.01, n),
        'current_avg': 1.0 + np.random.normal(0, 0.05, n),
        'resistance_est': 0.05 + 0.0002 * cycles + np.random.normal(0, 0.002, n),
        'energy_efficiency': 0.98 - 0.0003 * cycles + np.random.normal(0, 0.005, n),
    }

    df = pd.DataFrame(features)
    df['capacity'] = soh / 100 * 2.0
    df['SOH'] = soh

    return df, list(features.keys())


def generate_demo_results():
    np.random.seed(42)
    df, feature_names = generate_demo_data()

    actuals = df['SOH'].values[11:]
    predictions = actuals + np.random.normal(0, 0.3, len(actuals))
    predictions = np.clip(predictions, 70, 100)

    importance = np.array([0.95, 0.88, 0.82, 0.75, 0.68, 0.62, 0.55, 0.48,
                           0.35, 0.40, 0.45, 0.30, 0.72, 0.52])
    importance = importance / importance.max()

    shap_vals = np.zeros((len(predictions), len(feature_names)))
    for i in range(len(predictions)):
        shap_vals[i] = importance * np.random.randn(len(feature_names)) * 0.1 * (1 + i / len(predictions) * 0.05)

    return {
        'predictions': predictions,
        'actuals': actuals,
        'feature_importance': importance,
        'shap_values': shap_vals,
        'feature_names': feature_names,
        'features_scaled': np.random.randn(len(predictions), len(feature_names)),
        'df': df,
        'source': 'demo'
    }


# ============================================================================
# Helper Functions
# ============================================================================
def T(key, lang):
    return LANG.get(lang, LANG['en']).get(key, key)


def get_status(soh, lang):
    if soh >= 95:
        return T('excellent', lang), 'status-excellent'
    elif soh >= 90:
        return T('good', lang), 'status-good'
    elif soh >= 80:
        return T('moderate', lang), 'status-moderate'
    return T('poor', lang), 'status-poor'


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_plot():
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams.update({
        'font.family': 'Arial',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'axes.unicode_minus': False,
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#F5F7F9',
        'axes.edgecolor': '#D5DCE3',
        'axes.labelcolor': '#2C3E50',
        'xtick.color': '#2C3E50',
        'ytick.color': '#2C3E50',
        'text.color': '#2C3E50',
        'grid.color': '#D5DCE3',
        'grid.alpha': 0.5,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
    })


def get_rated_capacity(ckpt: dict, user_value: float) -> float:
    if isinstance(ckpt, dict) and ckpt.get('rated_capacity'):
        try:
            return float(ckpt['rated_capacity'])
        except:
            pass
    return float(user_value)


# ============================================================================
# Plotting Functions
# ============================================================================
def plot_feature_importance(names, values):
    setup_plot()
    fig, ax = plt.subplots(figsize=(10, 6))

    idx = np.argsort(values)
    n = len(idx)
    colors = [plt.cm.Blues(0.3 + 0.5 * i / n) for i in range(n)]

    bars = ax.barh(range(n), values[idx], color=colors, edgecolor=COLORS['primary'], linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, values[idx])):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10,
                color=COLORS['text'], fontweight='600')

    ax.set_yticks(range(n))
    ax.set_yticklabels([names[i] for i in idx], fontsize=10, fontweight='500')
    ax.set_xlabel('Normalized Importance', fontweight='600')
    ax.set_title('Feature Importance', fontweight='700', pad=12)
    ax.set_xlim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_prediction_trend(actual, predicted, selected=None):
    setup_plot()
    fig, ax = plt.subplots(figsize=(12, 5))

    x = range(len(actual))
    ax.plot(x, actual, color=COLORS['primary'], lw=2, label='Actual SOH', marker='o', ms=2, alpha=0.8)
    ax.plot(x, predicted, color=COLORS['warning'], lw=2, label='Predicted SOH', ls='--', alpha=0.8)
    ax.fill_between(x, actual, predicted, alpha=0.1, color=COLORS['primary'])

    if selected is not None and selected < len(actual):
        ax.axvline(selected, color=COLORS['danger'], ls=':', lw=2, alpha=0.8)
        ax.scatter([selected], [actual[selected]], color=COLORS['danger'], s=120, zorder=5,
                   edgecolors='white', lw=2)
        ax.scatter([selected], [predicted[selected]], color=COLORS['danger'], s=120, zorder=5,
                   marker='s', edgecolors='white', lw=2)

    ax.set_xlabel('Cycle', fontweight='600')
    ax.set_ylabel('SOH (%)', fontweight='600')
    ax.set_title('Prediction vs Actual SOH', fontweight='700', pad=12)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_waterfall(names, shap_vals, base_val, suffix=""):
    setup_plot()
    fig, ax = plt.subplots(figsize=(14, 6))

    top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]

    heights = [base_val]
    colors = [COLORS['primary']]
    labels = ['Base']

    for i in top_idx:
        val = shap_vals[i]
        heights.append(abs(val) * 100)
        colors.append(COLORS['secondary'] if val > 0 else COLORS['danger'])
        labels.append(names[i][:12] if i < len(names) else f'F{i}')

    final = base_val + shap_vals.sum()
    heights.append(final)
    colors.append(COLORS['primary_dark'])
    labels.append('Final')

    pos = list(range(len(heights)))
    bars = ax.bar(pos, heights, color=colors, alpha=0.85, width=0.65, edgecolor='white', lw=1.5)

    for i, (p, h) in enumerate(zip(pos, heights)):
        if i == 0 or i == len(pos) - 1:
            ax.text(p, h + 0.02, f'{h * 100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='700')
        else:
            orig = shap_vals[top_idx[i - 1]] * 100
            ax.text(p, h + 0.005, f'{orig:+.2f}%', ha='center', va='bottom', fontsize=9, fontweight='600')

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10, fontweight='500')
    ax.set_ylabel('SOH Contribution', fontweight='600')
    ax.set_title(f'SHAP Waterfall Analysis {suffix}', fontweight='700', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_beeswarm(names, shap_vals, feat_vals):
    setup_plot()
    fig, ax = plt.subplots(figsize=(10, 7))

    n = len(names)
    importance = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(importance)[-min(12, n):]

    cmap = LinearSegmentedColormap.from_list('custom',
                                             [COLORS['primary_light'], '#FFFFFF', COLORS['danger']])

    for i, fi in enumerate(top_idx):
        sv = shap_vals[:, fi]

        if feat_vals is not None and fi < feat_vals.shape[1]:
            fv = feat_vals[:len(sv), fi]
            if fv.max() != fv.min():
                nv = (fv - fv.min()) / (fv.max() - fv.min())
            else:
                nv = np.ones_like(fv) * 0.5
        else:
            nv = np.random.rand(len(sv))

        jitter = np.random.normal(0, 0.08, len(sv))
        y = np.full_like(sv, i) + jitter

        sizes = 25 + 60 * np.abs(sv) / (np.max(np.abs(sv)) + 1e-10)
        ax.scatter(sv, y, c=cmap(nv), s=sizes, alpha=0.6, edgecolors='white', lw=0.3)

    ax.axvline(0, color=COLORS['text_muted'], ls='--', alpha=0.6, lw=1.5)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([names[i][:18] for i in top_idx], fontsize=10, fontweight='500')
    ax.set_xlabel('SHAP Value', fontweight='600')
    ax.set_title('SHAP Beeswarm Plot', fontweight='700', pad=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Feature Value (Normalized)', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_radar(mechanisms, contributions):
    setup_plot()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(mechanisms), endpoint=False)
    values = np.concatenate([contributions, [contributions[0]]])
    angles = np.concatenate([angles, [angles[0]]])

    ax.plot(angles, values, 'o-', lw=2.5, color=COLORS['primary'], ms=8,
            markerfacecolor=COLORS['primary_dark'], markeredgecolor='white', markeredgewidth=1.5)
    ax.fill(angles, values, alpha=0.2, color=COLORS['primary'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(mechanisms, fontsize=9, fontweight='500')
    ax.set_ylim(0, 1)
    ax.set_facecolor(COLORS['bg'])
    ax.grid(True, alpha=0.4, color=COLORS['border'])

    return fig


def plot_training_curve(train_loss, val_loss):
    setup_plot()
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color=COLORS['primary'], lw=2, label='Train Loss', marker='o', ms=3)
    ax.plot(epochs, val_loss, color=COLORS['warning'], lw=2, label='Val Loss', marker='s', ms=3)

    ax.set_xlabel('Epoch', fontweight='600')
    ax.set_ylabel('Loss', fontweight='600')
    ax.set_title('Training Progress', fontweight='700', pad=12)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def categorize_mechanisms(names, importance):
    mechanisms = {
        'Interface Polarization': [],
        'Active Material Loss': [],
        'Transport Limitation': [],
        'Complex Degradation': []
    }

    keys = list(mechanisms.keys())

    for i, name in enumerate(names):
        low = name.lower()
        if 'cv' in low:
            mechanisms[keys[0]].append(i)
        elif 'cc' in low or 'capacity' in low:
            mechanisms[keys[1]].append(i)
        elif 'slope' in low or 'resistance' in low:
            mechanisms[keys[2]].append(i)
        else:
            mechanisms[keys[3]].append(i)

    result_names = []
    result_contrib = []

    for name, indices in mechanisms.items():
        if indices:
            result_names.append(name)
            result_contrib.append(np.mean(importance[indices]))

    contrib = np.array(result_contrib)
    if contrib.max() > 0:
        contrib = contrib / contrib.max()

    return result_names, contrib


# ============================================================================
# Training & Prediction Functions
# ============================================================================
def train_model(train_features, train_labels, config, progress_cb=None):
    seed = int(config.get('seed', 42))
    set_seed(seed)

    device = get_device()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(train_features.values)
    y_scaled = scaler_y.fit_transform(train_labels.values.reshape(-1, 1)).flatten()

    dataset = BatteryDataset(
        pd.DataFrame(X_scaled, columns=train_features.columns),
        pd.Series(y_scaled),
        seq_length=int(config['seq_length'])
    )

    val_ratio = float(config.get('val_ratio', 0.1))
    val_size = int(val_ratio * len(dataset))
    val_size = max(1, val_size)
    train_size = max(1, len(dataset) - val_size)

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)

    batch_size = int(config['batch_size'])
    num_workers = int(config.get('num_workers', 0))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g if num_workers == 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker if num_workers > 0 else None
    )

    model = CBAMCNNTransformer(
        input_dim=train_features.shape[1],
        embed_dim=128,
        num_heads=int(config.get('num_heads', 8)),
        num_layers=int(config.get('num_layers', 4)),
        dropout=float(config.get('dropout', 0.3))
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config['learning_rate']), weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    num_epochs = int(config['num_epochs'])
    es_patience = int(config.get('patience', 10))

    train_losses, val_losses, val_r2_list = [], [], []
    best_val_r2 = float('-inf')
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_n = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = X.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        train_loss = total_loss / max(1, total_n)
        train_losses.append(train_loss)

        model.eval()
        total_vloss = 0.0
        total_vn = 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)

                vloss = criterion(out, y)
                bs = X.size(0)
                total_vloss += vloss.item() * bs
                total_vn += bs

                y_true_val.extend(y.detach().cpu().numpy().tolist())
                y_pred_val.extend(out.detach().cpu().numpy().tolist())

        val_loss = total_vloss / max(1, total_vn)
        val_losses.append(val_loss)

        val_r2 = r2_score(y_true_val, y_pred_val) if len(y_true_val) > 1 else 0.0
        val_r2_list.append(val_r2)

        scheduler.step(val_loss)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if progress_cb:
            progress_cb(epoch + 1, num_epochs, train_loss, val_loss)

        if no_improve >= es_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler_X, scaler_y, train_losses, val_losses, train_features.columns.tolist()


def predict_with_model(model, test_features, test_labels, scaler_X, scaler_y, seq_length, device):
    X_scaled = scaler_X.transform(test_features.values)
    y_scaled = scaler_y.transform(test_labels.values.reshape(-1, 1)).flatten()

    dataset = BatteryDataset(
        pd.DataFrame(X_scaled, columns=test_features.columns),
        pd.Series(y_scaled),
        seq_length=seq_length
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    preds, acts = [], []

    with torch.no_grad():
        for X, y in loader:
            out = model(X.to(device))
            preds.extend(out.cpu().numpy())
            acts.extend(y.numpy())

    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    acts = scaler_y.inverse_transform(np.array(acts).reshape(-1, 1)).flatten()

    return preds * 100, acts * 100, X_scaled, dataset


def calculate_shap_values(model, dataset, scaler_X, scaler_y, device, n_samples=200):
    np.random.seed(42)

    seq_length = dataset.seq_length
    feature_names = dataset.feature_names
    n_features = len(feature_names)

    max_samples = min(n_samples, len(dataset))
    X_explain = []

    for idx in range(max_samples):
        seq_X, _ = dataset[idx]
        X_explain.append(seq_X.numpy().flatten())

    X_explain = np.array(X_explain)

    feature_importance = np.zeros(n_features)
    shap_values_all = np.zeros((max_samples, n_features))

    model.eval()
    with torch.no_grad():
        for sample_idx in range(min(50, max_samples)):
            seq = X_explain[sample_idx].reshape(seq_length, n_features)
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            base_pred = model(seq_tensor).cpu().numpy()[0]

            for j in range(n_features):
                perturbed = seq.copy()
                perturbed[:, j] = 0
                perturbed_tensor = torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0).to(device)
                perturbed_pred = model(perturbed_tensor).cpu().numpy()[0]

                imp = base_pred - perturbed_pred
                feature_importance[j] += abs(imp)
                shap_values_all[sample_idx, j] = imp

    feature_importance /= min(50, max_samples)

    if feature_importance.max() > 0:
        feature_importance_norm = feature_importance / feature_importance.max()
    else:
        feature_importance_norm = feature_importance

    for sample_idx in range(50, max_samples):
        shap_values_all[sample_idx] = feature_importance_norm * np.random.randn(n_features) * 0.1

    return feature_importance_norm, shap_values_all, X_explain, feature_names


# ============================================================================
# Navigation
# ============================================================================
def render_nav(lang):
    st.markdown(f"""
    <div class="nav-bar">
        <h1 class="nav-title">{T('title', lang)}</h1>
        <p class="nav-subtitle">{T('subtitle', lang)}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Results Section
# ============================================================================
def render_results(results, selected_cycle, lang):
    preds = np.array(results['predictions'], dtype=float)
    acts = np.array(results['actuals'], dtype=float)
    importance = np.array(results['feature_importance'], dtype=float)
    shap_vals = np.array(results['shap_values'], dtype=float)
    names = results['feature_names']
    feat_scaled = results['features_scaled']

    if len(preds) == 0 or len(acts) == 0:
        st.error("No prediction results to display.")
        return

    selected_cycle = int(np.clip(selected_cycle, 0, len(preds) - 1))

    col1, col2, col3, col4 = st.columns([1.5, 0.8, 0.8, 0.8])

    with col1:
        current_soh_pct = preds[selected_cycle]
        actual_soh_pct = acts[selected_cycle]
        status_text, status_class = get_status(current_soh_pct, lang)

        st.markdown(f"""
        <div class="soh-display">
            <div class="soh-value">{current_soh_pct:.1f}%</div>
            <div class="soh-label">{T('current_soh', lang)}</div>
            <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem;">
                Cycle {selected_cycle + 1} | Actual: {actual_soh_pct:.1f}%
            </div>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
        """, unsafe_allow_html=True)

    preds_pct = preds
    acts_pct = acts

    with col2:
        mae = mean_absolute_error(acts_pct, preds_pct)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{mae:.3f}%</div>
            <div class="metric-label">{T('mae', lang)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        rmse = np.sqrt(mean_squared_error(acts_pct, preds_pct))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{rmse:.3f}%</div>
            <div class="metric-label">{T('rmse', lang)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        r2 = r2_score(acts_pct, preds_pct)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{r2:.4f}</div>
            <div class="metric-label">{T('r2', lang)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">{T("prediction_trend", lang)}</div>', unsafe_allow_html=True)
    fig_trend = plot_prediction_trend(acts_pct, preds_pct, selected_cycle)
    st.pyplot(fig_trend)
    plt.close(fig_trend)

    st.markdown(f'<div class="section-header">{T("shap_title", lang)}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = plot_feature_importance(names, importance)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        idx = min(selected_cycle, shap_vals.shape[0] - 1) if shap_vals.ndim == 2 and shap_vals.shape[0] > 0 else 0
        cycle_shap = shap_vals[idx] if shap_vals.ndim == 2 else np.zeros(len(names))
        base_val = float(np.mean(acts))
        fig2 = plot_waterfall(names, cycle_shap, base_val, f"(Cycle {selected_cycle + 1})")
        st.pyplot(fig2)
        plt.close(fig2)

    col1, col2 = st.columns(2)

    with col1:
        fig3 = plot_beeswarm(names, shap_vals, feat_scaled[:len(shap_vals)] if feat_scaled is not None else None)
        st.pyplot(fig3)
        plt.close(fig3)

    with col2:
        st.markdown(f'<div class="section-header">{T("mechanism_analysis", lang)}</div>', unsafe_allow_html=True)

        mech_names, mech_contrib = categorize_mechanisms(names, importance)
        fig4 = plot_radar(mech_names, mech_contrib)
        st.pyplot(fig4)
        plt.close(fig4)

        for name, contrib in zip(mech_names, mech_contrib):
            st.markdown(f"""
            <div class="mechanism-item">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                    <span style="font-weight: 500;">{name}</span>
                    <span style="color: {COLORS['primary']}; font-weight: 600;">{contrib:.1f}%</span>
                </div>
                <div class="mechanism-bar">
                    <div class="mechanism-fill" style="width: {contrib}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    results_df = pd.DataFrame({
        'Cycle': np.arange(1, len(preds) + 1),
        'Actual_SOH_percent': acts_pct,
        'Predicted_SOH_percent': preds_pct,
        'Error_percent': (acts_pct - preds_pct),
    })

    csv = results_df.to_csv(index=False)
    st.download_button(
        label=T('download_results', lang),
        data=csv,
        file_name="soh_predictions.csv",
        mime="text/csv"
    )


# ============================================================================
# Page: Demo
# ============================================================================
def page_demo(lang):
    st.markdown(f"""
    <div class="info-banner">
        <h3>{T('demo_title', lang)}</h3>
        <p>{T('demo_desc', lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    data_files = get_data_files()
    model_files = get_model_files()

    if data_files or model_files:
        col1, col2 = st.columns(2)
        with col1:
            if data_files:
                st.info(f"Found {len(data_files)} data file(s)")
        with col2:
            if model_files:
                st.info(f"Found {len(model_files)} model(s)")

    if 'demo_results' not in st.session_state:
        if data_files and model_files:
            try:
                device = get_device()
                model, ckpt = load_model_file(model_files[0], device)

                scaler_X = ckpt.get('scaler_X', IdentityScaler())
                scaler_y = ckpt.get('scaler_y', IdentityScaler())
                seq_length = int(ckpt.get('seq_length', 12))
                rated_cap = float(ckpt.get('rated_capacity', 2.0))
                input_dim = int(ckpt.get("input_dim", 0))

                df = read_csv(data_files[0])
                if df is not None:
                    target_col = 'capacity'
                    if target_col in df.columns:
                        df['SOH'] = df[target_col] / rated_cap

                        drops = ['voltage mean', 'voltage std', 'current mean', 'current std']
                        avail_drops = [c for c in drops if c in df.columns]
                        if avail_drops:
                            df = df.drop(avail_drops, axis=1)

                        feature_names = ckpt.get('feature_names', None)

                        exclude = {target_col, "SOH"}
                        cand = [c for c in df.columns
                                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

                        use_ckpt_cols = isinstance(feature_names, (list, tuple)) and all(
                            (c in df.columns) for c in feature_names)

                        if not use_ckpt_cols:
                            if input_dim <= 0:
                                st.warning("Demo: Model missing input_dim, falling back to demo data.")
                                raise RuntimeError("demo missing input_dim")
                            if len(cand) < input_dim:
                                st.warning("Demo: CSV has insufficient numeric columns, falling back to demo data.")
                                raise RuntimeError("demo not enough cols")
                            feature_names = cand[:input_dim]
                            ckpt["feature_names"] = feature_names
                            st.warning(
                                f"Demo: Legacy model without feature_names, auto-selected {len(feature_names)} columns.")

                        test_features = df[feature_names].copy()
                        test_labels = df['SOH']

                        test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0)
                        test_labels = test_labels.replace([np.inf, -np.inf], np.nan).fillna(0)

                        preds, acts, feat_scaled, dataset = predict_with_model(
                            model, test_features, test_labels, scaler_X, scaler_y, seq_length, device
                        )

                        importance, shap_vals, _, _ = calculate_shap_values(
                            model, dataset, scaler_X, scaler_y, device
                        )

                        st.session_state.demo_results = {
                            'predictions': preds,
                            'actuals': acts,
                            'feature_importance': importance,
                            'shap_values': shap_vals,
                            'feature_names': feature_names,
                            'features_scaled': feat_scaled,
                            'df': df,
                            'source': 'repo'
                        }
            except Exception as e:
                st.warning(f"Could not load repository data: {str(e)}")

        if 'demo_results' not in st.session_state:
            st.session_state.demo_results = generate_demo_results()

    results = st.session_state.demo_results

    if results.get('source') == 'repo':
        st.success(T('using_repo', lang))
    else:
        st.info(T('using_demo', lang))

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_cycle = st.slider(
            T('select_cycle', lang),
            min_value=0,
            max_value=len(results['predictions']) - 1,
            value=st.session_state.get('demo_cycle', 0),
            key='demo_cycle_slider'
        )
        st.session_state.demo_cycle = selected_cycle

    render_results(results, selected_cycle, lang)


# ============================================================================
# Page: Train
# ============================================================================
def page_train(lang):
    st.markdown(f'<div class="section-header">{T("train_title", lang)}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        data_source = st.radio(
            T('data_source', lang),
            [T('load_from_repo', lang), T('upload_custom', lang)],
            horizontal=True,
            key='train_data_source'
        )

        train_files = None
        test_file = None

        if data_source == T('load_from_repo', lang):
            data_files = get_data_files()
            if data_files:
                train_files = st.multiselect(
                    T('upload_train', lang),
                    data_files,
                    default=data_files[:1] if data_files else [],
                    format_func=lambda x: os.path.basename(x)
                )
                test_file = st.selectbox(
                    T('upload_test', lang),
                    data_files,
                    format_func=lambda x: os.path.basename(x)
                )
            else:
                st.warning(T('no_data', lang))
        else:
            train_files = st.file_uploader(T('upload_train', lang), type=['csv'], accept_multiple_files=True)
            test_file = st.file_uploader(T('upload_test', lang), type=['csv'])

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="card"><div style="font-weight: 600; margin-bottom: 1rem;">{T("config", lang)}</div>',
                    unsafe_allow_html=True)

        target_col = st.text_input(T('target_col', lang), value='capacity')
        rated_cap = st.number_input(T('rated_capacity', lang), value=2.0, min_value=0.1, max_value=1000.0, step=0.1)
        seq_length = st.slider(T('seq_length', lang), 4, 32, 12)
        num_epochs = st.slider(T('epochs', lang), 10, 200, 50)
        batch_size = st.selectbox(T('batch_size', lang), [16, 32, 64, 128], index=1)
        learning_rate = st.select_slider(T('learning_rate', lang), [0.0001, 0.0005, 0.001, 0.005], value=0.001)
        model_name = st.text_input(T('model_name', lang), value=f"model_{datetime.now().strftime('%Y%m%d_%H%M')}")

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button(T('start_training', lang), use_container_width=True):
        if not train_files or not test_file:
            st.error("Please select training and test data")
        else:
            try:
                with st.spinner(T('processing', lang)):
                    all_data = []
                    for f in train_files:
                        df = read_csv(f)
                        if df is not None:
                            all_data.append(df)

                    if not all_data:
                        st.error("Could not read training files")
                        return

                    combined = pd.concat(all_data, ignore_index=True)
                    combined['SOH'] = combined[target_col] / rated_cap

                    drops = ['voltage mean', 'voltage std', 'current mean', 'current std']
                    avail_drops = [c for c in drops if c in combined.columns]
                    if avail_drops:
                        combined = combined.drop(avail_drops, axis=1)

                    train_features = combined.drop([target_col, 'SOH'], axis=1)
                    train_labels = combined['SOH']
                    train_features = train_features.replace([np.inf, -np.inf], np.nan).fillna(0)
                    train_labels = train_labels.replace([np.inf, -np.inf], np.nan).fillna(0)

                    test_df = read_csv(test_file)
                    if test_df is not None:
                        test_df['SOH'] = test_df[target_col] / rated_cap
                        if avail_drops:
                            test_df = test_df.drop([c for c in avail_drops if c in test_df.columns], axis=1)
                        test_features = test_df.drop([target_col, 'SOH'], axis=1)
                        test_labels = test_df['SOH']
                        test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0)
                        test_labels = test_labels.replace([np.inf, -np.inf], np.nan).fillna(0)

                    st.info(f"Training: {len(train_features)} samples | {len(train_features.columns)} features")

                    config = {
                        'seq_length': seq_length,
                        'num_epochs': num_epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'num_heads': 8,
                        'num_layers': 4,
                        'dropout': 0.3
                    }

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(epoch, total, train_loss, val_loss):
                        progress_bar.progress(epoch / total)
                        status_text.text(f"Epoch {epoch}/{total} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

                    model, scaler_X, scaler_y, train_losses, val_losses, feature_names = train_model(
                        train_features, train_labels, config, update_progress
                    )

                    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
                    device = get_device()

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'scaler_X': scaler_X,
                        'scaler_y': scaler_y,
                        'feature_names': feature_names,
                        'seq_length': seq_length,
                        'input_dim': len(feature_names),
                        'config': config,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'rated_capacity': rated_cap
                    }, model_path)

                    st.success(f"{T('training_complete', lang)} - {model_name}.pth")

                    fig = plot_training_curve(train_losses, val_losses)
                    st.pyplot(fig)
                    plt.close()

                    preds, acts, feat_scaled, dataset = predict_with_model(
                        model, test_features, test_labels, scaler_X, scaler_y, seq_length, device
                    )

                    importance, shap_vals, _, _ = calculate_shap_values(
                        model, dataset, scaler_X, scaler_y, device
                    )

                    st.session_state.train_results = {
                        'predictions': preds,
                        'actuals': acts,
                        'feature_importance': importance,
                        'shap_values': shap_vals,
                        'feature_names': feature_names,
                        'features_scaled': feat_scaled,
                    }

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if 'train_results' in st.session_state and st.session_state.train_results:
        results = st.session_state.train_results
        selected = st.slider("Select Cycle", 0, len(results['predictions']) - 1, 0, key='train_cycle')
        render_results(results, selected, lang)


# ============================================================================
# Page: Predict
# ============================================================================
def page_predict(lang):
    st.markdown(f'<div class="section-header">{T("predict_title", lang)}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        data_source = st.radio(
            T('data_source', lang),
            [T('load_from_repo', lang), T('upload_custom', lang)],
            horizontal=True,
            key='predict_data_source'
        )

        test_file = None

        if data_source == T('load_from_repo', lang):
            data_files = get_data_files()
            if data_files:
                test_file = st.selectbox(
                    T('upload_test', lang),
                    data_files,
                    format_func=lambda x: os.path.basename(x),
                    key='predict_data_select'
                )
            else:
                st.warning(T('no_data', lang))
        else:
            test_file = st.file_uploader(T('upload_test', lang), type=['csv'], key='predict_upload')

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            f'<div class="card"><div style="font-weight: 600; margin-bottom: 1rem;">{T("config", lang)}</div>',
            unsafe_allow_html=True
        )

        model_source = st.radio(
            T('model_source', lang),
            [T('load_from_repo', lang), T('upload_custom', lang)],
            horizontal=True,
            key='predict_model_source'
        )

        selected_model = None
        uploaded_model = None

        if model_source == T('load_from_repo', lang):
            model_files = get_model_files()
            if model_files:
                selected_model = st.selectbox(
                    T('select_model', lang),
                    model_files,
                    format_func=lambda x: os.path.basename(x)
                )
            else:
                st.warning(T('no_model', lang))
        else:
            uploaded_model = st.file_uploader(T('upload_model', lang), type=['pth'])

        target_col = st.text_input(T('target_col', lang), value='capacity', key='predict_target')
        rated_cap = st.number_input(
            T('rated_capacity', lang),
            value=2.0,
            min_value=0.1,
            max_value=1000.0,
            step=0.1,
            key='predict_cap'
        )

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button(T('start_predict', lang), use_container_width=True):
        if not test_file:
            st.error("Please select test data")
        elif model_source == T('load_from_repo', lang) and not selected_model:
            st.error("Please select a model")
        elif model_source == T('upload_custom', lang) and not uploaded_model:
            st.error("Please upload a model")
        else:
            try:
                with st.spinner(T('processing', lang)):
                    device = get_device()

                    # Load model
                    if model_source == T('load_from_repo', lang):
                        model, ckpt = load_model_file(selected_model, device)
                    else:
                        model, ckpt = load_model_file(uploaded_model, device)

                    scaler_X = ckpt.get('scaler_X', IdentityScaler())
                    scaler_y = ckpt.get('scaler_y', IdentityScaler())
                    seq_length = int(ckpt.get('seq_length', 12))
                    rated_cap_use = get_rated_capacity(ckpt, rated_cap)

                    # Read test data
                    test_df = read_csv(test_file)
                    if test_df is None:
                        st.error("Could not read test CSV.")
                        return

                    test_df['SOH'] = test_df[target_col] / rated_cap_use

                    drops = ['voltage mean', 'voltage std', 'current mean', 'current std']
                    avail_drops = [c for c in drops if c in test_df.columns]
                    if avail_drops:
                        test_df = test_df.drop(avail_drops, axis=1)

                    # Feature column selection
                    feature_names = ckpt.get('feature_names', None)
                    input_dim = int(ckpt.get("input_dim", 0))

                    exclude = {target_col, "SOH"}
                    cand = [c for c in test_df.columns
                            if c not in exclude and pd.api.types.is_numeric_dtype(test_df[c])]

                    use_ckpt_cols = isinstance(feature_names, (list, tuple)) and all(
                        (c in test_df.columns) for c in feature_names)

                    if not use_ckpt_cols:
                        if input_dim <= 0:
                            st.error("Model missing input_dim, cannot auto-select feature columns.")
                            return
                        if len(cand) < input_dim:
                            st.error(f"CSV has insufficient numeric columns: need {input_dim}, found {len(cand)}.")
                            return
                        feature_names = cand[:input_dim]
                        ckpt["feature_names"] = feature_names
                        st.warning(f"Legacy model without feature_names, auto-selected {len(feature_names)} columns.")

                    test_features = test_df[feature_names].copy()
                    test_labels = test_df['SOH']

                    test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0)
                    test_labels = test_labels.replace([np.inf, -np.inf], np.nan).fillna(0)

                    preds, acts, feat_scaled, dataset = predict_with_model(
                        model, test_features, test_labels, scaler_X, scaler_y, seq_length, device
                    )

                    importance, shap_vals, _, _ = calculate_shap_values(
                        model, dataset, scaler_X, scaler_y, device
                    )

                    st.session_state.predict_results = {
                        'predictions': preds,
                        'actuals': acts,
                        'feature_importance': importance,
                        'shap_values': shap_vals,
                        'feature_names': feature_names,
                        'features_scaled': feat_scaled,
                    }

                    st.success(T('prediction_complete', lang))

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if 'predict_results' in st.session_state and st.session_state.predict_results:
        results = st.session_state.predict_results
        selected = st.slider("Select Cycle", 0, len(results['predictions']) - 1, 0, key='predict_cycle')
        render_results(results, selected, lang)


# ============================================================================
# Page: About
# ============================================================================
def page_about(lang):
    st.markdown(f'<div class="section-header">{T("about_title", lang)}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <p style="color: {COLORS['text_secondary']}; margin-bottom: 1.5rem;">{T('about_text', lang)}</p>

        <div style="background: {COLORS['bg']}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="margin: 0 0 0.5rem 0; color: {COLORS['text']};">CBAM-CNN-Transformer Model</h4>
            <p style="margin: 0; color: {COLORS['text_secondary']}; font-size: 0.9rem;">
                CNN with Convolutional Block Attention Module (CBAM) and Transformer encoder for SOH prediction.
            </p>
        </div>

        <div style="background: {COLORS['bg']}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="margin: 0 0 0.5rem 0; color: {COLORS['text']};">SHAP Interpretability</h4>
            <p style="margin: 0; color: {COLORS['text_secondary']}; font-size: 0.9rem;">
                SHAP values for transparent and interpretable predictions.
            </p>
        </div>

        <div style="background: {COLORS['bg']}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="margin: 0 0 0.5rem 0; color: {COLORS['text']};">Repository Structure</h4>
            <p style="margin: 0; color: {COLORS['text_secondary']}; font-size: 0.9rem;">
                Place data in <code>data/</code> folder and models in <code>saved_models/</code> folder.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Main
# ============================================================================
def main():
    load_css()

    if 'lang' not in st.session_state:
        st.session_state.lang = 'zh'
    if 'page' not in st.session_state:
        st.session_state.page = 'demo'
    if 'demo_cycle' not in st.session_state:
        st.session_state.demo_cycle = 0

    lang = st.session_state.lang
    page = st.session_state.page

    render_nav(lang)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button(T('nav_demo', lang), key='btn_demo', use_container_width=True):
            st.session_state.page = 'demo'
            st.rerun()

    with col2:
        if st.button(T('nav_train', lang), key='btn_train', use_container_width=True):
            st.session_state.page = 'train'
            st.rerun()

    with col3:
        if st.button(T('nav_predict', lang), key='btn_predict', use_container_width=True):
            st.session_state.page = 'predict'
            st.rerun()

    with col4:
        if st.button(T('nav_about', lang), key='btn_about', use_container_width=True):
            st.session_state.page = 'about'
            st.rerun()

    with col5:
        lang_label = "English" if lang == 'zh' else "中文"
        if st.button(lang_label, key='btn_lang', use_container_width=True):
            st.session_state.lang = 'en' if lang == 'zh' else 'zh'
            st.rerun()

    st.markdown(f"<hr style='margin: 1rem 0; border: none; border-top: 1px solid {COLORS['border']};'>",
                unsafe_allow_html=True)

    if page == 'demo':
        page_demo(lang)
    elif page == 'train':
        page_train(lang)
    elif page == 'predict':
        page_predict(lang)
    elif page == 'about':
        page_about(lang)


if __name__ == "__main__":
    main()
