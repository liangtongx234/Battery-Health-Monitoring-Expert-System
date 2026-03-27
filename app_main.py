# -*- coding: utf-8 -*
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import warnings
from datetime import datetime
import glob
import random
import copy
import shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Battery Health Monitor",
    page_icon="battery",
    layout="wide",
    initial_sidebar_state="collapsed"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

SHAP_DATA_DIR = os.path.join(BASE_DIR, 'shap_data')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SHAP_DATA_DIR, exist_ok=True)


def detect_battery_type(filepath):

    if filepath is None:
        return None
    if isinstance(filepath, str):
        basename = os.path.basename(filepath)
    else:
        basename = getattr(filepath, 'name', '')
    name_no_ext = os.path.splitext(basename)[0]
    parts = name_no_ext.split('_')
    if parts:
        prefix = parts[0]
        check_file = os.path.join(SHAP_DATA_DIR, f"shap_values_{prefix}.csv")
        if os.path.isfile(check_file):
            return prefix
    return None

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


LANG = {
    "en": {
        "title": "Battery Health Monitoring System",
        "subtitle": "CCT-Net (CBAM-CNN-Transformer) with SHAP Interpretability",
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
        "feature_importance": "Feature Importance Ranking",
        "prediction_trend": "Prediction vs Actual SOH",
        "dependency_plot": "Feature Dependency",
        "download_results": "Download Results",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R2",
        "mape": "MAPE",
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
        "about_text": "Battery SOH estimation tool using CCT-Net (CNN-CBAM-Transformer) with SHAP-based interpretability analysis.",
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
        "feature_importance": "特征重要性排序",
        "prediction_trend": "预测与实际SOH对比",
        "dependency_plot": "特征依赖关系",
        "download_results": "下载结果",
        "mae": "MAE",
        "rmse": "RMSE",
        "mape": "MAPE",
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
        "about_text": "使用CCT-Net (CNN-CBAM-Transformer)和SHAP可解释性分析的电池SOH预测工具。",
        "config": "配置",
        "using_repo": "使用仓库中的数据和模型",
        "using_demo": "使用生成的演示快速数据"
    }
}


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

    .screenshot-mode .stButton > button { display: none !important; }
    .screenshot-mode [data-testid="stSidebar"] { display: none !important; }
    .screenshot-mode .nav-bar { margin: 0 0 1.5rem 0; border-radius: 12px; }
    .screenshot-mode hr { display: none !important; }

    @media print {
        .stButton { display: none !important; }
        .nav-bar { break-inside: avoid; }
        .card { break-inside: avoid; }
    }

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
            d_model=embed_dim, nhead=num_heads, dropout=dropout,
            dim_feedforward=embed_dim * 2, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=2, dropout=dropout, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
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
        x = self.cnn_block1(x);  x = self.cbam1(x)
        x = self.cnn_block2(x);  x = self.cbam2(x)
        x = x.permute(0, 2, 1)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.attention_pool(query, x, x)
        pooled = pooled.squeeze(1)
        return self.fc_out(pooled).squeeze(1)


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
    def fit(self, X): return self
    def transform(self, X): return X
    def inverse_transform(self, X): return X


def _infer_input_dim_from_state_dict(sd: dict):
    possible_keys = ["cnn_block1.0.weight", "cnn1.0.weight"]
    for k in possible_keys:
        if k in sd:
            weight = sd[k]
            if hasattr(weight, "shape") and len(weight.shape) == 3:
                return int(weight.shape[1])
    return None


def _remap_legacy_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        if nk == "query": nk = "pool_query"
        for old_prefix, new_prefix in [
            ("cnn1.", "cnn_block1."), ("cnn2.", "cnn_block2."),
            ("pos_enc.", "positional_encoding."), ("transformer.", "transformer_encoder."),
            ("attn_pool.", "attention_pool."), ("fc.", "fc_out."),
            ("cbam1.ca.", "cbam1.channel_attention."), ("cbam1.sa.", "cbam1.spatial_attention."),
            ("cbam2.ca.", "cbam2.channel_attention."), ("cbam2.sa.", "cbam2.spatial_attention."),
        ]:
            if nk.startswith(old_prefix):
                nk = nk.replace(old_prefix, new_prefix, 1)
                break
        out[nk] = v
    return out


def load_model_file(path_or_file, device):
    ckpt = torch.load(path_or_file, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd_raw = ckpt["model_state_dict"]
    else:
        sd_raw = ckpt
        ckpt = {"model_state_dict": sd_raw}

    sd = _remap_legacy_state_dict(sd_raw)

    input_dim = None
    stored_dim = ckpt.get("input_dim")
    if stored_dim is not None:
        try: input_dim = int(stored_dim)
        except: pass
    if input_dim is None:
        fn = ckpt.get("feature_names")
        if isinstance(fn, (list, tuple)) and len(fn) > 0:
            input_dim = len(fn)
    if input_dim is None:
        input_dim = _infer_input_dim_from_state_dict(sd)
    if input_dim is None:
        raise ValueError("Cannot determine input_dim from checkpoint.")
    input_dim = int(input_dim)

    cfg = ckpt.get("config") or {}
    num_heads  = int(cfg.get("num_heads", 8))
    num_layers = int(cfg.get("num_layers", 4))
    dropout    = float(cfg.get("dropout", 0.3))

    pe_key = "positional_encoding.pe"
    max_len = 5000
    if pe_key in sd and hasattr(sd[pe_key], "shape") and len(sd[pe_key].shape) == 3:
        max_len = int(sd[pe_key].shape[1])

    model = CBAMCNNTransformer(
        input_dim=input_dim, embed_dim=128,
        num_heads=num_heads, num_layers=num_layers, dropout=dropout
    ).to(device)
    if max_len != 5000:
        model.positional_encoding = PositionalEncoding(128, max_len=max_len).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    ckpt["input_dim"] = input_dim
    ckpt["config"] = cfg
    if "seq_length" not in ckpt:       ckpt["seq_length"] = 12
    if "rated_capacity" not in ckpt:   ckpt["rated_capacity"] = 2.0
    if ckpt.get("scaler_X") is None:   ckpt["scaler_X"] = IdentityScaler()
    if ckpt.get("scaler_y") is None:   ckpt["scaler_y"] = IdentityScaler()
    if not isinstance(ckpt.get("feature_names"), (list, tuple)):
        ckpt["feature_names"] = [f"f{i}" for i in range(input_dim)]

    if missing or unexpected:
        st.warning(
            f"Model loaded (strict=False): {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys."
        )
    return model, ckpt


def generate_demo_data():
    np.random.seed(42)
    n = 200
    cycles = np.arange(1, n + 1)
    soh = 100 - 0.05 * cycles - 0.0001 * cycles ** 1.5 + np.random.normal(0, 0.3, n)
    soh = np.clip(soh, 70, 100)
    features = {
        'CC_time':           3600 * (1 - 0.002 * cycles) + np.random.normal(0, 30, n),
        'CV_time':           600 + 5 * cycles + np.random.normal(0, 20, n),
        'CC_capacity':       1.6 * soh / 100 + np.random.normal(0, 0.02, n),
        'CV_capacity':       0.4 * soh / 100 + np.random.normal(0, 0.01, n),
        'CC_slope_1':        -0.001 - 0.00001 * cycles + np.random.normal(0, 0.0001, n),
        'CC_slope_2':        -0.002 - 0.00002 * cycles + np.random.normal(0, 0.0001, n),
        'CV_slope_1':        -0.01 - 0.0001 * cycles + np.random.normal(0, 0.001, n),
        'CV_slope_2':        -0.005 - 0.00005 * cycles + np.random.normal(0, 0.0005, n),
        'temperature_avg':   25 + 0.01 * cycles + np.random.normal(0, 1, n),
        'temperature_max':   35 + 0.015 * cycles + np.random.normal(0, 1.5, n),
        'voltage_end':       4.2 - 0.0005 * cycles + np.random.normal(0, 0.01, n),
        'current_avg':       1.0 + np.random.normal(0, 0.05, n),
        'resistance_est':    0.05 + 0.0002 * cycles + np.random.normal(0, 0.002, n),
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
        shap_vals[i] = importance * np.random.randn(len(feature_names)) * 0.1
    return {
        'predictions': predictions, 'actuals': actuals,
        'feature_importance': importance, 'shap_values': shap_vals,
        'feature_names': feature_names,
        'features_scaled': np.random.randn(len(predictions), len(feature_names)),
        'shap_source': 'computed',
        'df': df, 'source': 'demo'
    }

def T(key, lang):
    return LANG.get(lang, LANG['en']).get(key, key)

def get_status(soh, lang):
    if soh >= 95:   return T('excellent', lang), 'status-excellent'
    elif soh >= 90: return T('good', lang), 'status-good'
    elif soh >= 80: return T('moderate', lang), 'status-moderate'
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

def get_rated_capacity(ckpt, user_value):
    if isinstance(ckpt, dict) and ckpt.get('rated_capacity'):
        try: return float(ckpt['rated_capacity'])
        except: pass
    return float(user_value)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0: return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100



def beeswarm_jitter(values, max_jitter=0.40, n_bins=60):

    values = np.asarray(values, dtype=float)
    n = len(values)
    offsets = np.zeros(n)
    if n == 0:
        return offsets

    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        return np.random.uniform(-max_jitter * 0.3, max_jitter * 0.3, n)

    bin_edges = np.linspace(v_min - 1e-12, v_max + 1e-12, n_bins + 1)
    bin_idx   = np.digitize(values, bin_edges) - 1
    bin_idx   = np.clip(bin_idx, 0, n_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_bins)
    max_count = counts.max() if counts.max() > 0 else 1

    for i in range(n):
        density = counts[bin_idx[i]] / max_count
        jitter_range = max_jitter * density
        offsets[i] = np.random.uniform(-jitter_range, jitter_range)

    return offsets


def plot_feature_importance(names, values):
    setup_plot()
    fig, ax = plt.subplots(figsize=(10, 6))
    idx = np.argsort(values)
    n = len(idx)
    colors = [plt.cm.Blues(0.3 + 0.5 * i / n) for i in range(n)]
    bars = ax.barh(range(n), values[idx], color=colors,
                   edgecolor=COLORS['primary'], linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, values[idx])):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10,
                color=COLORS['text'], fontweight='600')
    ax.set_yticks(range(n))
    ax.set_yticklabels([names[i] for i in idx], fontsize=10, fontweight='500')
    ax.set_xlabel('Normalized Importance', fontweight='600')
    ax.set_title('Feature Importance Ranking', fontweight='700', pad=12)
    ax.set_xlim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def plot_prediction_trend(actual, predicted, selected=None):
    setup_plot()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(actual))
    ax.plot(x, actual, color=COLORS['primary'], lw=2, label='Actual SOH',
            marker='o', ms=2, alpha=0.8)
    ax.plot(x, predicted, color=COLORS['warning'], lw=2, label='Predicted SOH',
            ls='--', alpha=0.8)
    ax.fill_between(x, actual, predicted, alpha=0.1, color=COLORS['primary'])
    if selected is not None and selected < len(actual):
        ax.axvline(selected, color=COLORS['danger'], ls=':', lw=2, alpha=0.8)
        ax.scatter([selected], [actual[selected]], color=COLORS['danger'],
                   s=120, zorder=5, edgecolors='white', lw=2)
        ax.scatter([selected], [predicted[selected]], color=COLORS['danger'],
                   s=120, zorder=5, marker='s', edgecolors='white', lw=2)
    ax.set_xlabel('Cycle', fontweight='600')
    ax.set_ylabel('SOH (%)', fontweight='600')
    ax.set_title('Predicted vs Actual SOH', fontweight='700', pad=12)
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
    colors  = [COLORS['primary']]
    labels  = ['Base']
    for i in top_idx:
        val = shap_vals[i]
        heights.append(abs(val) * 100)
        colors.append(COLORS['secondary'] if val > 0 else COLORS['danger'])
        labels.append(names[i][:12] if i < len(names) else f'F{i}')
    final = base_val + shap_vals.sum() * 100
    heights.append(final)
    colors.append(COLORS['primary_dark'])
    labels.append('Final')
    pos = list(range(len(heights)))
    ax.bar(pos, heights, color=colors, alpha=0.85, width=0.65,
           edgecolor='white', lw=1.5)
    for i, (p, h) in enumerate(zip(pos, heights)):
        if i == 0 or i == len(pos) - 1:
            ax.text(p, h + 1, f'{h:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='700')
        else:
            orig = shap_vals[top_idx[i - 1]] * 100
            ax.text(p, h + 0.5, f'{orig:+.2f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='600')
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10, fontweight='500')
    ax.set_ylabel('SOH (%)', fontweight='600')
    ax.set_title(f'SHAP Decision Decomposition {suffix}', fontweight='700', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def plot_beeswarm(names, shap_vals, feat_vals):
    """
    论文风格蜂群图：
    - 蓝色系色带  #deebf7 → #084594
    - 按特征重要性从上到下排列（最重要在最上方）
    - 淡蓝背景 + 蓝色边框
    - 使用 beeswarm_jitter 做密度感知抖动
    """
    # ── 论文配色 ──
    CMAP_BEE = LinearSegmentedColormap.from_list(
        'bee_blues', ['#deebf7', '#9ecae1', '#4292c6', '#2171b5', '#084594'])
    SPINE_COL = '#5b9bd5'
    BG_COL    = '#f7fbff'

    matplotlib.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.linewidth': 0.4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    n_feat = len(names)

    importance = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, max(5, n_feat * 0.45)))

    for rank, fi in enumerate(sorted_idx):
        sv = shap_vals[:, fi]

        if feat_vals is not None and fi < feat_vals.shape[1]:
            fv = feat_vals[:len(sv), fi]
        else:
            fv = np.random.rand(len(sv))
        fv_min, fv_max = fv.min(), fv.max()
        fv_norm = (fv - fv_min) / (fv_max - fv_min + 1e-12)

        # y 方向密度感知抖动
        y_off = beeswarm_jitter(sv, max_jitter=0.40, n_bins=60)

        ax.scatter(sv, rank + y_off,
                   c=CMAP_BEE(fv_norm),
                   s=12, alpha=0.78,
                   linewidths=0.10, edgecolors='white',
                   rasterized=True, zorder=2)


    ax.axvline(0, color='#888888', lw=0.8, ls='--', alpha=0.85, zorder=1)


    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(
        [names[i] for i in sorted_idx],
        fontsize=7.8, fontweight='bold', color='black')
    ax.set_xlabel('SHAP Value', fontsize=11, fontweight='600')
    ax.set_ylabel('Features',   fontsize=11, fontweight='600')
    ax.set_title('SHAP Value Distribution',
                 fontsize=12, fontweight='700', pad=12)


    ax.set_facecolor(BG_COL)
    ax.grid(axis='x', lw=0.4, alpha=0.5, color='#c6dbef', zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)
        sp.set_color(SPINE_COL)
    ax.tick_params(colors='black', width=0.8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.06)
    sm  = plt.cm.ScalarMappable(cmap=CMAP_BEE, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['0.0', '0.5', '1.0'],
                        fontsize=7, fontweight='bold', color='black')
    cbar.set_label('Feature Value', fontsize=7.5, fontweight='bold',
                   color='black', labelpad=3)
    cbar.outline.set_linewidth(0.7)
    cbar.outline.set_edgecolor(SPINE_COL)
    cbar.ax.tick_params(colors='black', width=0.6)

    plt.tight_layout()
    return fig


def plot_dependency(names, shap_vals, feat_vals):
    setup_plot()
    importance = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:4]
    n_plots = len(top_idx)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, fi in enumerate(top_idx):
        ax = axes[i]
        sv = shap_vals[:, fi]
        if feat_vals is not None and fi < feat_vals.shape[1]:
            fv = feat_vals[:len(sv), fi]
        else:
            fv = np.arange(len(sv))
        colors_pos = np.where(sv >= 0, COLORS['secondary'], COLORS['danger'])
        ax.scatter(fv, sv, c=colors_pos, s=30, alpha=0.55, edgecolors='white', lw=0.3)
        ax.axhline(0, color=COLORS['text_muted'], ls='--', alpha=0.4, lw=1)
        if len(fv) > 5:
            z = np.polyfit(fv, sv, 1)
            p = np.poly1d(z)
            x_sorted = np.sort(fv)
            ax.plot(x_sorted, p(x_sorted), color=COLORS['primary'], lw=2, alpha=0.7)
        fname = names[fi] if fi < len(names) else f'Feature {fi}'
        ax.set_xlabel(fname[:20], fontweight='500', fontsize=9)
        ax.set_ylabel('SHAP Value', fontweight='500', fontsize=9)
        ax.set_title(fname[:20], fontweight='600', fontsize=10, pad=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
    for i in range(n_plots, 4):
        axes[i].set_visible(False)
    fig.suptitle('SHAP Feature Dependency', fontweight='700', fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_training_curve(train_loss, val_loss):
    setup_plot()
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color=COLORS['primary'], lw=2, label='Train Loss',
            marker='o', ms=3)
    ax.plot(epochs, val_loss, color=COLORS['warning'], lw=2, label='Val Loss',
            marker='s', ms=3)
    ax.set_xlabel('Epoch', fontweight='600')
    ax.set_ylabel('Loss', fontweight='600')
    ax.set_title('Training Progress', fontweight='700', pad=12)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig



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
        pd.Series(y_scaled), seq_length=int(config['seq_length'])
    )

    val_ratio = float(config.get('val_ratio', 0.1))
    val_size  = max(1, int(val_ratio * len(dataset)))
    train_size = max(1, len(dataset) - val_size)

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)

    batch_size  = int(config['batch_size'])
    num_workers = int(config.get('num_workers', 0))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g if num_workers == 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None
    )

    model = CBAMCNNTransformer(
        input_dim=train_features.shape[1], embed_dim=128,
        num_heads=int(config.get('num_heads', 8)),
        num_layers=int(config.get('num_layers', 4)),
        dropout=float(config.get('dropout', 0.3))
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config['learning_rate']),
                            weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    num_epochs  = int(config['num_epochs'])
    es_patience = int(config.get('patience', 10))

    train_losses, val_losses = [], []
    best_val_r2  = float('-inf')
    best_state   = None
    no_improve   = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_n = 0.0, 0
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
        total_vloss, total_vn = 0.0, 0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                vloss = criterion(out, y)
                total_vloss += vloss.item() * X.size(0)
                total_vn += X.size(0)
                y_true_val.extend(y.cpu().numpy().tolist())
                y_pred_val.extend(out.cpu().numpy().tolist())
        val_loss = total_vloss / max(1, total_vn)
        val_losses.append(val_loss)
        val_r2 = r2_score(y_true_val, y_pred_val) if len(y_true_val) > 1 else 0.0
        scheduler.step(val_loss)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
        if progress_cb:
            progress_cb(epoch + 1, num_epochs, train_loss, val_loss)
        if no_improve >= es_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler_X, scaler_y, train_losses, val_losses, train_features.columns.tolist()


def predict_with_model(model, test_features, test_labels, scaler_X, scaler_y,
                       seq_length, device):
    X_scaled = scaler_X.transform(test_features.values)
    y_scaled = scaler_y.transform(test_labels.values.reshape(-1, 1)).flatten()
    dataset = BatteryDataset(
        pd.DataFrame(X_scaled, columns=test_features.columns),
        pd.Series(y_scaled), seq_length=seq_length
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
    acts  = scaler_y.inverse_transform(np.array(acts).reshape(-1, 1)).flatten()
    return preds * 100, acts * 100, X_scaled, dataset



def load_precomputed_shap(shap_data_dir: str, battery_type: str):
    """读取论文代码保存的 4 个 CSV 文件，秒级返回。"""
    files = {
        "shap":       os.path.join(shap_data_dir, f"shap_values_{battery_type}.csv"),
        "features":   os.path.join(shap_data_dir, f"feature_values_{battery_type}.csv"),
        "importance": os.path.join(shap_data_dir, f"feature_importance_{battery_type}.csv"),
        "soh":        os.path.join(shap_data_dir, f"soh_values_{battery_type}.csv"),
    }
    for key, path in files.items():
        if not os.path.isfile(path):
            print(f"[load_precomputed_shap] Missing: {path}")
            return None
    try:
        shap_df       = pd.read_csv(files["shap"])
        feat_df       = pd.read_csv(files["features"])
        importance_df = pd.read_csv(files["importance"])
        feature_names = shap_df.columns.tolist()
        shap_values   = shap_df.values
        feature_values = feat_df.values
        imp_map = dict(zip(importance_df["feature"],
                           importance_df["importance_normalized"]))
        feature_importance_norm = np.array(
            [imp_map.get(f, 0.0) for f in feature_names])
        print(f"[load_precomputed_shap] Loaded {len(shap_values)} samples, "
              f"{len(feature_names)} features")
        return feature_importance_norm, shap_values, feature_values, feature_names, 'precomputed'
    except Exception as e:
        print(f"[load_precomputed_shap] Error: {e}")
        return None


def calculate_shap_values(
    model, dataset, scaler_X, scaler_y, device,
    n_samples=500, bg_size=100, nsamples_kernel=200,
    shap_data_dir=None, battery_type=None,
):

    # ── 0. 尝试加载预计算结果 ──
    if shap_data_dir and battery_type:
        result = load_precomputed_shap(shap_data_dir, battery_type)
        if result is not None:
            return result

    np.random.seed(42)
    torch.manual_seed(42)
    seq_length    = dataset.seq_length
    feature_names = dataset.feature_names
    n_features    = len(feature_names)
    total_samples = len(dataset)
    max_samples   = min(n_samples, total_samples)
    sample_indices = np.random.choice(total_samples, max_samples, replace=False)

    X_explain, X_data_3d = [], []
    for idx in sample_indices:
        seq_X, _ = dataset[idx]
        arr = seq_X.numpy()
        X_data_3d.append(arr)
        X_explain.append(arr.flatten())
    X_explain = np.array(X_explain)
    X_data_3d = np.array(X_data_3d)


    def model_predict(x_flat):
        model.eval()
        with torch.no_grad():
            x = x_flat.reshape(-1, seq_length, n_features)
            outputs = []
            for i in range(0, x.shape[0], 16):
                xb = torch.tensor(x[i:i + 16], dtype=torch.float32).to(device)
                if xb.size(0) > 0:
                    outputs.extend(model(xb).cpu().numpy())
            return np.array(outputs) if outputs else np.zeros(x_flat.shape[0])

    bg_size_actual = min(bg_size, max_samples // 3)
    background     = X_explain[:bg_size_actual]
    print(f"[SHAP] KernelExplainer: {max_samples} samples, "
          f"{bg_size_actual} bg, nsamples={nsamples_kernel}")
    explainer   = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples_kernel)

    shap_vals_3d  = shap_values.reshape(max_samples, seq_length, n_features)
    shap_vals_agg = shap_vals_3d.mean(axis=1)
    feat_importance = np.abs(shap_vals_agg).mean(axis=0)
    mx = feat_importance.max()
    feature_importance_norm = (feat_importance / mx) if mx > 0 else feat_importance

    return feature_importance_norm, shap_vals_agg, X_data_3d, feature_names, 'computed'


def render_nav(lang):
    st.markdown(f"""
    <div class="nav-bar">
        <h1 class="nav-title">{T('title', lang)}</h1>
        <p class="nav-subtitle">{T('subtitle', lang)}</p>
    </div>
    """, unsafe_allow_html=True)


def render_results(results, selected_cycle, lang):
    preds      = np.array(results['predictions'], dtype=float)
    acts       = np.array(results['actuals'], dtype=float)
    importance = np.array(results['feature_importance'], dtype=float)
    shap_vals  = np.array(results['shap_values'], dtype=float)
    names      = results['feature_names']
    feat_scaled = results['features_scaled']

    if len(preds) == 0 or len(acts) == 0:
        st.error("No prediction results to display.")
        return

    selected_cycle = int(np.clip(selected_cycle, 0, len(preds) - 1))

    col1, col2, col3, col4 = st.columns([1.5, 0.8, 0.8, 0.8])
    with col1:
        current_soh = preds[selected_cycle]
        actual_soh  = acts[selected_cycle]
        status_text, status_class = get_status(current_soh, lang)
        st.markdown(f"""
        <div class="soh-display">
            <div class="soh-value">{current_soh:.1f}%</div>
            <div class="soh-label">{T('current_soh', lang)}</div>
            <div style="font-size:0.8rem;opacity:0.7;margin-top:0.5rem;">
                Cycle {selected_cycle+1} | Actual: {actual_soh:.1f}%
            </div>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        mae = mean_absolute_error(acts, preds)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{mae:.3f}%</div>
            <div class="metric-label">{T('mae', lang)}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        rmse = np.sqrt(mean_squared_error(acts, preds))
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{rmse:.3f}%</div>
            <div class="metric-label">{T('rmse', lang)}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        mape = mean_absolute_percentage_error(acts, preds)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{mape:.2f}%</div>
            <div class="metric-label">{T('mape', lang)}</div>
        </div>""", unsafe_allow_html=True)


    st.markdown(f'<div class="section-header">{T("prediction_trend", lang)}</div>',
                unsafe_allow_html=True)
    fig_trend = plot_prediction_trend(acts, preds, selected_cycle)
    st.pyplot(fig_trend); plt.close(fig_trend)


    st.markdown(f'<div class="section-header">{T("shap_title", lang)}</div>',
                unsafe_allow_html=True)

    shap_source = results.get('shap_source', 'computed')


    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_feature_importance(names, importance)
        st.pyplot(fig1); plt.close(fig1)
    with col2:
        fig3 = plot_beeswarm(
            names, shap_vals,
            feat_scaled[:len(shap_vals)] if feat_scaled is not None else None)
        st.pyplot(fig3); plt.close(fig3)


    base_val = float(np.mean(acts))
    if shap_source == 'computed':
        idx = min(selected_cycle, shap_vals.shape[0] - 1) \
              if shap_vals.ndim == 2 and shap_vals.shape[0] > 0 else 0
        cycle_shap = shap_vals[idx] if shap_vals.ndim == 2 else np.zeros(len(names))
        fig2 = plot_waterfall(names, cycle_shap, base_val,
                              f"(Cycle {selected_cycle + 1})")
        st.pyplot(fig2); plt.close(fig2)
    else:
        mean_shap = shap_vals.mean(axis=0) if shap_vals.ndim == 2 else np.zeros(len(names))
        fig2 = plot_waterfall(names, mean_shap, base_val,
                              "(Average across all samples)")
        st.pyplot(fig2); plt.close(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    results_df = pd.DataFrame({
        'Cycle': np.arange(1, len(preds) + 1),
        'Actual_SOH_percent': acts,
        'Predicted_SOH_percent': preds,
        'Error_percent': (acts - preds),
    })
    csv = results_df.to_csv(index=False)
    st.download_button(label=T('download_results', lang), data=csv,
                       file_name="soh_predictions.csv", mime="text/csv")


def page_demo(lang):
    st.markdown(f"""
    <div class="info-banner">
        <h3>{T('demo_title', lang)}</h3>
        <p>{T('demo_desc', lang)}</p>
    </div>""", unsafe_allow_html=True)

    data_files  = get_data_files()
    model_files = get_model_files()

    if data_files or model_files:
        c1, c2 = st.columns(2)
        with c1:
            if data_files:  st.info(f"Found {len(data_files)} data file(s)")
        with c2:
            if model_files: st.info(f"Found {len(model_files)} model(s)")

    if 'demo_results' not in st.session_state:
        if data_files and model_files:
            try:
                device = get_device()
                model, ckpt = load_model_file(model_files[0], device)
                scaler_X   = ckpt.get('scaler_X', IdentityScaler())
                scaler_y   = ckpt.get('scaler_y', IdentityScaler())
                seq_length = int(ckpt.get('seq_length', 12))
                rated_cap  = float(ckpt.get('rated_capacity', 2.0))
                input_dim  = int(ckpt.get("input_dim", 0))

                df = read_csv(data_files[0])
                if df is not None:
                    target_col = 'capacity'
                    if target_col in df.columns:
                        df['SOH'] = df[target_col] / rated_cap
                        drops = ['voltage mean', 'voltage std',
                                 'current mean', 'current std']
                        avail = [c for c in drops if c in df.columns]
                        if avail: df = df.drop(avail, axis=1)

                        feature_names = ckpt.get('feature_names', None)
                        exclude = {target_col, "SOH"}
                        cand = [c for c in df.columns
                                if c not in exclude
                                and pd.api.types.is_numeric_dtype(df[c])]
                        use_ckpt = (isinstance(feature_names, (list, tuple))
                                    and all(c in df.columns for c in feature_names))
                        if not use_ckpt:
                            if input_dim <= 0:
                                raise RuntimeError("demo missing input_dim")
                            if len(cand) < input_dim:
                                raise RuntimeError("demo not enough cols")
                            feature_names = cand[:input_dim]
                            ckpt["feature_names"] = feature_names

                        test_features = df[feature_names].replace(
                            [np.inf, -np.inf], np.nan).fillna(0)
                        test_labels = df['SOH'].replace(
                            [np.inf, -np.inf], np.nan).fillna(0)

                        preds, acts, feat_scaled, dataset = predict_with_model(
                            model, test_features, test_labels,
                            scaler_X, scaler_y, seq_length, device)

                        with st.spinner(
                            "SHAP分析计算中..." if lang == 'zh'
                            else "Computing SHAP analysis..."):
                            # Demo 页面始终实时计算，瀑布图才能跟随 cycle 变化
                            importance, shap_vals, _, _, shap_source = calculate_shap_values(
                                model, dataset, scaler_X, scaler_y, device,
                                n_samples=200, bg_size=50, nsamples_kernel=200,
                            )

                        st.session_state.demo_results = {
                            'predictions': preds, 'actuals': acts,
                            'feature_importance': importance,
                            'shap_values': shap_vals,
                            'feature_names': feature_names,
                            'features_scaled': feat_scaled,
                            'shap_source': shap_source,
                            'df': df, 'source': 'repo'
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

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        selected_cycle = st.slider(
            T('select_cycle', lang), min_value=0,
            max_value=len(results['predictions']) - 1,
            value=st.session_state.get('demo_cycle', 0),
            key='demo_cycle_slider')
        st.session_state.demo_cycle = selected_cycle

    render_results(results, selected_cycle, lang)


def page_train(lang):
    st.markdown(f'<div class="section-header">{T("train_title", lang)}</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        data_source = st.radio(
            T('data_source', lang),
            [T('load_from_repo', lang), T('upload_custom', lang)],
            horizontal=True, key='train_data_source')
        train_files, test_file = None, None
        if data_source == T('load_from_repo', lang):
            data_files = get_data_files()
            if data_files:
                train_files = st.multiselect(
                    T('upload_train', lang), data_files,
                    default=data_files[:1],
                    format_func=lambda x: os.path.basename(x))
                test_file = st.selectbox(
                    T('upload_test', lang), data_files,
                    format_func=lambda x: os.path.basename(x))
            else:
                st.warning(T('no_data', lang))
        else:
            train_files = st.file_uploader(T('upload_train', lang), type=['csv'],
                                           accept_multiple_files=True)
            test_file   = st.file_uploader(T('upload_test', lang), type=['csv'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            f'<div class="card"><div style="font-weight:600;margin-bottom:1rem;">'
            f'{T("config", lang)}</div>', unsafe_allow_html=True)
        target_col = st.text_input(T('target_col', lang), value='capacity')
        rated_cap  = st.number_input(T('rated_capacity', lang), value=2.0,
                                     min_value=0.1, max_value=1000.0, step=0.1)
        seq_length    = st.slider(T('seq_length', lang), 4, 32, 12)
        num_epochs    = st.slider(T('epochs', lang), 10, 200, 50)
        batch_size    = st.selectbox(T('batch_size', lang), [16, 32, 64, 128], index=1)
        learning_rate = st.select_slider(T('learning_rate', lang),
                                         [0.0001, 0.0005, 0.001, 0.005], value=0.001)
        model_name = st.text_input(
            T('model_name', lang),
            value=f"model_{datetime.now().strftime('%Y%m%d_%H%M')}")
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
                        if df is not None: all_data.append(df)
                    if not all_data:
                        st.error("Could not read training files"); return
                    combined = pd.concat(all_data, ignore_index=True)
                    combined['SOH'] = combined[target_col] / rated_cap
                    drops = ['voltage mean', 'voltage std',
                             'current mean', 'current std']
                    avail = [c for c in drops if c in combined.columns]
                    if avail: combined = combined.drop(avail, axis=1)
                    train_features = combined.drop([target_col, 'SOH'], axis=1)
                    train_labels   = combined['SOH']
                    train_features = train_features.replace(
                        [np.inf, -np.inf], np.nan).fillna(0)
                    train_labels = train_labels.replace(
                        [np.inf, -np.inf], np.nan).fillna(0)

                    test_df = read_csv(test_file)
                    if test_df is not None:
                        test_df['SOH'] = test_df[target_col] / rated_cap
                        if avail:
                            test_df = test_df.drop(
                                [c for c in avail if c in test_df.columns], axis=1)
                        test_features = test_df.drop([target_col, 'SOH'], axis=1)
                        test_labels   = test_df['SOH']
                        test_features = test_features.replace(
                            [np.inf, -np.inf], np.nan).fillna(0)
                        test_labels = test_labels.replace(
                            [np.inf, -np.inf], np.nan).fillna(0)

                    st.info(f"Training: {len(train_features)} samples | "
                            f"{len(train_features.columns)} features")

                    config = {
                        'seq_length': seq_length, 'num_epochs': num_epochs,
                        'batch_size': batch_size, 'learning_rate': learning_rate,
                        'num_heads': 8, 'num_layers': 4, 'dropout': 0.3
                    }

                    progress_bar = st.progress(0)
                    status_text  = st.empty()
                    def update_progress(epoch, total, tl, vl):
                        progress_bar.progress(epoch / total)
                        status_text.text(
                            f"Epoch {epoch}/{total} | Train: {tl:.6f} | Val: {vl:.6f}")

                    model, scaler_X, scaler_y, tl, vl, feature_names = train_model(
                        train_features, train_labels, config, update_progress)

                    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
                    device = get_device()
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'scaler_X': scaler_X, 'scaler_y': scaler_y,
                        'feature_names': feature_names, 'seq_length': seq_length,
                        'input_dim': len(feature_names), 'config': config,
                        'train_losses': tl, 'val_losses': vl,
                        'rated_capacity': rated_cap
                    }, model_path)

                    st.success(f"{T('training_complete', lang)} - {model_name}.pth")
                    fig = plot_training_curve(tl, vl)
                    st.pyplot(fig); plt.close()

                    preds, acts, feat_scaled, dataset = predict_with_model(
                        model, test_features, test_labels,
                        scaler_X, scaler_y, seq_length, device)

                    with st.spinner(
                        "SHAP分析计算中..." if lang == 'zh'
                        else "Computing SHAP analysis..."):
                        importance, shap_vals, _, _, shap_source = calculate_shap_values(
                            model, dataset, scaler_X, scaler_y, device,
                            n_samples=200, bg_size=50, nsamples_kernel=200,
                            # 训练新模型 → 不传 shap_data_dir，强制实时计算
                        )

                    st.session_state.train_results = {
                        'predictions': preds, 'actuals': acts,
                        'feature_importance': importance,
                        'shap_values': shap_vals,
                        'feature_names': feature_names,
                        'features_scaled': feat_scaled,
                        'shap_source': shap_source,
                    }
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if 'train_results' in st.session_state and st.session_state.train_results:
        results = st.session_state.train_results
        selected = st.slider("Select Cycle", 0,
                             len(results['predictions']) - 1, 0, key='train_cycle')
        render_results(results, selected, lang)


def page_predict(lang):
    st.markdown(f'<div class="section-header">{T("predict_title", lang)}</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        data_source = st.radio(
            T('data_source', lang),
            [T('load_from_repo', lang), T('upload_custom', lang)],
            horizontal=True, key='predict_data_source')
        test_file = None
        if data_source == T('load_from_repo', lang):
            data_files = get_data_files()
            if data_files:
                test_file = st.selectbox(
                    T('upload_test', lang), data_files,
                    format_func=lambda x: os.path.basename(x),
                    key='predict_data_select')
            else:
                st.warning(T('no_data', lang))
        else:
            test_file = st.file_uploader(T('upload_test', lang), type=['csv'],
                                         key='predict_upload')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            f'<div class="card"><div style="font-weight:600;margin-bottom:1rem;">'
            f'{T("config", lang)}</div>', unsafe_allow_html=True)
        model_source = st.radio(
            T('model_source', lang),
            [T('load_from_repo', lang), T('upload_custom', lang)],
            horizontal=True, key='predict_model_source')
        selected_model, uploaded_model = None, None
        if model_source == T('load_from_repo', lang):
            model_files = get_model_files()
            if model_files:
                selected_model = st.selectbox(
                    T('select_model', lang), model_files,
                    format_func=lambda x: os.path.basename(x))
            else:
                st.warning(T('no_model', lang))
        else:
            uploaded_model = st.file_uploader(T('upload_model', lang), type=['pth'])
        target_col = st.text_input(T('target_col', lang), value='capacity',
                                   key='predict_target')
        rated_cap = st.number_input(
            T('rated_capacity', lang), value=2.0,
            min_value=0.1, max_value=1000.0, step=0.1, key='predict_cap')
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
                    if model_source == T('load_from_repo', lang):
                        model, ckpt = load_model_file(selected_model, device)
                    else:
                        model, ckpt = load_model_file(uploaded_model, device)

                    scaler_X = ckpt.get('scaler_X', IdentityScaler())
                    scaler_y = ckpt.get('scaler_y', IdentityScaler())
                    seq_length   = int(ckpt.get('seq_length', 12))
                    rated_cap_use = get_rated_capacity(ckpt, rated_cap)

                    test_df = read_csv(test_file)
                    if test_df is None:
                        st.error("Could not read test CSV."); return
                    test_df['SOH'] = test_df[target_col] / rated_cap_use
                    drops = ['voltage mean', 'voltage std',
                             'current mean', 'current std']
                    avail = [c for c in drops if c in test_df.columns]
                    if avail: test_df = test_df.drop(avail, axis=1)

                    feature_names = ckpt.get('feature_names', None)
                    input_dim = int(ckpt.get("input_dim", 0))
                    exclude = {target_col, "SOH"}
                    cand = [c for c in test_df.columns
                            if c not in exclude
                            and pd.api.types.is_numeric_dtype(test_df[c])]
                    use_ckpt = (isinstance(feature_names, (list, tuple))
                                and all(c in test_df.columns for c in feature_names))
                    if not use_ckpt:
                        if input_dim <= 0:
                            st.error("Model missing input_dim."); return
                        if len(cand) < input_dim:
                            st.error(f"Need {input_dim} cols, found {len(cand)}."); return
                        feature_names = cand[:input_dim]
                        ckpt["feature_names"] = feature_names

                    test_features = test_df[feature_names].replace(
                        [np.inf, -np.inf], np.nan).fillna(0)
                    test_labels = test_df['SOH'].replace(
                        [np.inf, -np.inf], np.nan).fillna(0)

                    preds, acts, feat_scaled, dataset = predict_with_model(
                        model, test_features, test_labels,
                        scaler_X, scaler_y, seq_length, device)

                    with st.spinner(
                        "SHAP分析计算中..." if lang == 'zh'
                        else "Computing SHAP analysis..."):
                        # 自动检测电池类型，仅匹配到预计算文件时用缓存
                        detected_type = detect_battery_type(test_file)
                        if detected_type:
                            importance, shap_vals, _, _, shap_source = calculate_shap_values(
                                model, dataset, scaler_X, scaler_y, device,
                                n_samples=500, bg_size=100, nsamples_kernel=200,
                                shap_data_dir=SHAP_DATA_DIR,
                                battery_type=detected_type,
                            )
                        else:
                            # 非预计算数据 → 200 样本实时计算
                            importance, shap_vals, _, _, shap_source = calculate_shap_values(
                                model, dataset, scaler_X, scaler_y, device,
                                n_samples=200, bg_size=50, nsamples_kernel=200,
                            )

                    st.session_state.predict_results = {
                        'predictions': preds, 'actuals': acts,
                        'feature_importance': importance,
                        'shap_values': shap_vals,
                        'feature_names': feature_names,
                        'features_scaled': feat_scaled,
                        'shap_source': shap_source,
                    }
                    st.success(T('prediction_complete', lang))
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if 'predict_results' in st.session_state and st.session_state.predict_results:
        results = st.session_state.predict_results
        selected = st.slider("Select Cycle", 0,
                             len(results['predictions']) - 1, 0, key='predict_cycle')
        render_results(results, selected, lang)


def page_about(lang):
    st.markdown(f'<div class="section-header">{T("about_title", lang)}</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <p style="color:#5D6D7E;margin-bottom:1.5rem;font-size:1rem;line-height:1.6;">
            {T('about_text', lang)}
        </p>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h4 style="margin:0 0 0.8rem 0;color:#2C3E50;font-size:1.1rem;">
            CCT-Net Architecture</h4>
        <p style="margin:0;color:#5D6D7E;font-size:0.95rem;line-height:1.5;">
            A hybrid deep learning architecture combining CNN with CBAM and
            Transformer encoder for accurate SOH prediction.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h4 style="margin:0 0 0.8rem 0;color:#2C3E50;font-size:1.1rem;">
            SHAP Interpretability</h4>
        <p style="margin:0;color:#5D6D7E;font-size:0.95rem;line-height:1.5;">
            SHAP values quantify each feature's contribution to SOH predictions,
            revealing importance rankings and evolution across degradation stages.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h4 style="margin:0 0 0.8rem 0;color:#2C3E50;font-size:1.1rem;">
            Repository Structure</h4>
        <pre style="background:#F5F7F9;padding:1rem;border-radius:6px;font-size:0.85rem;
                    color:#2C3E50;overflow-x:auto;">
├── app_main.py         # Main Streamlit application
├── data/               # CSV data files
├── saved_models/       # Trained model checkpoints (.pth)
├── shap_data/          # Pre-computed SHAP CSV files (from paper)
│   ├── shap_values_Sim.csv
│   ├── feature_values_Sim.csv
│   ├── feature_importance_Sim.csv
│   └── soh_values_Sim.csv
└── requirements.txt
        </pre>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;margin-top:2rem;color:#7F8C9A;font-size:0.85rem;">
        <p>Battery Health Monitoring System v1.2</p>
        <p>Built with Streamlit · PyTorch · SHAP</p>
    </div>""", unsafe_allow_html=True)



def main():
    load_css()

    if 'lang' not in st.session_state:            st.session_state.lang = 'en'
    if 'page' not in st.session_state:             st.session_state.page = 'demo'
    if 'demo_cycle' not in st.session_state:       st.session_state.demo_cycle = 0
    if 'screenshot_mode' not in st.session_state:  st.session_state.screenshot_mode = False

    lang = st.session_state.lang
    page = st.session_state.page
    screenshot_mode = st.session_state.screenshot_mode

    if screenshot_mode:
        st.markdown(
            '<style>.main{padding-top:0!important;}'
            '.block-container{padding-top:1rem!important;}</style>',
            unsafe_allow_html=True)

    render_nav(lang)

    if not screenshot_mode:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if st.button(T('nav_demo', lang), key='btn_demo', use_container_width=True):
                st.session_state.page = 'demo'; st.rerun()
        with c2:
            if st.button(T('nav_train', lang), key='btn_train', use_container_width=True):
                st.session_state.page = 'train'; st.rerun()
        with c3:
            if st.button(T('nav_predict', lang), key='btn_predict', use_container_width=True):
                st.session_state.page = 'predict'; st.rerun()
        with c4:
            if st.button(T('nav_about', lang), key='btn_about', use_container_width=True):
                st.session_state.page = 'about'; st.rerun()
        with c5:
            lbl = "English" if lang == 'zh' else "中文"
            if st.button(lbl, key='btn_lang', use_container_width=True):
                st.session_state.lang = 'en' if lang == 'zh' else 'zh'; st.rerun()
        border_color = COLORS['border']
        st.markdown(
            f"<hr style='margin:1rem 0;border:none;"
            f"border-top:1px solid {border_color};'>",
            unsafe_allow_html=True)
    else:
        if st.button("Exit Screenshot Mode", key='btn_exit_screenshot'):
            st.session_state.screenshot_mode = False; st.rerun()

    if page == 'demo':      page_demo(lang)
    elif page == 'train':   page_train(lang)
    elif page == 'predict':  page_predict(lang)
    elif page == 'about':    page_about(lang)


if __name__ == "__main__":
    main()
