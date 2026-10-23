# Battery SOH Estimation with EOC Fingerprint Features

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://battery-health-monitoring-expert-system-crzz64eajisrvnvajxoont.streamlit.app/)

An interactive web-based tool for lithium-ion battery state-of-health (SOH) estimation using the CCT-Net (CNN-CBAM-Transformer) architecture with SHAP-based interpretability analysis.

## Overview

This repository provides the implementation and interactive demonstration accompanying the following article:

> Fuyuan Liang, Xing Zhou, Haiqing Yu, Tao Zhang, and Yajie Liu.
> **State-of-health estimation of lithium-ion batteries using end-of-charge features with cross-domain validation.**
> *Journal of Energy Storage*, 181, Part C (2026), 124183.

**Paper:** [DOI](https://doi.org/10.1016/j.est.2026.124183) | [Publisher page](https://www.sciencedirect.com/science/article/abs/pii/S2352152X26038478)

**Live Demo:** [https://battery-health-monitoring-expert-system-crzz64eajisrvnvajxoont.streamlit.app/](https://battery-health-monitoring-expert-system-crzz64eajisrvnvajxoont.streamlit.app/)

## Features

- **Demo Mode**: Explore the system with pre-loaded battery degradation data and trained models
- **Train Mode**: Upload custom datasets and train CCT-Net models with configurable parameters
- **Predict Mode**: Real-time SOH estimation with evaluation metrics (MAE, RMSE, MAPE)
- **SHAP Analysis**: Feature importance visualization across degradation stages

## System Architecture

![System Interface](figures/Web-based%20battery%20health%20monitoring%20system.png)

The system integrates:
- End-of-charge (EOC) fingerprint window feature extraction
- CCT-Net (CNN-CBAM-Transformer) architecture for SOH prediction
- SHAP-based interpretability analysis for feature contribution quantification

## Installation

### Option 1: Use Online Demo (Recommended)

Visit the [live demo](https://battery-health-monitoring-expert-system-crzz64eajisrvnvajxoont.streamlit.app/) directly.

### Option 2: Local Installation
```bash
git clone https://github.com/liangtongx234/Battery-Health-Monitoring-Expert-System.git
cd Battery-Health-Monitoring-Expert-System
pip install -r requirements.txt
streamlit run app_main.py
```

## Data Format

CSV files should contain the following columns:
- `CC Q` - Constant current charging capacity
- `CV Q` - Constant voltage charging capacity  
- `CC charge time` - Constant current charging time
- `CV charge time` - Constant voltage charging time
- `voltage entropy`, `voltage skewness`, `voltage kurtosis` - Voltage statistical features
- `current entropy`, `current skewness`, `current kurtosis` - Current statistical features
- `voltage slope`, `current slope` - Temporal gradient features
- `capacity` - Measured capacity (target)

## Citation

If you use this work in your research, please cite:

```bibtex
@article{liang2026soh,
  author  = {Liang, Fuyuan and Zhou, Xing and Yu, Haiqing and Zhang, Tao and Liu, Yajie},
  title   = {State-of-health estimation of lithium-ion batteries using end-of-charge features with cross-domain validation},
  journal = {Journal of Energy Storage},
  year    = {2026},
  volume  = {181},
  pages   = {124183},
  note    = {Part C},
  doi     = {10.1016/j.est.2026.124183},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S2352152X26038478}
}
```


## License

This project is licensed under the MIT License.
