# Time Series Anomaly Detection Evaluation Framework 

:triangular_flag_on_post:**News** (2024.03) Our library is now fully functional!

:triangular_flag_on_post:**News** (2024.02) Scoring and thresholding functions are added!

:triangular_flag_on_post:**News** (2024.02) TAD evaluation metrics are added! 

:triangular_flag_on_post:**News** (2024.01) State-of-the-art DL models are added!

**Scoring functions** ☑ means that their codes have already been included in this repo. [[Code]](https://github.com/onatucsd/timeseries-intel/blob/main/utils/scoring.py)
- [x] **ML-based Scoring**
- [x] **Mean over Channels**
- [x] **Normalized Errors**
- [x] **Gauss-S**
- [x] **Gauss-D**

**Thresholding functions** ☑ means that their codes have already been included in this repo. [[Code]](https://github.com/onatucsd/timeseries-intel/blob/main/utils/thresholding.py)
- [x] **Top-k**
- [x] **Validation Percentile**
- [x] **Best F-Score**
- [x] **Tail-p**
- [x] **Dynamic Thresholding**
- [x] **Streaming Peak over Threshold (SPOT)**
- [x] **Streaming Peak over Threshold with Drift (DSPOT)**

**Evaluation metrics** ☑ means that their codes have already been included in this repo. [[Code]](https://github.com/onatucsd/timeseries-intel/blob/main/utils/eval.py)
- [x] **PR-AUC**
- [x] **ROC-AUC**
- [x] **F-score**
- [x] **F-PA**
- [x] **F-PAK**
- [x] **F-AUC**
- [x] **ROC-K-AUC**
- [x] **F-Composite**

**State-of-the-art models** ☑ means that their codes have already been included in this repo. [[Code]](https://github.com/onatucsd/timeseries-intel/tree/main/models)
- [x] **GPT2** - One Fits All: Power General Time Series Analysis by Pretrained LM [[NeurIPS 2024]](https://arxiv.org/abs/2302.11939)
- [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625)
- [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)
- [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)
- [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol)
- [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq)
- [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) 
- [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186)
- [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) 
- [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) 
- [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132)
- [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB)
- [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) 
- [x] **LSTM-AE** - LSTM-based encoder-decoder for multi-sensor anomaly detection [[ICML 2016]](https://arxiv.org/pdf/1607.00148.pdf)
- [x] **Traditional ML Methods: OCSVM, and LOF** - One-class support vector machine, and Local Outlier Factor

**Simple Baselines** ☑ means that their codes have already been included in this repo.
- [x] **Random** - Random anomaly scores
- [x] **Input copy** - Zero output-based anomaly scores
- [x] **Untrained model** - Untrained model anomaly scores
- [x] **Sensor Range** - The range of sensor values observed
- [x] **L2-norm** - Magnitude of the observed time stamp
- [x] **NN-distance** - Nearest neighbor distance to the normal training data
- [x] **PCA reconstruction** - Outlier detection on a lower dimensional linear approximation

## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing). Then place the downloaded data in the folder`./dataset`. Here are the datasets: SMD, MSL, SMAP, SWaT, WADI, and PSM. 

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
bash ./scripts/anomaly_detection/SWAT/TimesNet.sh
```

4. Some important anomaly detection script configurations:
- sc_function: scoring function &rarr; {ML, MoC, NE, GS, GD} 
- th_idp: thresholding independent evaluation &rarr; {0: dependent, 1: independent} 
- th_function: thresholding function &rarr; {Top-k, Best-F, ValiPer, Tail-p, Dyn-Th, SPOT, DSPOT}
- ratio: test ground truth anomaly access ratio &rarr; {<100: partial access, 100: full access}  
- baseline: TAD baseline &rarr; {0: no baseline, 1: Baseline 1, 2: Baseline 2, 3: Baseline 3}  

5. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

## Contact
If you have any questions or suggestions, feel free to contact:

- Onat Gungor (ogungor@ucsd.edu)
- Amanda Rios (amanda.rios@intel.com)

Or describe it in Issues.

## Acknowledgement

This library is constructed based on the following repositories:

- [Time Series Library (TSlib)](https://github.com/thuml/Time-Series-Library)
- [Diffusion AE](https://github.com/fbrad/DiffusionAE)
- [MVTS Anomaly](https://github.com/astha-chem/mvts-ano-eval)
- [QuoVadis TAD](https://github.com/ssarfraz/QuoVadisTAD)

