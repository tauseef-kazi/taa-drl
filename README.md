# **Tactical Asset Allocation Using Deep Reinforcement Learning And Latent Macroeconomic Conditions**

MSc Financial Engineering Thesis – **Tauseef Kazi**

---

> A reproducible research repo that trains deep‐reinforcement‑learning agents to allocate across **36 global ETFs** while conditioning on **13‑region macroeconomic signals**.
> Key contributions:
>
> 1. **Multi‑input Gym environment** combining price, regime (HMM), Technical indicators & macro state spaces.
> 2. **Attention‑based feature extractors** (No‑Macro MHA, per‑region LSTM, Bi‑LSTM + Transformer) built on *Stable‑Baselines3*.
> 3. Training the DRL agents from **2003-07-10 to 2016-11-01** and backtested from *2016-11-02 to 2025‑04‑11** with Kelly‐ensemble post‑processing.
> 4. Benchmarks: SPY, DIA, BuyAndHold & EquallyWeighted.
> 5. Rich analytics: rolling Sharpe, Sortino, drawdown heat‑maps & Greek attribution.

---

## 1  Folder Structure

```text
.
├── data/                                      # Raw & engineered datasets (auto‑created)
│   ├── price_data.zip                         # 36 ETF OHLCV + Calculated adjusted close price (Parquet)
│   ├── macro_data.zip                         # 13‑region monthly OECD macro with imputed values from other sources (Parquet)
│   └── hmm_rolling.zip                        # 36 ETF OHLCV + Adjusted Close + 2 state HMM regimes (Parquet)
│   └── alpha_etfs.zip                         # Raw ETF & Benchmark data downloaded from Alphavantage.co APIs (CSV)
│   └── oecd_macro_raw_data.zip                # Raw macroeconomic data downloaded from OECD APIs (CSV)
│   └── macroeconomic_data.zip                 # Raw macroeconomic data downloaded for imputations from IMF, Worldbank, Central Banks and National Bureau of Statistics of respective countries (CSV and XLSX)
├── notebooks/
│   └── M7 - Final Project - Capstone.ipynb    # **Main end‑to‑end notebook**
│   └── M7 - Macroeconomic Variables.ipynb     # **Notebook for EDA and imputation process of macro data**
├── trained_models/                            # Best SB3 checkpoints (auto‑saved)
├── tensorboard_log/                           # Training curves (auto‑saved)
├── requirements.txt                           # Exact python deps (see below)
└── README.md                                  
```

---

## 2  Prerequisites

| Tool                                         | Tested Version      |
| -------------------------------------------- | ------------------- |
| Python                                       | 3.11 (min 3.9)      |
| CUDA                                         | 12.1 (GPU optional) |
| JupyterLab / Notebook                        | 4.2                 |
|  (see `requirements.txt` for full PyPI pins) |                     |

Create a clean venv and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt  # CPU wheel
# GPU users – swap torch line, e.g.:
pip install torch==2.7.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Environment variables (optional secrets) live in `.env` and are loaded via **python‑dotenv**.

---

## 3  Data Setup

1. **Download** all the files under data folder.
2. Drop both files in repo root.
3. First run extracts them into `price_data/`, `hmm_rolling/` & `macro_data/` automatically.
4. For execution efficiency 3-year rolling HMMs are calculated and stored in hmm_rolling.zip, if you wish to recalculate the HMMs then **do not drop** hmm_rolling.zip in the repo root.

> Each macro file is `REGION.parquet`; each ETF price file is `etf_TICKER.parquet`; each HMM rolling file is `hmm_rolling_TICKER.parquet`.

---

## 4  Quick Start (Notebook)

```bash
jupyter notebook notebooks/M7 - Final Project - Capstone.ipynb
```

Run cells top‑to‑bottom:

1. **Imports** - imports required libraries.
2. **Extract and Populate saved price and macroeconomic dataset** – populates `adjusted_etf_data` & `macroeconomic_df` dicts.
3. **HMM regime labelling** – `compute_rolling_hmm_for_etfs()` (caches to `hmm_rolling/`).
4. **Feature prep** – `preprocess_etf_data()` & `preprocess_macro_data()`.
5. **Training loops** – `run_drl_backtest()` for every extractor/reward combination. If you wish not to train the agents from scratch and just use the already trained agents in deterministic mode for testing purpose, then drop the `trained_models/` directory in the repo root and set following flag when calling the training loop `run_drl_backtest(load_saved=True)` function.
6. **Kelly ensemble & analytics** – `BacktestTracker` plots + `metrics()` table.

TensorBoard launches automatically (cell magic):

```bash
%load_ext tensorboard
%tensorboard --logdir tensorboard_log
```

---

## 5  Citing This Work

If you find this project helpful, please cite:

```bibtex
@mastersthesis{kazi2025taa,
  title  = {Tactical Asset Allocation Using Deep Reinforcement Learning And Latent Macroeconomic Conditions},
  author = {Tauseef Kazi},
  school = {MSc Financial Engineering},
  year   = {2025}
}
```

---

## 6  License

This repository is released under the **MIT License**. See `LICENSE` for details.
