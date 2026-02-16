# 📈 Reliance Stock Price Prediction using LSTM (PyTorch)

## 🧠 Overview

This repository contains my **MCA Major Project** focused on building a deep learning–based time-series forecasting system to predict **Reliance Industries stock closing prices** using an **LSTM (Long Short-Term Memory) neural network** implemented in **PyTorch**.

The project demonstrates an end-to-end machine learning workflow including:

* Data acquisition from Yahoo Finance
* Data cleaning & preprocessing
* Feature scaling & sequence generation
* LSTM model design and training
* Model evaluation using RMSE and R²
* Visualization of predicted vs actual prices

The goal is to explore how recurrent neural networks can capture temporal dependencies in financial data and produce reliable predictions on unseen time periods.

---

## 🎓 Academic Context

This project was developed as part of my **Master of Computer Applications (MCA) Major Project**.

It showcases practical skills in:

* Deep Learning for Time Series
* PyTorch model development
* Data preprocessing pipelines
* Financial data analysis
* Model evaluation and visualization
* Reproducible ML workflows

---

## ⚙️ Tech Stack

* **Python**
* **PyTorch**
* **Pandas / NumPy**
* **scikit-learn**
* **Matplotlib**
* **yfinance API**

---

## 📊 Dataset

* Source: Yahoo Finance (`yfinance`)
* Stock: **RELIANCE.NS**
* Duration: **9 years of daily historical data**
* Train/Test split:

  * **7 years → training**
  * **2 years → testing**

Features used:

```
Open, High, Low, Close, Volume
```

Target:

```
Next-day Closing Price
```

---

## 🏗️ Project Pipeline

1. **Download historical stock data**
2. **Clean and preprocess dataset**
3. **Scale features using MinMaxScaler**
4. **Create sliding window sequences (20 days history)**
5. **Train multi-layer LSTM model**
6. **Predict on unseen test period**
7. **Inverse-scale predictions**
8. **Evaluate using RMSE and R²**
9. **Visualize results**

---

## 🧩 Model Architecture

* Multi-layer LSTM network
* Hidden size: 64
* Sequence length: 20 days
* Optimizer: Adam
* Loss function: Mean Squared Error

The LSTM captures temporal dependencies in stock price movements and outputs the predicted next-day closing price.

---

## 📉 Evaluation Metrics

The model is evaluated using:

* **RMSE** — prediction error in price units
* **R² Score** — variance explained by the model

These metrics help measure both numerical accuracy and generalization performance.

---

## 🚀 How to Run

### 1️⃣ Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/repo-name.git
cd repo-name
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

(or install manually: pandas, numpy, torch, sklearn, matplotlib, yfinance)

### 3️⃣ Run notebook

Open:

```
LSTM.ipynb
```

Run all cells sequentially.

The notebook will:

* download data
* train the model
* generate predictions
* show evaluation results

---

## 📁 Repository Structure

```
├── LSTM.ipynb                # Main project notebook
├── reliance.csv              # Downloaded dataset (generated automatically)
├── requirements.txt
├── README.md
```

---

## 🤝 Acknowledgement

This project was developed independently as part of my academic curriculum.
I used **ChatGPT as a learning assistant** for:

* conceptual clarification of LSTM architecture
* debugging guidance
* improving code structure and documentation

All implementation decisions, testing, and final integration were performed by me.

---

## 📌 Future Improvements

* Add technical indicators (SMA, EMA, RSI)
* Hyperparameter tuning
* Multi-output prediction (Open + Close)
* Deploy as a web app (Streamlit)

---

## 👤 Author

**Bhavik Buchke**
MCA Student | Aspiring Data Scientist
Interested in Machine Learning, Deep Learning, and Data Analytics
