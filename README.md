# 📈 STOCK Prediction Project - Dual Forecast

Welcome to the **Stock Prediction Project**! This repository contains a Python-based model that predicts **both the direction and price** of selected stocks using historical data. Perfect for learning, experimentation, or showcasing in your resume! 🚀

---

## 🔍 Project Overview

This project predicts:

1. **Direction** – Whether the stock price will go **UP** 📈 or **DOWN** 📉.  
2. **Price** – The forecasted price of the stock for **tomorrow** and **weekly** (5 days).

✅ Supports multiple popular stocks: **AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA**  
💡 Uses **6 simple features** for predictions to prevent overfitting.

---

## 🛠 Features

- Dual prediction: Direction + Price  
- Daily update of stock data using **Yahoo Finance API**  
- Confidence levels for predictions: Very High / High / Medium  
- Signals for trading decisions:  
  - 🟢 BUY  
  - 🔴 SELL / SHORT  
  - ⚡ Short-term trade  

---

## ⚡ Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Harishlal-me/STOCK-final.git
cd STOCK-final
2️⃣ Create and activate virtual environment
bash
Copy code
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Update stock data
bash
Copy code
python update_data.py AAPL MSFT GOOGL AMZN NVDA TSLA
5️⃣ Run predictions
bash
Copy code
python predict.py -s AAPL MSFT GOOGL AMZN NVDA TSLA
💡 Optional: Combine update + predict in one go:

bash
Copy code
python update_and_predict.py
📊 Example Output
yaml
Copy code
📊 TSLA - DUAL PREDICTION
💰 CURRENT PRICE: $481.20
📈 TOMORROW FORECAST: DOWN (probability: 25.9%)
Predicted price: $480.81 (-0.08%)
Signal: ✅ YES
📍 TRADING DECISION: 🔴 SELL/SHORT
🗂 Project Structure
php
Copy code
STOCK-final/
│
├─ data/               # Cached stock data CSVs
├─ .venv/              # Virtual environment
├─ update_data.py      # Script to fetch latest stock prices
├─ predict.py          # Script to run predictions
├─ update_and_predict.py # Combined update + prediction script
├─ requirements.txt    # Dependencies
└─ README.md           # This file
💡 Notes & Recommendations
Update your data daily to keep predictions accurate.

Model thresholds:

Tomorrow: 55%

Weekly: 58%

Use this project as a resume showcase for machine learning & stock prediction.

📌 Technologies Used
Python 🐍

TensorFlow / Keras

Yahoo Finance API (yfinance)

Pandas, NumPy

🤝 Contribution
Feel free to ⭐ star the project and contribute!
Any improvements in UI (Streamlit/Web) or model accuracy are welcome.

📜 License
MIT License © 2025 Harishlal Me
Do you want me to do that next?
