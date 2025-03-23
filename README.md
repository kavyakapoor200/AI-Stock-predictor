# AI-Stock-predictor

### **📌 AI Stock Price Predictor using LSTM**  
A deep learning model that predicts future stock prices based on historical trends using **LSTM (Long Short-Term Memory) networks**.  

![Stock Prediction](aapl_prediction.png)  

---

## **📖 Table of Contents**  
- [📌 About the Project](#-about-the-project)  
- [🛠 Tech Stack](#-tech-stack)  
- [📂 Dataset](#-dataset)  
- [🚀 Installation & Setup](#-installation--setup)  
- [⚡ Model Architecture](#-model-architecture)  
- [📈 Results](#-results)  
- [💡 Future Improvements](#-future-improvements)  
- [📜 License](#-license)  

---

## **📌 About the Project**  
This project is an **AI-powered stock price predictor** that uses **LSTM (Long Short-Term Memory)** to forecast future stock prices based on historical market data.  

✔ **Stock Data Source:** Yahoo Finance  
✔ **Model Type:** LSTM (Recurrent Neural Network)  
✔ **Prediction Window:** 30 Days  

---

## **🛠 Tech Stack**  
- **Python** 🐍  
- **TensorFlow/Keras** 🤖  
- **Yahoo Finance (`yfinance`)** 📈  
- **Matplotlib & Seaborn** 📊  
- **NumPy & Pandas** 🏗  

---

## **📂 Dataset**  
The dataset is fetched in **real-time** from **Yahoo Finance** using `yfinance`:  
```python
import yfinance as yf

# Download stock data (Example: Apple - AAPL)
stock_symbol = "AAPL"
data = yf.download(stock_symbol, start="2010-01-01", end="2024-01-01")
```
💡 **Supports any stock symbol (TSLA, GOOGL, MSFT, etc.)**  

---

## **🚀 Installation & Setup**  
### **1️⃣ Install Dependencies**  
```bash
pip install tensorflow pandas numpy matplotlib yfinance scikit-learn
```
### **2️⃣ Run the Code**  
```python
python AI_Stock_Predictor.ipynb
```
*(Ensure all dependencies are installed before running!)*  

---

## **⚡ Model Architecture**  
Our **LSTM Model** consists of:  
✔ **3 LSTM layers** 🏗  
✔ **Dropout layers** (to prevent overfitting)  
✔ **Dense layers** for output prediction  

```python
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    
    LSTM(100, return_sequences=True),
    Dropout(0.2),

    LSTM(100, return_sequences=False),
    Dropout(0.2),
    
    Dense(50),
    Dense(1)
])
```
📌 **The model learns stock price trends and predicts future values.**  

---

## **📈 Results**  
- **Predictions for the next 30 days** 📊  
- **Loss:** `~0.0006` *(Low loss = good model performance!)*  

✅ **Final Graph:** *(Replace with yours!)*  
![Predicted vs Actual](aapl_price.png)  

---

## **💡 Future Improvements**  
🚀 **Possible Enhancements:**  
✔ Deploy as a **Streamlit Web App** 🌐  
✔ Add **multiple stock predictions** (TSLA, GOOGL, etc.)  
✔ Fine-tune hyperparameters for better accuracy 📊  

---

## **📜 License**  
📜 This project is **open-source** and free to use!  
