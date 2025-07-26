# 📈 Stock Price Monte Carlo Simulator

A powerful web application that uses Monte Carlo simulation with Geometric Brownian Motion (GBM) to forecast stock price movements and analyze investment risk.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 **Live Demo**

[Try the live application here](https://stock-monte-carlo-simulator.streamlit.app/) 

## 📊 **Features**

- **Real-time Stock Data**: Fetches live historical data using Yahoo Finance API
- **Monte Carlo Simulation**: Runs 100-5,000 simulations using Geometric Brownian Motion
- **Risk Analysis**: Calculates Value-at-Risk (VaR), Expected Shortfall, and probability metrics
- **Interactive Visualizations**: Dynamic charts showing price paths and distributions
- **Confidence Intervals**: 80%, 90%, 95%, and 99% confidence ranges
- **Professional Interface**: Clean, responsive web interface built with Streamlit

## 🛠️ **Technical Stack**

- **Python 3.8+**
- **Streamlit** - Web application framework
- **yfinance** - Yahoo Finance API integration
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib** - Statistical visualizations
- **Plotly** - Interactive charts (optional)

## 📈 **How It Works**

The application uses **Geometric Brownian Motion (GBM)** to model stock price movements:

```
S(t+dt) = S(t) * exp((μ - σ²/2) * dt + σ * √(dt) * Z)
```

Where:
- `S(t)` = Stock price at time t
- `μ` = Expected return (drift)
- `σ` = Volatility
- `Z` = Random normal variable
- `dt` = Time step

## 🔧 **Installation & Setup**

### **Local Installation**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-monte-carlo-simulator.git
cd stock-monte-carlo-simulator
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
Navigate to `http://localhost:8501`

## 📱 **Usage**

1. **Enter Stock Symbol** (e.g., AAPL, GOOGL, TSLA)
2. **Configure Parameters**:
   - Forecast period (30-365 days)
   - Number of simulations (100-5,000)
   - Historical data period (1-5 years)
   - Confidence level (90-99%)
3. **Click "Run Simulation"**
4. **Analyze Results**:
   - View simulated price paths
   - Check probability distributions
   - Review risk metrics

## 📊 **Sample Output**

### **Risk Metrics Calculated:**
- **Value at Risk (VaR)**: Maximum expected loss at given confidence level
- **Expected Shortfall**: Average loss beyond VaR threshold  
- **Probability of Profit**: Likelihood of positive returns
- **Volatility Assessment**: Risk level categorization

### **Visualizations:**
- Monte Carlo price path simulations
- Final price distribution histograms
- Confidence interval ranges
- Risk assessment dashboard

## 🎯 **Use Cases**

- **Portfolio Risk Management**: Assess potential losses
- **Investment Planning**: Evaluate expected returns
- **Options Pricing**: Understand price volatility
- **Financial Education**: Learn Monte Carlo methods
- **Research & Analysis**: Academic and professional research

## ⚠️ **Disclaimer**

This tool is for **educational and research purposes only**. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## 📝 **Model Assumptions**

- Stock returns follow log-normal distribution
- Constant volatility and drift over time horizon
- No dividends or corporate actions
- Efficient market hypothesis applies

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 **Performance**

- **Fast Execution**: Optimized NumPy operations
- **Scalable**: Handles 1,000-5,000 simulations efficiently  
- **Memory Efficient**: Minimal resource usage
- **Error Handling**: Robust exception management


### By joshua Agbroko 
