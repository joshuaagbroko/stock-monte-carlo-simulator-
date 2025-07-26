import streamlit as st
import sys
import subprocess

# Function to install packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required packages
required_packages = {
    'yfinance': 'yfinance',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'plotly': 'plotly'
}

for package_name, pip_name in required_packages.items():
    try:
        __import__(package_name)
    except ImportError:
        with st.spinner(f"Installing {package_name}..."):
            try:
                install_package(pip_name)
                st.success(f"Successfully installed {package_name}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to install {package_name}: {str(e)}")
                st.stop()

# Now import all required libraries
try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime, timedelta
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.error("Please install required packages: pip install yfinance numpy pandas matplotlib plotly")
    st.stop()

# Set page configuration
try:
    st.set_page_config(
        page_title="Stock Price Monte Carlo Simulator",
        page_icon="📈",
        layout="wide"
    )
except Exception as e:
    st.error(f"Configuration error: {str(e)}")

# Title and description
st.title("📈 Stock Price Monte Carlo Simulator")
st.markdown("""
This tool uses **Monte Carlo simulation** with **Geometric Brownian Motion (GBM)** to forecast potential stock price paths.
It analyzes historical data to estimate future price distributions and risk metrics.
""")

# Test basic functionality
st.success("✅ Application loaded successfully!")

# Sidebar for inputs
st.sidebar.header("📊 Simulation Parameters")

# Stock symbol input
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock ticker (e.g., AAPL, GOOGL, TSLA)")

# Time horizon
time_horizon = st.sidebar.selectbox("Forecast Period (Days)", [30, 60, 90, 180, 365], index=0)

# Number of simulations
num_simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 500, step=100)

# Historical data period
data_period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "3y", "5y"], index=1)

# Confidence level for VaR
confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95)

# Run simulation button
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")

class MonteCarloStockSimulator:
    def __init__(self, symbol, period="2y"):
        self.symbol = symbol.upper()
        self.period = period
        self.stock_data = None
        self.returns = None
        self.mu = None  # drift
        self.sigma = None  # volatility
        
    def fetch_data(self):
        """Fetch historical stock data"""
        try:
            st.info(f"Fetching data for {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            self.stock_data = ticker.history(period=self.period)
            
            if self.stock_data.empty:
                st.error(f"No data found for symbol {self.symbol}. Please check the ticker symbol.")
                return False
            
            if len(self.stock_data) < 50:
                st.warning(f"Limited data available for {self.symbol}. Results may be less reliable.")
            
            st.success(f"Successfully fetched {len(self.stock_data)} days of data for {self.symbol}")
            return True
            
        except Exception as e:
            st.error(f"Error fetching data for {self.symbol}: {str(e)}")
            return False
    
    def calculate_parameters(self):
        """Calculate drift and volatility from historical data"""
        try:
            # Calculate daily returns
            self.returns = np.log(self.stock_data['Close'] / self.stock_data['Close'].shift(1)).dropna()
            
            if len(self.returns) == 0:
                raise ValueError("No returns data available")
            
            # Calculate annualized parameters
            self.mu = self.returns.mean() * 252  # Annualized drift
            self.sigma = self.returns.std() * np.sqrt(252)  # Annualized volatility
            
            current_price = float(self.stock_data['Close'].iloc[-1])
            
            return {
                'daily_return_mean': float(self.returns.mean()),
                'daily_return_std': float(self.returns.std()),
                'annualized_return': float(self.mu),
                'annualized_volatility': float(self.sigma),
                'current_price': current_price,
                'data_points': len(self.stock_data)
            }
        except Exception as e:
            st.error(f"Error calculating parameters: {str(e)}")
            return None
    
    def monte_carlo_simulation(self, days, num_sims):
        """Run Monte Carlo simulation using GBM"""
        try:
            current_price = float(self.stock_data['Close'].iloc[-1])
            dt = 1/252  # Daily time step
            
            # Initialize price paths array
            price_paths = np.zeros((num_sims, days + 1))
            price_paths[:, 0] = current_price
            
            # Generate random shocks
            np.random.seed(42)  # For reproducibility
            
            for t in range(1, days + 1):
                Z = np.random.standard_normal(num_sims)
                price_paths[:, t] = price_paths[:, t-1] * np.exp(
                    (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
                )
            
            return price_paths
        except Exception as e:
            st.error(f"Error in Monte Carlo simulation: {str(e)}")
            return None
    
    def calculate_risk_metrics(self, final_prices, current_price, confidence_level):
        """Calculate risk metrics"""
        try:
            returns = (final_prices - current_price) / current_price * 100
            
            var_level = (100 - confidence_level) / 100
            var = float(np.percentile(returns, var_level * 100))
            
            # Expected shortfall (Conditional VaR)
            es = float(returns[returns <= var].mean()) if len(returns[returns <= var]) > 0 else var
            
            return {
                'var': var,
                'expected_shortfall': es,
                'expected_return': float(returns.mean()),
                'volatility': float(returns.std()),
                'prob_profit': float((returns > 0).mean() * 100),
                'prob_loss_10': float((returns < -10).mean() * 100),
                'prob_loss_20': float((returns < -20).mean() * 100)
            }
        except Exception as e:
            st.error(f"Error calculating risk metrics: {str(e)}")
            return None

# Main application logic
if run_simulation:
    if not symbol.strip():
        st.error("Please enter a stock symbol")
    else:
        try:
            with st.spinner(f"Initializing simulation for {symbol}..."):
                simulator = MonteCarloStockSimulator(symbol, data_period)
                
                if simulator.fetch_data():
                    # Calculate parameters
                    params = simulator.calculate_parameters()
                    
                    if params is None:
                        st.error("Failed to calculate parameters")
                    else:
                        current_price = params['current_price']
                        
                        # Display current stock info
                        st.subheader(f"📊 {symbol.upper()} Stock Analysis")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("Annual Return", f"{params['annualized_return']:.2%}")
                        with col3:
                            st.metric("Annual Volatility", f"{params['annualized_volatility']:.2%}")
                        with col4:
                            st.metric("Data Points", params['data_points'])
                        
                        # Run Monte Carlo simulation
                        with st.spinner(f"Running {num_simulations:,} simulations for {time_horizon} days..."):
                            price_paths = simulator.monte_carlo_simulation(time_horizon, num_simulations)
                            
                            if price_paths is not None:
                                final_prices = price_paths[:, -1]
                                
                                # Calculate risk metrics
                                risk_metrics = simulator.calculate_risk_metrics(
                                    final_prices, current_price, confidence_level
                                )
                                
                                if risk_metrics is not None:
                                    # Create two main columns for results
                                    col_left, col_right = st.columns([2, 1])
                                    
                                    with col_left:
                                        st.subheader("📊 Simulated Price Paths")
                                        
                                        # Create plot with matplotlib (more reliable than plotly)
                                        fig, ax = plt.subplots(figsize=(12, 6))
                                        
                                        # Plot sample paths
                                        sample_size = min(50, num_simulations)
                                        sample_indices = np.random.choice(num_simulations, sample_size, replace=False)
                                        
                                        for i in sample_indices:
                                            ax.plot(range(time_horizon + 1), price_paths[i], 
                                                   alpha=0.3, linewidth=0.5, color='lightblue')
                                        
                                        # Plot mean path
                                        mean_path = np.mean(price_paths, axis=0)
                                        ax.plot(range(time_horizon + 1), mean_path, 
                                               color='red', linewidth=2, label='Mean Path')
                                        
                                        # Add current price line
                                        ax.axhline(y=current_price, color='green', linestyle='--', 
                                                  label='Current Price')
                                        
                                        ax.set_xlabel('Days')
                                        ax.set_ylabel('Stock Price ($)')
                                        ax.set_title(f'{symbol.upper()} - {num_simulations:,} Monte Carlo Simulations ({time_horizon} days)')
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        
                                        st.pyplot(fig)
                                        plt.close()
                                        
                                        # Price distribution histogram
                                        st.subheader("📈 Final Price Distribution")
                                        
                                        fig2, ax2 = plt.subplots(figsize=(10, 5))
                                        ax2.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                                        
                                        # Add percentile lines
                                        percentiles = [5, 25, 50, 75, 95]
                                        colors = ['red', 'orange', 'green', 'orange', 'red']
                                        
                                        for p, color in zip(percentiles, colors):
                                            value = np.percentile(final_prices, p)
                                            ax2.axvline(x=value, color=color, linestyle='--', 
                                                       label=f'{p}th percentile: ${value:.2f}')
                                        
                                        ax2.set_xlabel('Final Stock Price ($)')
                                        ax2.set_ylabel('Frequency')
                                        ax2.set_title(f'Distribution of {symbol.upper()} Prices after {time_horizon} days')
                                        ax2.legend()
                                        ax2.grid(True, alpha=0.3)
                                        
                                        st.pyplot(fig2)
                                        plt.close()
                                    
                                    with col_right:
                                        st.subheader("📋 Risk Analysis")
                                        
                                        # Key statistics
                                        st.markdown("### 📊 Key Statistics")
                                        stats_data = {
                                            'Mean Price': f"${np.mean(final_prices):.2f}",
                                            'Median Price': f"${np.median(final_prices):.2f}",
                                            'Min Price': f"${np.min(final_prices):.2f}",
                                            'Max Price': f"${np.max(final_prices):.2f}",
                                            'Std Dev': f"${np.std(final_prices):.2f}"
                                        }
                                        
                                        for key, value in stats_data.items():
                                            st.metric(key, value)
                                        
                                        # Risk metrics
                                        st.markdown("### ⚠️ Risk Metrics")
                                        st.metric("Expected Return", f"{risk_metrics['expected_return']:.2f}%")
                                        st.metric("Volatility", f"{risk_metrics['volatility']:.2f}%")
                                        st.metric(f"Value at Risk ({confidence_level}%)", f"{risk_metrics['var']:.2f}%")
                                        st.metric(f"Expected Shortfall", f"{risk_metrics['expected_shortfall']:.2f}%")
                                        
                                        # Probability metrics
                                        st.markdown("### 🎯 Probability Analysis")
                                        st.metric("Probability of Profit", f"{risk_metrics['prob_profit']:.1f}%")
                                        st.metric("Prob. of >10% Loss", f"{risk_metrics['prob_loss_10']:.1f}%")
                                        st.metric("Prob. of >20% Loss", f"{risk_metrics['prob_loss_20']:.1f}%")
                                    
                                    # Additional insights
                                    st.subheader("💡 Key Insights")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        expected_price = np.mean(final_prices)
                                        price_change = (expected_price - current_price) / current_price * 100
                                        st.info(f"""
                                        **Expected Outcome**
                                        - Expected price: ${expected_price:.2f}
                                        - Expected change: {price_change:+.2f}%
                                        """)
                                    
                                    with col2:
                                        st.warning(f"""
                                        **Risk Assessment**
                                        - {confidence_level}% VaR: {risk_metrics['var']:.2f}%
                                        - Chance of loss: {100-risk_metrics['prob_profit']:.1f}%
                                        """)
                                    
                                    with col3:
                                        vol = params['annualized_volatility']
                                        vol_level = "High" if vol > 0.3 else "Medium" if vol > 0.2 else "Low"
                                        st.success(f"""
                                        **Volatility Level**
                                        - Annual volatility: {vol:.1%}
                                        - Assessment: {vol_level}
                                        """)
                                else:
                                    st.error("Failed to calculate risk metrics")
                            else:
                                st.error("Failed to run Monte Carlo simulation")
                else:
                    st.error("Failed to fetch stock data")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.error("Please try again or contact support if the issue persists.")

else:
    # Initial state - show instructions
    st.info("👆 Configure your simulation parameters in the sidebar and click 'Run Simulation' to get started!")
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    1. **Enter a Stock Symbol** (e.g., AAPL, GOOGL, TSLA, MSFT)
    2. **Choose Forecast Period** (30-365 days)
    3. **Set Number of Simulations** (more = more accurate but slower)
    4. **Select Historical Data Period** (1-5 years for parameter estimation)
    5. **Click 'Run Simulation'** to generate forecasts
    """)
    
    # Sample stocks to try
    st.subheader("💡 Popular Stocks to Try")
    sample_stocks = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "TSLA": "Tesla Inc.",
        "MSFT": "Microsoft Corp.",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corp.",
        "META": "Meta Platforms Inc.",
        "SPY": "SPDR S&P 500 ETF"
    }
    
    cols = st.columns(4)
    for i, (ticker, name) in enumerate(sample_stocks.items()):
        with cols[i % 4]:
            st.code(f"{ticker}\n{name}")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This tool is for educational and research purposes only. 
Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

**Technical Details**: Uses Geometric Brownian Motion with historical volatility and drift estimation.
""")