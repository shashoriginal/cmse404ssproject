"""
Stock Analysis Command Line Interface
Author: rajshash

This module provides a command-line interface for analyzing stocks using Yahoo Finance data
and making predictions using the trained LSTM model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.utils.data
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from src.models.lstm import LSTMModel
from src.models.training import load_and_prepare_data, evaluate_and_visualize
from src.utils.helpers import plot_predictions, plot_correlation_matrix

def determine_optimal_parameters(stock_data):
    """Determine optimal parameters based on stock data characteristics.
    
    Args:
        stock_data (pd.DataFrame): Historical stock data
        
    Returns:
        dict: Dictionary containing optimal parameters
    """
    # Calculate volatility
    returns = stock_data['Close'].pct_change()
    volatility = returns.std()
    
    # Determine sequence length based on volatility
    if volatility > 0.03:  # High volatility
        sequence_length = 90  # Longer sequence for volatile stocks
    elif volatility > 0.02:  # Medium volatility
        sequence_length = 60
    else:  # Low volatility
        sequence_length = 30
    
    # Determine prediction horizon based on data length
    total_days = len(stock_data)
    if total_days > 1000:
        prediction_days = 30
    elif total_days > 500:
        prediction_days = 14
    else:
        prediction_days = 7
        
    return {
        'sequence_length': sequence_length,
        'prediction_days': prediction_days,
        'volatility': volatility
    }

def prepare_stock_data(symbol, start_date=None, end_date=None):
    """Download and prepare stock data from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        
    Returns:
        tuple: (processed_data, scaler, stock_info)
    """
    # If dates not provided, use last 2 years of data
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=730)  # 2 years
        
    # Download stock data
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data for {symbol}. Found only {len(df)} days of data.")
    
    # Calculate additional features
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    # Select features for prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
               'Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD']
    data = df[features].values
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler, stock.info

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def calculate_trading_signals(df):
    """Calculate trading signals based on technical indicators.
    
    Args:
        df (pd.DataFrame): Dataframe with technical indicators
        
    Returns:
        pd.DataFrame: DataFrame with trading signals
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    
    # RSI signals
    signals['rsi_signal'] = 0  # 0: neutral, 1: buy, -1: sell
    signals.loc[df['RSI'] < 30, 'rsi_signal'] = 1  # Buy signal: RSI < 30 (oversold)
    signals.loc[df['RSI'] > 70, 'rsi_signal'] = -1  # Sell signal: RSI > 70 (overbought)
    
    # Moving Average signals
    signals['ma_signal'] = 0
    signals.loc[df['Close'] > df['MA50'], 'ma_signal'] = 1  # Buy: price above 50-day MA
    signals.loc[df['Close'] < df['MA50'], 'ma_signal'] = -1  # Sell: price below 50-day MA
    
    # MACD signals
    signals['macd_signal'] = 0
    signals.loc[df['MACD'] > 0, 'macd_signal'] = 1  # Buy: MACD above 0
    signals.loc[df['MACD'] < 0, 'macd_signal'] = -1  # Sell: MACD below 0
    
    # Calculate the overall signal (simple average of all signals)
    signals['overall_signal'] = (signals['rsi_signal'] + signals['ma_signal'] + signals['macd_signal']) / 3
    
    # Trading recommendation
    signals['recommendation'] = 'Hold'
    signals.loc[signals['overall_signal'] > 0.5, 'recommendation'] = 'Buy'
    signals.loc[signals['overall_signal'] < -0.5, 'recommendation'] = 'Sell'
    
    return signals

def assess_risk(volatility, price_history, future_predictions):
    """Assess the risk level of a stock based on volatility and predictions.
    
    Args:
        volatility (float): Stock price volatility
        price_history (np.array): Recent price history
        future_predictions (np.array): Predicted future prices
        
    Returns:
        dict: Risk assessment metrics
    """
    # Categorize volatility risk
    if volatility > 0.04:
        volatility_risk = "High"
    elif volatility > 0.02:
        volatility_risk = "Medium"
    else:
        volatility_risk = "Low"
    
    # Calculate predicted price change
    predicted_change = (future_predictions[-1] - future_predictions[0]) / future_predictions[0]
    
    # Calculate historical price change (over the same time period)
    if len(price_history) >= len(future_predictions):
        historical_change = (price_history[-1] - price_history[-len(future_predictions)]) / price_history[-len(future_predictions)]
    else:
        historical_change = (price_history[-1] - price_history[0]) / price_history[0]
    
    # Calculate potential downside (based on 95% confidence interval lower bound)
    downside_risk = (future_predictions[0] - future_predictions.min()) / future_predictions[0]
    
    # Calculate Sharpe ratio-like metric (return / risk)
    if volatility > 0:
        risk_adjusted_return = predicted_change / volatility
    else:
        risk_adjusted_return = 0
    
    # Determine overall risk level
    if volatility_risk == "High" or downside_risk > 0.1:
        overall_risk = "High"
    elif volatility_risk == "Medium" or downside_risk > 0.05:
        overall_risk = "Medium"
    else:
        overall_risk = "Low"
    
    return {
        'volatility_risk': volatility_risk,
        'predicted_change': predicted_change,
        'historical_change': historical_change,
        'downside_risk': downside_risk,
        'risk_adjusted_return': risk_adjusted_return,
        'overall_risk': overall_risk
    }

def perform_sector_analysis(symbol, period="1y"):
    """Compare a stock against its sector.
    
    Args:
        symbol (str): Stock symbol
        period (str): Time period for comparison (default: 1 year)
        
    Returns:
        dict: Sector comparison metrics
    """
    try:
        # Get stock info
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get stock price history
        stock_history = stock.history(period=period)
        
        # Extract sector
        sector = info.get('sector', 'Unknown')
        
        # Get sector ETF based on the stock's sector
        sector_etfs = {
            'Technology': 'XLK',
            'Consumer Cyclical': 'XLY',
            'Financial Services': 'XLF',
            'Healthcare': 'XLV',
            'Communication Services': 'XLC',
            'Consumer Defensive': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Basic Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU'
        }
        
        sector_etf = sector_etfs.get(sector, 'SPY')  # Default to S&P 500 if sector not found
        
        # Get sector ETF data
        sector_ticker = yf.Ticker(sector_etf)
        sector_history = sector_ticker.history(period=period)
        
        # Calculate returns
        stock_returns = stock_history['Close'].pct_change().dropna()
        sector_returns = sector_history['Close'].pct_change().dropna()
        
        # Calculate metrics
        stock_total_return = (stock_history['Close'].iloc[-1] / stock_history['Close'].iloc[0]) - 1
        sector_total_return = (sector_history['Close'].iloc[-1] / sector_history['Close'].iloc[0]) - 1
        
        stock_volatility = stock_returns.std()
        sector_volatility = sector_returns.std()
        
        # Calculate beta (correlation with sector * stock volatility / sector volatility)
        # Use only the overlapping date range
        common_dates = stock_returns.index.intersection(sector_returns.index)
        if len(common_dates) > 0:
            stock_common = stock_returns.loc[common_dates]
            sector_common = sector_returns.loc[common_dates]
            
            beta = np.cov(stock_common, sector_common)[0, 1] / np.var(sector_common)
        else:
            beta = np.nan
        
        # Calculate alpha (stock return - risk free rate - beta * (sector return - risk free rate))
        # Simplifying by using 0 as risk-free rate
        alpha = stock_total_return - beta * sector_total_return
        
        # Calculate Sharpe ratio (return / volatility)
        stock_sharpe = stock_total_return / stock_volatility if stock_volatility > 0 else 0
        sector_sharpe = sector_total_return / sector_volatility if sector_volatility > 0 else 0
        
        return {
            'sector': sector,
            'sector_etf': sector_etf,
            'stock_return': stock_total_return,
            'sector_return': sector_total_return,
            'relative_performance': stock_total_return - sector_total_return,
            'stock_volatility': stock_volatility,
            'sector_volatility': sector_volatility,
            'beta': beta,
            'alpha': alpha,
            'stock_sharpe': stock_sharpe,
            'sector_sharpe': sector_sharpe
        }
    except Exception as e:
        print(f"Error in sector analysis: {str(e)}")
        return {
            'sector': 'Unknown',
            'error': str(e)
        }

def create_dashboard_summary(symbol, price_data, signals, predictions, risk_assessment, sector_analysis):
    """Create a dashboard summary image with key metrics.
    
    Args:
        symbol (str): Stock symbol
        price_data (pd.DataFrame): Stock price data
        signals (pd.DataFrame): Trading signals
        predictions (np.array): Future predictions
        risk_assessment (dict): Risk assessment metrics
        sector_analysis (dict): Sector comparison metrics
        
    Returns:
        str: Path to the saved dashboard image
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define GridSpec
    gs = plt.GridSpec(4, 3, figure=fig)
    
    # Header with stock name and summary
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')  # Turn off axis
    
    # Create header text
    header_text = f"{symbol} Stock Analysis Dashboard\n"
    header_text += f"Current Price: ${price_data['Close'].iloc[-1]:.2f} | "
    header_text += f"Recommendation: {signals['recommendation'].iloc[-1]} | "
    header_text += f"Risk Level: {risk_assessment['overall_risk']}"
    
    ax_header.text(0.5, 0.5, header_text, fontsize=16, 
                   ha='center', va='center', fontweight='bold')
    
    # Recent price chart with indicators
    ax_price = fig.add_subplot(gs[1, :])
    ax_price.plot(price_data['Close'][-60:], label='Close Price')
    ax_price.plot(price_data['MA20'][-60:], label='20-day MA', alpha=0.7)
    ax_price.plot(price_data['MA50'][-60:], label='50-day MA', alpha=0.7)
    
    # Add buy/sell markers
    buy_signals = signals[signals['recommendation'] == 'Buy'].index
    sell_signals = signals[signals['recommendation'] == 'Sell'].index
    
    if not buy_signals.empty:
        ax_price.scatter(buy_signals, price_data.loc[buy_signals, 'Close'], 
                         marker='^', color='g', s=100, label='Buy Signal')
    if not sell_signals.empty:
        ax_price.scatter(sell_signals, price_data.loc[sell_signals, 'Close'], 
                         marker='v', color='r', s=100, label='Sell Signal')
    
    ax_price.set_title('Recent Price History with Signals')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)
    
    # Future predictions chart
    ax_pred = fig.add_subplot(gs[2, :2])
    
    # Get the last 30 days of actual data
    last_days = 30
    x_hist = np.arange(last_days)
    y_hist = price_data['Close'].values[-last_days:]
    
    # Plot historical data
    ax_pred.plot(x_hist, y_hist, label='Historical', color='blue')
    
    # Plot future predictions
    x_future = np.arange(last_days, last_days + len(predictions))
    ax_pred.plot(x_future, predictions, label='Predicted', color='red', linestyle='--')
    
    # Add confidence interval
    if 'lower_bound' in locals() and 'upper_bound' in locals():
        ax_pred.fill_between(x_future, lower_bound, upper_bound, color='red', alpha=0.2)
    
    ax_pred.set_title('Price Forecast')
    ax_pred.legend(loc='upper left')
    ax_pred.grid(True, alpha=0.3)
    ax_pred.set_xlabel('Days')
    ax_pred.set_ylabel('Price ($)')
    
    # Technical indicators
    ax_tech = fig.add_subplot(gs[2, 2])
    ax_tech.axis('off')  # Turn off axis
    
    # Create a table with technical indicators
    table_data = [
        ['Indicator', 'Value', 'Signal'],
        ['RSI', f"{price_data['RSI'].iloc[-1]:.1f}", signals['rsi_signal'].iloc[-1]],
        ['MACD', f"{price_data['MACD'].iloc[-1]:.3f}", signals['macd_signal'].iloc[-1]],
        ['MA Cross', '', signals['ma_signal'].iloc[-1]],
        ['Overall', '', signals['overall_signal'].iloc[-1]]
    ]
    
    # Convert signal values to colored text
    signal_colors = {
        -1: 'red',
        0: 'black',
        1: 'green'
    }
    
    # Create table
    table = ax_tech.table(
        cellText=[[str(cell) for cell in row] for row in table_data],
        cellLoc='center',
        loc='center',
        cellColours=[['lightgray', 'lightgray', 'lightgray']] + 
                    [[None, None, signal_colors.get(row[2], 'black')] for row in table_data[1:]]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax_tech.set_title('Technical Signals')
    
    # Risk assessment
    ax_risk = fig.add_subplot(gs[3, 0])
    ax_risk.axis('off')  # Turn off axis
    
    risk_text = "Risk Assessment\n\n"
    risk_text += f"Volatility: {risk_assessment['volatility_risk']}\n"
    risk_text += f"Predicted Change: {risk_assessment['predicted_change']:.2%}\n"
    risk_text += f"Downside Risk: {risk_assessment['downside_risk']:.2%}\n"
    risk_text += f"Risk-Adjusted Return: {risk_assessment['risk_adjusted_return']:.2f}\n"
    risk_text += f"Overall Risk: {risk_assessment['overall_risk']}"
    
    ax_risk.text(0.5, 0.5, risk_text, fontsize=10,
                ha='center', va='center', linespacing=1.5)
    
    # Sector comparison
    ax_sector = fig.add_subplot(gs[3, 1])
    ax_sector.axis('off')  # Turn off axis
    
    sector_text = "Sector Comparison\n\n"
    sector_text += f"Sector: {sector_analysis['sector']}\n"
    sector_text += f"Stock Return: {sector_analysis['stock_return']:.2%}\n"
    sector_text += f"Sector Return: {sector_analysis['sector_return']:.2%}\n"
    sector_text += f"Alpha: {sector_analysis['alpha']:.2%}\n"
    sector_text += f"Beta: {sector_analysis['beta']:.2f}\n"
    
    ax_sector.text(0.5, 0.5, sector_text, fontsize=10,
                  ha='center', va='center', linespacing=1.5)
    
    # Model metrics
    ax_model = fig.add_subplot(gs[3, 2])
    ax_model.axis('off')  # Turn off axis
    
    model_text = "Model Performance\n\n"
    if 'mse' in locals():
        model_text += f"MSE: {mse:.4f}\n"
        model_text += f"MAE: {mae:.4f}\n"
        model_text += f"RMSE: {rmse:.4f}\n"
    model_text += f"Confidence: {'High' if risk_assessment['downside_risk'] < 0.05 else 'Medium' if risk_assessment['downside_risk'] < 0.1 else 'Low'}"
    
    ax_model.text(0.5, 0.5, model_text, fontsize=10,
                 ha='center', va='center', linespacing=1.5)
    
    plt.tight_layout()
    
    return fig

def analyze_stock(symbol, output_dir=None, start_date=None, end_date=None):
    """Analyze a stock and make predictions."""
    print(f"\nAnalyzing {symbol}...")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the data
    try:
        data, scaler, stock_info = prepare_stock_data(symbol, start_date, end_date)
        print(f"\nData shape after preparation: {data.shape}")
        
        # Create a DataFrame for visualization
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD']
        df_original = pd.DataFrame(scaler.inverse_transform(data), columns=features)
        
        # Calculate trading signals
        signals = calculate_trading_signals(df_original)
        
        # Get current trading recommendation
        current_signal = signals['overall_signal'].iloc[-1]
        current_recommendation = signals['recommendation'].iloc[-1]
        
        print(f"\nCurrent Trading Signals:")
        print(f"RSI Signal: {'Buy' if signals['rsi_signal'].iloc[-1] > 0 else 'Sell' if signals['rsi_signal'].iloc[-1] < 0 else 'Neutral'}")
        print(f"Moving Average Signal: {'Buy' if signals['ma_signal'].iloc[-1] > 0 else 'Sell' if signals['ma_signal'].iloc[-1] < 0 else 'Neutral'}")
        print(f"MACD Signal: {'Buy' if signals['macd_signal'].iloc[-1] > 0 else 'Sell' if signals['macd_signal'].iloc[-1] < 0 else 'Neutral'}")
        print(f"Overall Recommendation: {current_recommendation}")
        
        # Plot correlation matrix
        if output_dir:
            plt.figure(figsize=(12, 10))
            plot_correlation_matrix(df_original, save_path=os.path.join(output_dir, f'{symbol}_correlation.png'))
            plt.close()
            
            # Plot feature distributions
            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(features):
                plt.subplot(4, 3, i+1)
                sns.histplot(df_original[feature], kde=True)
                plt.title(f'{feature} Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_feature_distributions.png'))
            plt.close()
            
            # Plot time series of key features
            plt.figure(figsize=(15, 10))
            plt.subplot(3, 1, 1)
            plt.plot(df_original['Close'], label='Close Price')
            plt.plot(df_original['MA5'], label='5-day MA')
            plt.plot(df_original['MA20'], label='20-day MA')
            plt.plot(df_original['MA50'], label='50-day MA')
            plt.title(f'{symbol} Price and Moving Averages')
            plt.legend()
            
            plt.subplot(3, 1, 2)
            plt.plot(df_original['RSI'], label='RSI')
            plt.axhline(y=70, color='r', linestyle='--')
            plt.axhline(y=30, color='g', linestyle='--')
            plt.title('Relative Strength Index (RSI)')
            plt.legend()
            
            plt.subplot(3, 1, 3)
            plt.plot(df_original['MACD'], label='MACD')
            plt.axhline(y=0, color='k', linestyle='--')
            plt.title('MACD')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_technical_indicators.png'))
            plt.close()
            
            # Plot trading signals
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 1, 1)
            plt.plot(df_original['Close'], label='Close Price')
            
            # Add buy/sell markers
            buy_signals = signals[signals['recommendation'] == 'Buy'].index
            sell_signals = signals[signals['recommendation'] == 'Sell'].index
            
            plt.scatter(buy_signals, df_original.loc[buy_signals, 'Close'], marker='^', color='g', s=100, label='Buy Signal')
            plt.scatter(sell_signals, df_original.loc[sell_signals, 'Close'], marker='v', color='r', s=100, label='Sell Signal')
            
            plt.title(f'{symbol} Trading Signals')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(signals['overall_signal'], label='Signal Strength')
            plt.axhline(y=0.5, color='g', linestyle='--', label='Buy Threshold')
            plt.axhline(y=-0.5, color='r', linestyle='--', label='Sell Threshold')
            plt.title('Signal Strength')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_trading_signals.png'))
            plt.close()
        
    except Exception as e:
        print(f"Error preparing data for {symbol}: {str(e)}")
        return None
    
    # Determine optimal parameters
    params = determine_optimal_parameters(pd.DataFrame(scaler.inverse_transform(data), 
                                                     columns=['Open', 'High', 'Low', 'Close', 'Volume',
                                                             'Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD']))
    print(f"\nOptimal parameters:")
    print(f"Sequence length: {params['sequence_length']}")
    print(f"Prediction days: {params['prediction_days']}")
    print(f"Volatility: {params['volatility']:.4f}")
    
    # Create model with optimal parameters
    model = LSTMModel()
    # Update model configuration with dynamic sequence length
    model.config['model']['sequence_length'] = params['sequence_length']
    sequence_length = params['sequence_length']
    device = model.device  # Get the device from the model
    print(f"\nUsing device: {device}")
    
    # Prepare sequences
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 3])  # Index 3 is 'Close' price
    
    X = np.array(X)
    y = np.array(y)
    print(f"\nSequence shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f"\nTrain/Test split shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Convert to PyTorch tensors and move to the correct device
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    print(f"\nTensor shapes on {device}:")
    print(f"X_train tensor shape: {X_train.shape}")
    print(f"y_train tensor shape: {y_train.shape}")
    
    # Build and train the model
    print(f"\nBuilding model with input shape: {(sequence_length, data.shape[1])}")
    model.build_model((sequence_length, data.shape[1]))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train.reshape(-1, 1)),
        batch_size=32,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test.reshape(-1, 1)),
        batch_size=32,
        shuffle=False
    )
    
    print("\nTraining model...")
    history = model.train_model(train_loader, val_loader=test_loader)
    
    # Plot training history
    if output_dir:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        if 'mae' in history:
            plt.plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{symbol}_training_history.png'))
        plt.close()
    
    # Make predictions
    print("\nGenerating predictions...")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()  # Move predictions back to CPU for numpy operations
    print(f"Prediction shape: {y_pred.shape}")
    
    # Move tensors back to CPU for numpy operations
    y_test = y_test.cpu().numpy()
    
    # Inverse transform predictions
    y_test_inv = scaler.inverse_transform(np.zeros_like(data))[:, 3]
    y_pred_inv = scaler.inverse_transform(np.zeros_like(data))[:, 3]
    y_test_inv[:len(y_test)] = y_test
    y_pred_inv[:len(y_pred)] = y_pred.flatten()
    
    # Calculate error metrics
    mse = np.mean((y_test - y_pred.flatten()) ** 2)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    rmse = np.sqrt(mse)
    
    print(f"\nTest Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    # Generate future predictions
    print("\nGenerating future predictions...")
    try:
        # Get the last sequence from the data
        last_sequence = data[-sequence_length:]
        print(f"\nLast sequence shape before processing: {last_sequence.shape}")
        
        last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0)  # Add batch dimension
        print(f"Last sequence shape after adding batch dimension: {last_sequence.shape}")
        
        last_sequence = last_sequence.to(device)  # Move to the correct device
        print(f"Last sequence device: {last_sequence.device}")
        
        # Generate future predictions
        print(f"\nGenerating {params['prediction_days']} days of predictions...")
        future_predictions = model.predict_sequence(last_sequence, steps=params['prediction_days'])
        print(f"Future predictions shape: {future_predictions.shape}")
        
        # Create a proper template for inverse transformation
        future_template = np.zeros((len(future_predictions), data.shape[1]))
        # Set closing price (column 3) to predictions
        future_template[:, 3] = future_predictions
        
        # Inverse transform the predictions
        future_predictions_inv = scaler.inverse_transform(future_template)[:, 3]
        print(f"Inverse transformed predictions shape: {future_predictions_inv.shape}")
        
        print("\nPredicted prices for the next", params['prediction_days'], "days:")
        for i, price in enumerate(future_predictions_inv, 1):
            print(f"Day {i}: ${price:.2f}")
            
        if output_dir:
            # Plot test vs predicted
            plt.figure(figsize=(15, 8))
            
            # Plot the full test prediction comparison
            plt.subplot(2, 1, 1)
            plt.plot(y_test_inv, label='Actual')
            plt.plot(y_pred_inv[:len(y_test)], label='Predicted')
            plt.title(f'{symbol} Stock Price - Test vs Predicted (Full)')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            
            # Plot the most recent test data + future predictions
            plt.subplot(2, 1, 2)
            recent_window = 50
            plt.plot(y_test_inv[-recent_window:], label='Actual') 
            plt.plot(range(recent_window), y_pred_inv[-recent_window:], label='Predicted')
            
            # Plot future predictions
            future_x = range(recent_window, recent_window + len(future_predictions_inv))
            plt.plot(future_x, future_predictions_inv, label='Future Predictions', linestyle='--')
            
            # Add confidence intervals (simple approximation)
            std_dev = np.std(np.abs(y_test - y_pred.flatten()))
            lower_bound = future_predictions_inv - 1.96 * std_dev
            upper_bound = future_predictions_inv + 1.96 * std_dev
            plt.fill_between(future_x, lower_bound, upper_bound, alpha=0.2, color='red')
            
            plt.title(f'{symbol} Stock Price - Recent and Future Predictions')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_predictions.png'))
            plt.close()
            
            # Create a residual plot
            plt.figure(figsize=(12, 6))
            residuals = y_test - y_pred.flatten()
            plt.scatter(y_pred.flatten(), residuals)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Residual Plot')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.savefig(os.path.join(output_dir, f'{symbol}_residuals.png'))
            plt.close()
            
            # Save predictions to CSV
            dates = pd.date_range(start=datetime.now(), periods=len(future_predictions_inv))
            predictions_df = pd.DataFrame({
                'Date': dates,
                'Predicted_Price': future_predictions_inv,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound
            })
            predictions_df.to_csv(os.path.join(output_dir, f'{symbol}_predictions.csv'), index=False)
        
        # Calculate risk assessment
        risk_assessment = assess_risk(
            params['volatility'], 
            df_original['Close'].values,
            future_predictions_inv
        )
        
        print("\nRisk Assessment:")
        print(f"Volatility Risk: {risk_assessment['volatility_risk']}")
        print(f"Predicted Price Change: {risk_assessment['predicted_change']:.2%}")
        print(f"Downside Risk: {risk_assessment['downside_risk']:.2%}")
        print(f"Risk-Adjusted Return: {risk_assessment['risk_adjusted_return']:.2f}")
        print(f"Overall Risk Level: {risk_assessment['overall_risk']}")
        
        # Perform sector analysis
        sector_analysis = perform_sector_analysis(symbol)
        print(f"\nSector Analysis:")
        print(f"Sector: {sector_analysis['sector']}")
        print(f"Stock Return: {sector_analysis['stock_return']:.2%}")
        print(f"Sector Return: {sector_analysis['sector_return']:.2%}")
        print(f"Alpha: {sector_analysis['alpha']:.2%}")
        print(f"Beta: {sector_analysis['beta']:.2f}")
        
        # Create dashboard summary
        dashboard_fig = create_dashboard_summary(
            symbol, 
            df_original, 
            signals, 
            future_predictions_inv,
            risk_assessment,
            sector_analysis
        )
        
        if output_dir:
            dashboard_fig.savefig(os.path.join(output_dir, f'{symbol}_dashboard.png'))
            plt.close(dashboard_fig)
        
        # Return a properly formatted result dictionary
        results = {
            'symbol': symbol,
            'company_name': stock_info.get('shortName', symbol),
            'current_price': stock_info.get('regularMarketPrice', 0),
            'metrics': {
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            },
            'model_params': params,
            'future_predictions': {
                'dates': pd.date_range(start=datetime.now(), periods=len(future_predictions_inv)),
                'prices': future_predictions_inv,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            },
            'trading_signals': {
                'current_signal': float(current_signal),
                'recommendation': current_recommendation,
                'rsi_signal': float(signals['rsi_signal'].iloc[-1]),
                'ma_signal': float(signals['ma_signal'].iloc[-1]), 
                'macd_signal': float(signals['macd_signal'].iloc[-1])
            },
            'risk_assessment': risk_assessment,
            'sector_analysis': sector_analysis
        }
        
        return results
        
    except Exception as e:
        print(f"Error analyzing stock: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return None

def main():
    """Main function to run the stock analysis from command line."""
    parser = argparse.ArgumentParser(description='Analyze stock data and make predictions')
    parser.add_argument('symbol', type=str, help='Stock symbol to analyze (e.g., AAPL)')
    parser.add_argument('--output', type=str, help='Directory to save results', default='results')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed report')
    
    args = parser.parse_args()
    
    output_dir = None if args.no_plots else args.output
    
    try:
        results = analyze_stock(args.symbol, output_dir, args.start_date, args.end_date)
        if results:
            # Save results to file
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                results_file = os.path.join(args.output, f'{args.symbol}_analysis.txt')
                
                with open(results_file, 'w') as f:
                    # Report header
                    f.write(f"═════════════════════════════════════════════════════════\n")
                    f.write(f"  STOCK ANALYSIS REPORT FOR {results['company_name'].upper()} ({args.symbol})\n")
                    f.write(f"═════════════════════════════════════════════════════════\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Current stock information
                    f.write(f"CURRENT STOCK INFORMATION\n")
                    f.write(f"━━━━━━━━━━━━━━━━━━━━━━━━\n")
                    f.write(f"Current Price: ${results['current_price']:.2f}\n")
                    f.write(f"Volatility: {results['model_params']['volatility']:.4f}\n")
                    
                    if 'stock_info' in results and results['stock_info'] is not None:
                        stock_info = results['stock_info']
                        if 'marketCap' in stock_info:
                            market_cap = stock_info['marketCap']
                            if market_cap >= 1e12:
                                market_cap_str = f"${market_cap/1e12:.2f}T"
                            elif market_cap >= 1e9:
                                market_cap_str = f"${market_cap/1e9:.2f}B"
                            else:
                                market_cap_str = f"${market_cap/1e6:.2f}M"
                            f.write(f"Market Cap: {market_cap_str}\n")
                            
                        if 'fiftyTwoWeekHigh' in stock_info and 'fiftyTwoWeekLow' in stock_info:
                            f.write(f"52-Week Range: ${stock_info['fiftyTwoWeekLow']:.2f} - ${stock_info['fiftyTwoWeekHigh']:.2f}\n")
                            
                        if 'averageVolume' in stock_info:
                            avg_vol = stock_info['averageVolume']
                            if avg_vol >= 1e6:
                                avg_vol_str = f"{avg_vol/1e6:.2f}M"
                            else:
                                avg_vol_str = f"{avg_vol/1e3:.2f}K"
                            f.write(f"Average Volume: {avg_vol_str}\n")
                            
                        if 'dividendYield' in stock_info and stock_info['dividendYield'] is not None:
                            f.write(f"Dividend Yield: {stock_info['dividendYield']:.2%}\n")
                            
                        if 'trailingPE' in stock_info and stock_info['trailingPE'] is not None:
                            f.write(f"P/E Ratio: {stock_info['trailingPE']:.2f}\n")
                    f.write("\n")
                    
                    # Trading recommendation
                    f.write(f"TRADING RECOMMENDATION\n")
                    f.write(f"━━━━━━━━━━━━━━━━━━━━\n")
                    
                    recommendation = results['trading_signals']['recommendation']
                    signal_strength = results['trading_signals']['current_signal']
                    
                    # Format recommendation with emphasis
                    if recommendation == "Buy":
                        rec_formatted = "BUY"
                    elif recommendation == "Sell":
                        rec_formatted = "SELL"
                    else:
                        rec_formatted = "HOLD"
                        
                    f.write(f"Current Recommendation: {rec_formatted}\n")
                    f.write(f"Signal Strength: {signal_strength:.2f}\n")
                    f.write(f"RSI Signal: {'Bullish' if results['trading_signals']['rsi_signal'] > 0 else 'Bearish' if results['trading_signals']['rsi_signal'] < 0 else 'Neutral'}\n")
                    f.write(f"Moving Average Signal: {'Bullish' if results['trading_signals']['ma_signal'] > 0 else 'Bearish' if results['trading_signals']['ma_signal'] < 0 else 'Neutral'}\n")
                    f.write(f"MACD Signal: {'Bullish' if results['trading_signals']['macd_signal'] > 0 else 'Bearish' if results['trading_signals']['macd_signal'] < 0 else 'Neutral'}\n\n")
                    
                    # Risk assessment
                    f.write(f"RISK ASSESSMENT\n")
                    f.write(f"━━━━━━━━━━━━━━\n")
                    f.write(f"Volatility Risk: {results['risk_assessment']['volatility_risk']}\n")
                    f.write(f"Predicted Change: {results['risk_assessment']['predicted_change']:.2%}\n")
                    f.write(f"Downside Risk: {results['risk_assessment']['downside_risk']:.2%}\n")
                    f.write(f"Risk-Adjusted Return: {results['risk_assessment']['risk_adjusted_return']:.2f}\n")
                    f.write(f"Overall Risk Level: {results['risk_assessment']['overall_risk']}\n\n")
                    
                    # Sector comparison
                    f.write(f"SECTOR COMPARISON\n")
                    f.write(f"━━━━━━━━━━━━━━━━\n")
                    f.write(f"Sector: {results['sector_analysis']['sector']}\n")
                    f.write(f"Sector ETF: {results['sector_analysis']['sector_etf']}\n")
                    f.write(f"Stock Return: {results['sector_analysis']['stock_return']:.2%}\n")
                    f.write(f"Sector Return: {results['sector_analysis']['sector_return']:.2%}\n")
                    f.write(f"Relative Performance: {results['sector_analysis']['relative_performance']:.2%}\n")
                    f.write(f"Alpha: {results['sector_analysis']['alpha']:.2%}\n")
                    f.write(f"Beta: {results['sector_analysis']['beta']:.2f}\n")
                    if 'stock_sharpe' in results['sector_analysis']:
                        f.write(f"Stock Sharpe Ratio: {results['sector_analysis']['stock_sharpe']:.2f}\n")
                        f.write(f"Sector Sharpe Ratio: {results['sector_analysis']['sector_sharpe']:.2f}\n")
                    f.write("\n")
                    
                    # Model performance metrics
                    f.write(f"MODEL PERFORMANCE METRICS\n")
                    f.write(f"━━━━━━━━━━━━━━━━━━━━━━━\n")
                    f.write(f"Mean Absolute Error: {results['metrics']['mae']:.4f}\n")
                    f.write(f"Mean Squared Error: {results['metrics']['mse']:.4f}\n")
                    f.write(f"Root Mean Squared Error: {results['metrics']['rmse']:.4f}\n\n")
                    
                    # Model parameters
                    f.write(f"MODEL PARAMETERS\n")
                    f.write(f"━━━━━━━━━━━━━━━\n")
                    f.write(f"Sequence Length: {results['model_params']['sequence_length']}\n")
                    f.write(f"Prediction Horizon: {results['model_params']['prediction_days']} days\n\n")
                    
                    # Future price predictions
                    f.write(f"FUTURE PRICE PREDICTIONS\n")
                    f.write(f"━━━━━━━━━━━━━━━━━━━━━\n")
                    f.write("Date            | Price    | 95% Confidence Interval\n")
                    f.write("-------------------------------------------------\n")
                    
                    for date, price, lower, upper in zip(
                        results['future_predictions']['dates'],
                        results['future_predictions']['prices'],
                        results['future_predictions']['lower_bound'],
                        results['future_predictions']['upper_bound']):
                        f.write(f"{date.strftime('%Y-%m-%d')} | ${price:.2f}  | (${lower:.2f} - ${upper:.2f})\n")
                    f.write("\n")
                    
                    # Analysis conclusion
                    f.write(f"ANALYSIS CONCLUSION\n")
                    f.write(f"━━━━━━━━━━━━━━━━━\n")
                    
                    # Generate conclusion text based on all the analysis
                    conclusion = []
                    
                    # Trading conclusion
                    if recommendation == "Buy":
                        conclusion.append(f"Technical indicators suggest a BUY recommendation for {args.symbol} with a signal strength of {signal_strength:.2f}.")
                    elif recommendation == "Sell":
                        conclusion.append(f"Technical indicators suggest a SELL recommendation for {args.symbol} with a signal strength of {signal_strength:.2f}.")
                    else:
                        conclusion.append(f"Technical indicators suggest a HOLD recommendation for {args.symbol} with a neutral signal strength of {signal_strength:.2f}.")
                    
                    # Risk conclusion
                    risk_level = results['risk_assessment']['overall_risk']
                    if risk_level == "Low":
                        conclusion.append(f"The stock shows low risk characteristics with a volatility of {results['model_params']['volatility']:.4f} and a positive risk-adjusted return of {results['risk_assessment']['risk_adjusted_return']:.2f}.")
                    elif risk_level == "Medium":
                        conclusion.append(f"The stock shows moderate risk with a volatility of {results['model_params']['volatility']:.4f} and a risk-adjusted return of {results['risk_assessment']['risk_adjusted_return']:.2f}.")
                    else:
                        conclusion.append(f"The stock shows high risk characteristics with a volatility of {results['model_params']['volatility']:.4f} and a risk-adjusted return of {results['risk_assessment']['risk_adjusted_return']:.2f}.")
                    
                    # Sector comparison conclusion
                    rel_perf = results['sector_analysis']['relative_performance']
                    if rel_perf > 0:
                        conclusion.append(f"{args.symbol} has outperformed its sector by {rel_perf:.2%} with a beta of {results['sector_analysis']['beta']:.2f} and alpha of {results['sector_analysis']['alpha']:.2%}.")
                    else:
                        conclusion.append(f"{args.symbol} has underperformed its sector by {-rel_perf:.2%} with a beta of {results['sector_analysis']['beta']:.2f} and alpha of {results['sector_analysis']['alpha']:.2%}.")
                    
                    # Future prediction conclusion
                    future_prices = results['future_predictions']['prices']
                    current_price = results['current_price']
                    price_change = (future_prices[-1] - current_price) / current_price
                    
                    if price_change > 0:
                        conclusion.append(f"The model predicts a {price_change:.2%} increase in price over the next {results['model_params']['prediction_days']} days, with an upward trend.")
                    else:
                        conclusion.append(f"The model predicts a {-price_change:.2%} decrease in price over the next {results['model_params']['prediction_days']} days, with a downward trend.")
                    
                    # Overall conclusion
                    if recommendation == "Buy" and risk_level != "High" and price_change > 0:
                        conclusion.append(f"Overall Analysis: POSITIVE - {args.symbol} shows strong buy signals with acceptable risk and positive price momentum.")
                    elif recommendation == "Sell" and price_change < 0:
                        conclusion.append(f"Overall Analysis: NEGATIVE - {args.symbol} shows sell signals with projected price decline.")
                    else:
                        conclusion.append(f"Overall Analysis: NEUTRAL - {args.symbol} shows mixed signals. Consider your investment strategy, risk tolerance, and market conditions before making a decision.")
                    
                    # Write conclusions
                    for line in conclusion:
                        f.write(f"{line}\n")
                    
                    # Disclaimer
                    f.write("\n")
                    f.write("DISCLAIMER\n")
                    f.write("━━━━━━━━━━\n")
                    f.write("This analysis is for informational purposes only and not a recommendation to buy or sell any securities.\n")
                    f.write("Past performance does not guarantee future results. All investments involve risk.\n")
                    f.write("Always conduct your own research and consider seeking advice from a financial professional.\n")
                
                print(f"\nResults saved to {results_file}")
                print(f"Visualizations saved to {args.output} directory")
                if not args.no_plots:
                    print(f"Dashboard saved as {args.symbol}_dashboard.png")
    except Exception as e:
        print(f"Error analyzing stock: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 