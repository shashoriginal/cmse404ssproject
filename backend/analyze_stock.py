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

def analyze_stock(symbol, output_dir=None, start_date=None, end_date=None, save_data=True):
    """Analyze stock data and make predictions.
    
    Args:
        symbol (str): Stock symbol
        output_dir (str, optional): Directory to save results
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        save_data (bool): Whether to save original data for dashboard
        
    Returns:
        dict: Analysis results
    """
    try:
        print(f"\nAnalyzing {symbol}...")
        
        # Download and prepare stock data
        data, scaler, stock_info = prepare_stock_data(symbol, start_date, end_date)
        
        # Keep a copy of the original dataframe
        df_original = None
        if save_data and stock_info is not None:
            # Recreate original dataframe for visualization
            stock = yf.Ticker(symbol)
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=730)  # 2 years
                
            df_original = stock.history(start=start_date, end=end_date)
            
            # Calculate additional features
            df_original['Returns'] = df_original['Close'].pct_change()
            df_original['MA5'] = df_original['Close'].rolling(window=5).mean()
            df_original['MA20'] = df_original['Close'].rolling(window=20).mean()
            df_original['MA50'] = df_original['Close'].rolling(window=50).mean()
            df_original['RSI'] = calculate_rsi(df_original['Close'])
            df_original['MACD'] = calculate_macd(df_original['Close'])
            
            # Drop NaN values
            df_original = df_original.dropna()
        
        # Determine optimal parameters
        params = determine_optimal_parameters(df_original if df_original is not None else pd.DataFrame())
        sequence_length = params['sequence_length']
        prediction_days = params['prediction_days']
        
        # Prepare data for LSTM
        print(f"Preparing data with sequence length {sequence_length}...")
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            # Predict the closing price (index 3)
            y.append(data[i + sequence_length, 3])
            
        X, y = np.array(X), np.array(y)
        
        # Split data into training and testing sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Create and train LSTM model
        print("Training LSTM model...")
        
        # Initialize model with configuration from model_config.yaml
        config_path = os.path.join('configs', 'model_config.yaml')
        if os.path.exists(config_path):
            model = LSTMModel(config_path)
        else:
            model = LSTMModel()
        
        # Get the sequence length from the model's configuration
        model_sequence_length = model.config['model']['sequence_length']
        
        # If the determined sequence length doesn't match the model's, use the model's
        if sequence_length != model_sequence_length:
            print(f"Adjusting sequence length from {sequence_length} to {model_sequence_length} to match model configuration")
            sequence_length = model_sequence_length
            params['sequence_length'] = model_sequence_length
        
        # Get the input shape for building the model
        input_shape = (sequence_length, X_train.shape[2])
        model.build_model(input_shape)
        
        # Train the model
        device = model.device
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Train the model
        model.train_model(train_loader, val_loader=None, num_epochs=50)
        
        # Evaluate model
        print("Evaluating model performance...")
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
        
        # Calculate MSE, MAE, RMSE
        mse = ((y_pred - y_test) ** 2).mean().item()
        mae = (y_pred - y_test).abs().mean().item()
        rmse = np.sqrt(mse)
        
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Create visualizations
        if output_dir and save_data:
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.cpu().numpy(), label='Actual')
            plt.plot(y_pred.cpu().numpy(), label='Predicted')
            plt.title(f'{symbol} - Actual vs Predicted')
            plt.xlabel('Test Sample')
            plt.ylabel('Scaled Price')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{symbol}_predictions.png'))
            plt.close()
            
            # Plot residuals
            plt.figure(figsize=(10, 6))
            residuals = (y_pred - y_test).cpu().numpy()
            plt.hist(residuals, bins=50)
            plt.title(f'{symbol} - Prediction Residuals')
            plt.xlabel('Residual')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{symbol}_residuals.png'))
            plt.close()
            
            # Plot training history if available
            if hasattr(model, 'history') and model.history:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                if 'loss' in model.history:
                    plt.plot(model.history['loss'], label='Train Loss')
                if 'val_loss' in model.history:
                    plt.plot(model.history['val_loss'], label='Validation Loss')
                plt.title('Loss During Training')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                if 'mae' in model.history:
                    plt.plot(model.history['mae'], label='Train MAE')
                if 'val_mae' in model.history:
                    plt.plot(model.history['val_mae'], label='Validation MAE')
                plt.title('MAE During Training')
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{symbol}_training_history.png'))
                plt.close()
        
        # Calculate trading signals from the original data
        signals = None
        if df_original is not None:
            print("Calculating trading signals...")
            signals = calculate_trading_signals(df_original)
            current_signal = signals['overall_signal'].iloc[-1]
            current_recommendation = signals['recommendation'].iloc[-1]
            
            # Save signals data for dashboard if requested
            if output_dir and save_data:
                os.makedirs(output_dir, exist_ok=True)
                signals.to_csv(os.path.join(output_dir, f"{symbol}_signals.csv"))
                df_original.to_csv(os.path.join(output_dir, f"{symbol}_data.csv"))
        else:
            # Default values if original data is not available
            current_signal = 0
            current_recommendation = "Hold"
        
        # Print current trading signals
        print(f"\nCurrent Trading Signals for {symbol}:")
        if signals is not None:
            print(f"RSI Signal: {'Bullish' if signals['rsi_signal'].iloc[-1] > 0 else 'Bearish' if signals['rsi_signal'].iloc[-1] < 0 else 'Neutral'}")
            print(f"Moving Average Signal: {'Bullish' if signals['ma_signal'].iloc[-1] > 0 else 'Bearish' if signals['ma_signal'].iloc[-1] < 0 else 'Neutral'}")
            print(f"MACD Signal: {'Bullish' if signals['macd_signal'].iloc[-1] > 0 else 'Bearish' if signals['macd_signal'].iloc[-1] < 0 else 'Neutral'}")
            print(f"Overall Recommendation: {current_recommendation}")
        
        # Make future predictions
        print(f"\nGenerating {prediction_days} days of future predictions...")
        
        # Prepare the last sequence from the data with the correct sequence length
        last_sequence = torch.FloatTensor(data[-sequence_length:]).unsqueeze(0).to(device)
        
        # Use the model's predict_sequence method
        future_predictions = model.predict_sequence(last_sequence, steps=prediction_days)
        
        # Inverse transform predictions to get actual price values
        # We need to reconstruct full feature vectors for inverse transformation
        print("Transforming predictions back to price scale...")
        
        # Get the last actual data point as a template
        future_template = data[-1].copy()
        
        # Create array to hold all predicted values with all features
        future_points = np.array([future_template] * len(future_predictions))
        
        # Update the Close price (index 3) with our predictions
        for i, pred in enumerate(future_predictions):
            future_points[i, 3] = pred
        
        # Inverse transform to get actual prices
        future_predictions_inv = scaler.inverse_transform(future_points)[:, 3]
        
        # Calculate confidence intervals (95%)
        std_dev = np.std(y_test.cpu().numpy() - y_pred.cpu().numpy())
        lower_bound = future_predictions_inv - (1.96 * std_dev)
        upper_bound = future_predictions_inv + (1.96 * std_dev)
        
        # Print predictions
        print("\nFuture Price Predictions:")
        for i, price in enumerate(future_predictions_inv):
            pred_date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
            print(f"{pred_date}: ${price:.2f} (95% CI: ${lower_bound[i]:.2f} - ${upper_bound[i]:.2f})")
        
        # Save predictions to CSV
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            pred_dates = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(future_predictions_inv))]
            pred_df = pd.DataFrame({
                'Date': pred_dates,
                'Predicted_Price': future_predictions_inv,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound
            })
            pred_df.to_csv(os.path.join(output_dir, f"{symbol}_predictions.csv"), index=False)
            print(f"\nPredictions saved to {os.path.join(output_dir, f'{symbol}_predictions.csv')}")
        
        # Calculate risk assessment
        print("\nAssessing risk profile...")
        recent_prices = df_original['Close'].values[-30:] if df_original is not None else []
        risk_assessment = assess_risk(params['volatility'], recent_prices, future_predictions_inv)
        
        # Print risk assessment
        print(f"Volatility Risk: {risk_assessment['volatility_risk']}")
        print(f"Predicted Price Change: {risk_assessment['predicted_change']:.2%}")
        print(f"Overall Risk Level: {risk_assessment['overall_risk']}")
        
        # Perform sector analysis
        print("\nComparing against sector performance...")
        sector_analysis = perform_sector_analysis(symbol)
        
        # Print sector comparison
        print(f"Sector: {sector_analysis['sector']}")
        print(f"Stock Return (1Y): {sector_analysis['stock_return']:.2%}")
        print(f"Sector Return (1Y): {sector_analysis['sector_return']:.2%}")
        print(f"Alpha: {sector_analysis['alpha']:.2%}")
        print(f"Beta: {sector_analysis['beta']:.2f}")
        
        # Create visualizations
        if output_dir and save_data:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot correlation matrix
            plt.figure(figsize=(12, 10))
            plot_correlation_matrix(df_original)
            plt.title(f'{symbol} Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_correlation.png'))
            plt.close()
            
            # Plot feature distributions
            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(['Close', 'Volume', 'Returns', 'RSI', 'MACD']):
                plt.subplot(2, 3, i+1)
                sns.histplot(df_original[feature], kde=True)
                plt.title(f'{feature} Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_distributions.png'))
            plt.close()
            
            # Plot time series of key features
            plt.figure(figsize=(15, 15))
            
            # Price and Moving Averages
            plt.subplot(3, 1, 1)
            plt.plot(df_original.index, df_original['Close'], label='Close')
            plt.plot(df_original.index, df_original['MA20'], label='20-day MA')
            plt.plot(df_original.index, df_original['MA50'], label='50-day MA')
            plt.title(f'{symbol} Price and Moving Averages')
            plt.legend()
            
            # RSI
            plt.subplot(3, 1, 2)
            plt.plot(df_original.index, df_original['RSI'], label='RSI')
            plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            plt.title('Relative Strength Index (RSI)')
            plt.legend()
            
            # MACD
            plt.subplot(3, 1, 3)
            plt.plot(df_original.index, df_original['MACD'], label='MACD')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            plt.title('Moving Average Convergence Divergence (MACD)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_technical_indicators.png'))
            plt.close()
            
            # Plot trading signals
            plt.figure(figsize=(15, 10))
            
            # Prices with buy/sell markers
            plt.subplot(2, 1, 1)
            plt.plot(df_original.index, df_original['Close'], label='Close Price')
            
            # Add buy and sell markers
            buy_signals = signals[signals['recommendation'] == 'Buy']
            sell_signals = signals[signals['recommendation'] == 'Sell']
            
            plt.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=100, label='Buy Signal')
            plt.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=100, label='Sell Signal')
            
            plt.title(f'{symbol} Price with Trading Signals')
            plt.legend()
            
            # Signal strength over time
            plt.subplot(2, 1, 2)
            plt.plot(signals.index, signals['overall_signal'], label='Signal Strength')
            plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Buy Threshold')
            plt.axhline(y=-0.5, color='r', linestyle='--', alpha=0.5, label='Sell Threshold')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            plt.title('Trading Signal Strength Over Time')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_trading_signals.png'))
            plt.close()
            
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
            'stock_info': stock_info,
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
                'rsi_signal': float(signals['rsi_signal'].iloc[-1]) if signals is not None else 0,
                'ma_signal': float(signals['ma_signal'].iloc[-1]) if signals is not None else 0,
                'macd_signal': float(signals['macd_signal'].iloc[-1]) if signals is not None else 0
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

def backtest_strategy(symbol, start_date=None, end_date=None, output_dir=None, initial_capital=10000):
    """
    Backtest the trading strategy based on model predictions.
    
    Args:
        symbol (str): Stock symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        output_dir (str, optional): Directory to save backtest results
        initial_capital (float): Initial investment amount
        
    Returns:
        dict: Backtest performance metrics
    """
    print(f"\nBacktesting trading strategy for {symbol}...")
    
    try:
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
        
        # Calculate features needed for trading signals
        df['Returns'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'] = calculate_macd(df['Close'])
        
        # Drop any rows with NaN values
        df = df.dropna().copy()
        
        # Initialize model and parameters
        params = determine_optimal_parameters(df)
        sequence_length = params['sequence_length']
        
        # Set up strategy parameters
        lookback_window = sequence_length
        prediction_window = 5  # Number of days to predict ahead
        
        # Prepare backtest dataframe
        backtest_df = df.copy()
        backtest_df['Position'] = 0   # 1: Long, -1: Short, 0: No position
        backtest_df['Signal'] = 0     # 1: Buy, -1: Sell, 0: Hold
        backtest_df['PredictedReturn'] = np.nan
        
        # Prepare data for model
        # Select features for prediction
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'Returns', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD']
        
        if all(feature in backtest_df.columns for feature in features):
            data = backtest_df[features].values
        else:
            available_features = [f for f in features if f in backtest_df.columns]
            data = backtest_df[available_features].values
            print(f"Warning: Using only available features: {available_features}")
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Load or create model
        config_path = os.path.join('configs', 'model_config.yaml')
        if os.path.exists(config_path):
            model = LSTMModel(config_path)
        else:
            model = LSTMModel()  # Use default config
        
        # Set up model parameters
        input_shape = (sequence_length, data.shape[1])
        model.build_model(input_shape)
        
        model_path = os.path.join('models', f'{symbol}_lstm_model.pth')
        
        if not os.path.exists(model_path):
            # Train model if it doesn't exist
            print(f"Model not found. Training new model for {symbol}...")
            
            # Prepare data for training
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length - prediction_window):
                X.append(scaled_data[i:i+sequence_length])
                # Predict the closing price
                # Find the index of 'Close' in the features list
                close_idx = features.index('Close') if 'Close' in features else 3  # Default to 3 if missing
                y.append(scaled_data[i+sequence_length+prediction_window, close_idx])
                
            X, y = np.array(X), np.array(y)
            
            # Split the data
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train).unsqueeze(1)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val).unsqueeze(1)
            
            # Create datasets and data loaders
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Train model
            model.train_model(train_loader, val_loader=val_loader, num_epochs=50)
            
            # Save model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
        else:
            model.load(model_path)
        
        device = model.device
        
        # Perform backtesting
        print(f"Running backtest from {start_date} to {end_date}...")
        
        # We need at least lookback_window + prediction_window days before making predictions
        start_idx = lookback_window + prediction_window
        
        positions = []
        capital = initial_capital
        portfolio_values = [capital]
        cash = capital
        shares = 0
        trades = []
        entry_price = 0
        
        # For each trading day in our backtest period
        for i in range(start_idx, len(backtest_df) - prediction_window):
            current_date = backtest_df.index[i]
            current_price = backtest_df['Close'].iloc[i]
            
            # Get the sequence for prediction
            sequence = scaled_data[i-lookback_window:i]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                predicted_scaled = model(sequence_tensor)
                
            # Convert prediction back to price scale (denormalize)
            # We need to reconstruct a full feature vector to inverse transform
            close_idx = features.index('Close') if 'Close' in features else 3  # Default to 3 if missing
            
            # Create a template of the current features
            last_features = scaled_data[i].copy()
            # Update closing price with the prediction
            last_features[close_idx] = predicted_scaled.item()
            # Inverse transform to get the actual price
            predicted_features = scaler.inverse_transform(last_features.reshape(1, -1))
            predicted_price = predicted_features[0, close_idx]
            
            # Calculate predicted return
            predicted_return = (predicted_price / current_price) - 1
            backtest_df.loc[current_date, 'PredictedReturn'] = predicted_return
            
            # Generate signals based on predicted return
            if predicted_return > 0.01:  # 1% threshold for buy
                backtest_df.loc[current_date, 'Signal'] = 1
                
                # If we don't have a position, enter a long position
                if backtest_df.loc[backtest_df.index[i-1], 'Position'] <= 0:
                    backtest_df.loc[current_date, 'Position'] = 1
                    
                    # Record trade
                    if cash > 0:
                        entry_price = current_price
                        shares_to_buy = cash // current_price
                        shares += shares_to_buy
                        cash -= shares_to_buy * current_price
                        trades.append({
                            'date': current_date,
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': shares_to_buy * current_price
                        })
            
            elif predicted_return < -0.01:  # -1% threshold for sell
                backtest_df.loc[current_date, 'Signal'] = -1
                
                # If we have a long position, exit
                if backtest_df.loc[backtest_df.index[i-1], 'Position'] >= 1:
                    backtest_df.loc[current_date, 'Position'] = -1
                    
                    # Record trade
                    if shares > 0:
                        exit_price = current_price
                        cash += shares * current_price
                        trades.append({
                            'date': current_date,
                            'type': 'sell',
                            'price': current_price,
                            'shares': shares,
                            'value': shares * current_price,
                            'profit': shares * (exit_price - entry_price)
                        })
                        shares = 0
            
            else:
                # Hold current position
                backtest_df.loc[current_date, 'Signal'] = 0
                backtest_df.loc[current_date, 'Position'] = backtest_df.loc[backtest_df.index[i-1], 'Position']
            
            # Calculate portfolio value
            portfolio_value = cash + (shares * current_price)
            portfolio_values.append(portfolio_value)
            positions.append(backtest_df.loc[current_date, 'Position'])
        
        # Calculate backtest metrics
        backtest_df = backtest_df.iloc[start_idx:-prediction_window].copy()
        
        # Fix the portfolio values length to match the dataframe index
        if len(portfolio_values) != len(backtest_df):
            print(f"Adjusting portfolio values length from {len(portfolio_values)} to {len(backtest_df)}")
            portfolio_values = portfolio_values[:len(backtest_df)]
            
        backtest_df['PortfolioValue'] = portfolio_values
        backtest_df['CumulativeReturn'] = (backtest_df['PortfolioValue'] / initial_capital) - 1
        
        # Calculate benchmark (buy and hold)
        initial_price = backtest_df['Close'].iloc[0]
        final_price = backtest_df['Close'].iloc[-1]
        benchmark_return = (final_price / initial_price) - 1
        
        # Calculate performance metrics
        total_days = len(backtest_df)
        trading_days_per_year = 252
        years = total_days / trading_days_per_year
        
        # Total return
        total_return = (portfolio_values[-1] / initial_capital) - 1
        
        # Annualized return
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Volatility
        daily_returns = backtest_df['PortfolioValue'].pct_change().dropna()
        volatility = daily_returns.std()
        annualized_volatility = volatility * (trading_days_per_year ** 0.5)
        
        # Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / rolling_max) - 1
        max_drawdown = drawdowns.min()
        
        # Filter out trades with zero shares (signals that didn't result in actual trades)
        actual_trades = [trade for trade in trades if trade.get('shares', 0) > 0]
        sell_trades = [trade for trade in actual_trades if trade.get('type') == 'sell' and 'profit' in trade]
        
        # Win rate based on completed trades (buy and sell pairs)
        profitable_trades = sum(1 for trade in sell_trades if trade.get('profit', 0) > 0)
        win_rate = profitable_trades / len(sell_trades) if sell_trades else 0
        
        # Calculate final metrics
        metrics = {
            'initial_capital': initial_capital,
            'final_value': portfolio_values[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': len(actual_trades),
            'completed_trades': len(sell_trades),
            'win_rate': win_rate,
            'duration': f"{total_days} days ({years:.2f} years)"
        }
        
        # Generate visualizations
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot portfolio value and benchmark
            plt.figure(figsize=(12, 6))
            plt.plot(backtest_df.index, backtest_df['PortfolioValue'], label='Strategy')
            plt.plot(backtest_df.index, initial_capital * (1 + benchmark_return * (backtest_df.index - backtest_df.index[0]).days / (backtest_df.index[-1] - backtest_df.index[0]).days), label='Buy and Hold')
            plt.title(f'Backtest Results: {symbol} Strategy vs Buy and Hold')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_backtest_performance.png'))
            plt.close()
            
            # Plot signals and positions
            plt.figure(figsize=(12, 8))
            
            # Price and signals subplot
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(backtest_df.index, backtest_df['Close'], label='Close Price')
            
            # Plot buy signals
            buy_signals = backtest_df[backtest_df['Signal'] == 1]
            ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal')
            
            # Plot sell signals
            sell_signals = backtest_df[backtest_df['Signal'] == -1]
            ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell Signal')
            
            ax1.set_title(f'{symbol} Price and Trading Signals')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True)
            
            # Position and portfolio value subplot
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(backtest_df.index, backtest_df['CumulativeReturn'] * 100, label='Strategy Return (%)')
            ax2.set_title(f'{symbol} Cumulative Return')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Return (%)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_backtest_signals.png'))
            plt.close()
            
            # Plot trade distribution
            if trades:
                profits = [trade.get('profit', 0) for trade in trades if 'profit' in trade]
                
                plt.figure(figsize=(10, 6))
                plt.hist(profits, bins=20, color='skyblue', edgecolor='black')
                plt.axvline(0, color='red', linestyle='--')
                plt.title(f'{symbol} Trade Profit Distribution')
                plt.xlabel('Profit ($)')
                plt.ylabel('Number of Trades')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{symbol}_backtest_trades.png'))
                plt.close()
                
                # Save trade log
                trade_df = pd.DataFrame(trades)
                trade_df.to_csv(os.path.join(output_dir, f'{symbol}_trade_log.csv'), index=False)
            
            # Save backtest results
            backtest_df.to_csv(os.path.join(output_dir, f'{symbol}_backtest_results.csv'))
            
            # Save backtest metrics
            metrics_file = os.path.join(output_dir, f'{symbol}_backtest_metrics.txt')
            with open(metrics_file, 'w', encoding='utf-8') as f:
                f.write(f"=================================================\n")
                f.write(f"  BACKTEST RESULTS FOR {symbol}\n")
                f.write(f"=================================================\n\n")
                f.write(f"Backtest Period: {backtest_df.index[0].strftime('%Y-%m-%d')} to {backtest_df.index[-1].strftime('%Y-%m-%d')} ({metrics['duration']})\n\n")
                
                f.write(f"PERFORMANCE METRICS\n")
                f.write(f"-------------------\n")
                f.write(f"Initial Capital: ${metrics['initial_capital']:,.2f}\n")
                f.write(f"Final Value: ${metrics['final_value']:,.2f}\n")
                f.write(f"Total Return: {metrics['total_return']:.2%}\n")
                f.write(f"Annualized Return: {metrics['annualized_return']:.2%}\n")
                f.write(f"Benchmark Return (Buy & Hold): {metrics['benchmark_return']:.2%}\n")
                f.write(f"Excess Return: {metrics['excess_return']:.2%}\n")
                f.write(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n")
                f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
                f.write(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n\n")
                
                f.write(f"TRADING STATISTICS\n")
                f.write(f"------------------\n")
                f.write(f"Total Trades: {metrics['trades']}\n")
                f.write(f"Completed Trades: {metrics['completed_trades']}\n")
                f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
                if sell_trades:
                    avg_profit = sum(trade.get('profit', 0) for trade in sell_trades) / len(sell_trades)
                    f.write(f"Average Profit per Trade: ${avg_profit:.2f}\n")
                    
                    avg_win = sum(trade.get('profit', 0) for trade in sell_trades if trade.get('profit', 0) > 0) / profitable_trades if profitable_trades else 0
                    losing_trades = len(sell_trades) - profitable_trades
                    avg_loss = sum(trade.get('profit', 0) for trade in sell_trades if trade.get('profit', 0) <= 0) / losing_trades if losing_trades else 0
                    f.write(f"Average Win: ${avg_win:.2f}\n")
                    f.write(f"Average Loss: ${avg_loss:.2f}\n")
                    
                    if avg_loss != 0:
                        profit_factor = abs(sum(trade.get('profit', 0) for trade in sell_trades if trade.get('profit', 0) > 0) / 
                                        sum(trade.get('profit', 0) for trade in sell_trades if trade.get('profit', 0) <= 0))
                        f.write(f"Profit Factor: {profit_factor:.2f}\n")
                
                f.write("\n")
                f.write("CONCLUSION\n")
                f.write("----------\n")
                
                # Generate conclusion
                if metrics['total_return'] > metrics['benchmark_return']:
                    f.write(f"The trading strategy outperformed the buy-and-hold approach by {metrics['excess_return']:.2%}.\n")
                else:
                    f.write(f"The trading strategy underperformed the buy-and-hold approach by {-metrics['excess_return']:.2%}.\n")
                
                if metrics['sharpe_ratio'] > 1:
                    f.write(f"With a Sharpe Ratio of {metrics['sharpe_ratio']:.2f}, the strategy shows good risk-adjusted returns.\n")
                else:
                    f.write(f"With a Sharpe Ratio of {metrics['sharpe_ratio']:.2f}, the strategy may not offer sufficient risk-adjusted returns.\n")
                
                if metrics['completed_trades'] > 0:
                    if metrics['win_rate'] > 0.5:
                        f.write(f"The win rate of {metrics['win_rate']:.2%} indicates a reliable signal generation mechanism.\n")
                    else:
                        f.write(f"The win rate of {metrics['win_rate']:.2%} suggests the signal generation needs improvement.\n")
                else:
                    f.write("No completed trades occurred during the backtest period.\n")
                
                f.write("\n")
                f.write("DISCLAIMER\n")
                f.write("----------\n")
                f.write("Past performance is not indicative of future results.\n")
                f.write("This backtest is based on historical data and does not account for trading costs, slippage, or market impact.\n")
                f.write("All investment strategies involve risk and may result in loss of capital.\n")
            
            print(f"\nBacktest results saved to {output_dir}")
        
        return metrics
        
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_comprehensive_dashboard(symbol, analysis_results, backtest_metrics, df_original, signals, future_predictions, output_dir):
    """
    Create a comprehensive dashboard showing both analysis and backtesting results.
    
    Args:
        symbol (str): Stock symbol
        analysis_results (dict): Results from analyze_stock function
        backtest_metrics (dict): Results from backtest_strategy function
        df_original (pd.DataFrame): Original stock price data
        signals (pd.DataFrame): Trading signals
        future_predictions (np.array): Future price predictions
        output_dir (str): Directory to save the dashboard
    """
    # Create a large figure for the dashboard
    plt.figure(figsize=(20, 16))
    plt.suptitle(f'Comprehensive Analysis & Backtest Dashboard: {symbol}', fontsize=20, y=0.98)
    
    # Main grid: 4 rows, 3 columns
    grid = plt.GridSpec(4, 3, hspace=0.4, wspace=0.3)
    
    # Row 1: Price Chart with Signals and Predictions
    ax_price = plt.subplot(grid[0, :])
    ax_price.plot(df_original.index, df_original['Close'], label='Close Price')
    
    # Add trading signals to the chart
    buy_signals = signals[signals['recommendation'] == 'Buy']
    sell_signals = signals[signals['recommendation'] == 'Sell']
    
    ax_price.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=100, label='Buy Signal')
    ax_price.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=100, label='Sell Signal')
    
    # Add future predictions
    last_date = df_original.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(future_predictions)+1)[1:]
    ax_price.plot(future_dates, future_predictions, 'b--', label='Predictions')
    
    # Add confidence intervals if available
    if 'lower_bound' in analysis_results['future_predictions'] and 'upper_bound' in analysis_results['future_predictions']:
        lb = analysis_results['future_predictions']['lower_bound']
        ub = analysis_results['future_predictions']['upper_bound']
        ax_price.fill_between(future_dates, lb, ub, color='blue', alpha=0.1, label='95% Confidence')
    
    ax_price.set_title(f'{symbol} Price, Signals & Predictions')
    ax_price.set_ylabel('Price ($)')
    ax_price.legend(loc='best')
    ax_price.grid(True)
    
    # Row 2, Col 1: Backtesting Performance
    if backtest_metrics:
        ax_backtest = plt.subplot(grid[1, 0:2])
        
        # Get backtest results from the saved file
        backtest_file = os.path.join(output_dir, f'{symbol}_backtest_results.csv')
        if os.path.exists(backtest_file):
            backtest_df = pd.read_csv(backtest_file, index_col=0, parse_dates=True)
            
            # Plot strategy vs buy & hold
            initial_capital = backtest_metrics['initial_capital']
            ax_backtest.plot(backtest_df.index, backtest_df['PortfolioValue'], label='Strategy')
            
            benchmark_values = initial_capital * (1 + backtest_df['Close'].pct_change().cumsum())
            ax_backtest.plot(backtest_df.index, benchmark_values, label='Buy and Hold')
            
            ax_backtest.set_title(f'Backtest Performance: Strategy vs Buy & Hold')
            ax_backtest.set_ylabel('Portfolio Value ($)')
            ax_backtest.legend()
            ax_backtest.grid(True)
        
        # Row 2, Col 3: Backtest Metrics
        ax_metrics = plt.subplot(grid[1, 2])
        metrics_text = [
            f"Initial Capital: ${backtest_metrics['initial_capital']:,.0f}",
            f"Final Value: ${backtest_metrics['final_value']:,.0f}",
            f"Total Return: {backtest_metrics['total_return']:.2%}",
            f"Buy & Hold: {backtest_metrics['benchmark_return']:.2%}",
            f"Excess Return: {backtest_metrics['excess_return']:.2%}",
            f"Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}",
            f"Max Drawdown: {backtest_metrics['max_drawdown']:.2%}",
            f"Win Rate: {backtest_metrics['win_rate']:.2%}",
            f"Total Trades: {backtest_metrics['trades']}"
        ]
        
        ax_metrics.axis('off')
        
        # Use different colors based on performance
        color_total = 'green' if backtest_metrics['total_return'] > 0 else 'red'
        color_excess = 'green' if backtest_metrics['excess_return'] > 0 else 'red'
        color_sharpe = 'green' if backtest_metrics['sharpe_ratio'] > 1 else 'orange' if backtest_metrics['sharpe_ratio'] > 0 else 'red'
        
        # Add metrics with colors
        ax_metrics.text(0.5, 0.95, "BACKTEST METRICS", ha='center', va='top', fontsize=14, fontweight='bold')
        ax_metrics.text(0.0, 0.85, metrics_text[0], fontsize=11)
        ax_metrics.text(0.0, 0.75, metrics_text[1], fontsize=11)
        ax_metrics.text(0.0, 0.65, metrics_text[2], fontsize=11, color=color_total, fontweight='bold')
        ax_metrics.text(0.0, 0.55, metrics_text[3], fontsize=11)
        ax_metrics.text(0.0, 0.45, metrics_text[4], fontsize=11, color=color_excess, fontweight='bold')
        ax_metrics.text(0.0, 0.35, metrics_text[5], fontsize=11, color=color_sharpe)
        ax_metrics.text(0.0, 0.25, metrics_text[6], fontsize=11)
        ax_metrics.text(0.0, 0.15, metrics_text[7], fontsize=11)
        ax_metrics.text(0.0, 0.05, metrics_text[8], fontsize=11)
    
    # Row 3: Technical Indicators
    ax_indicators = plt.subplot(grid[2, :])
    
    # Plot RSI
    ax_indicators.plot(df_original.index, df_original['RSI'], label='RSI')
    ax_indicators.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax_indicators.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    
    # Add MACD
    ax_indicators.plot(df_original.index, df_original['MACD'], label='MACD')
    ax_indicators.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    ax_indicators.set_title('Technical Indicators')
    ax_indicators.legend(loc='best')
    ax_indicators.grid(True)
    
    # Row 4, Col 1: Trading Signal Distribution
    ax_signals = plt.subplot(grid[3, 0])
    signal_counts = signals['recommendation'].value_counts()
    colors = ['green' if x == 'Buy' else 'red' if x == 'Sell' else 'gray' for x in signal_counts.index]
    ax_signals.bar(signal_counts.index, signal_counts.values, color=colors)
    ax_signals.set_title('Trading Signal Distribution')
    ax_signals.set_ylabel('Count')
    ax_signals.grid(axis='y')
    
    # Row 4, Col 2: Risk Assessment
    ax_risk = plt.subplot(grid[3, 1])
    ax_risk.axis('off')
    
    # Risk assessment text
    risk_text = [
        f"Volatility Risk: {analysis_results['risk_assessment']['volatility_risk']}",
        f"Predicted Change: {analysis_results['risk_assessment']['predicted_change']:.2%}",
        f"Risk-Adjusted Return: {analysis_results['risk_assessment']['risk_adjusted_return']:.2f}",
        f"Overall Risk Level: {analysis_results['risk_assessment']['overall_risk']}"
    ]
    
    ax_risk.text(0.5, 0.95, "RISK ASSESSMENT", ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Color-code risk levels
    vol_color = 'green' if analysis_results['risk_assessment']['volatility_risk'] == 'Low' else 'orange' if analysis_results['risk_assessment']['volatility_risk'] == 'Medium' else 'red'
    change_color = 'green' if analysis_results['risk_assessment']['predicted_change'] > 0 else 'red'
    overall_color = 'green' if analysis_results['risk_assessment']['overall_risk'] == 'Low' else 'orange' if analysis_results['risk_assessment']['overall_risk'] == 'Medium' else 'red'
    
    ax_risk.text(0.0, 0.75, risk_text[0], fontsize=11, color=vol_color)
    ax_risk.text(0.0, 0.55, risk_text[1], fontsize=11, color=change_color)
    ax_risk.text(0.0, 0.35, risk_text[2], fontsize=11)
    ax_risk.text(0.0, 0.15, risk_text[3], fontsize=11, color=overall_color, fontweight='bold')
    
    # Row 4, Col 3: Sector Comparison
    ax_sector = plt.subplot(grid[3, 2])
    ax_sector.axis('off')
    
    sector_text = [
        f"Sector: {analysis_results['sector_analysis']['sector']}",
        f"Stock Return: {analysis_results['sector_analysis']['stock_return']:.2%}",
        f"Sector Return: {analysis_results['sector_analysis']['sector_return']:.2%}",
        f"Alpha: {analysis_results['sector_analysis']['alpha']:.2%}",
        f"Beta: {analysis_results['sector_analysis']['beta']:.2f}"
    ]
    
    ax_sector.text(0.5, 0.95, "SECTOR COMPARISON", ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Color-code sector comparison
    stock_color = 'green' if analysis_results['sector_analysis']['stock_return'] > 0 else 'red'
    sector_color = 'green' if analysis_results['sector_analysis']['sector_return'] > 0 else 'red'
    alpha_color = 'green' if analysis_results['sector_analysis']['alpha'] > 0 else 'red'
    
    ax_sector.text(0.0, 0.8, sector_text[0], fontsize=11)
    ax_sector.text(0.0, 0.65, sector_text[1], fontsize=11, color=stock_color)
    ax_sector.text(0.0, 0.5, sector_text[2], fontsize=11, color=sector_color)
    ax_sector.text(0.0, 0.35, sector_text[3], fontsize=11, color=alpha_color, fontweight='bold')
    ax_sector.text(0.0, 0.2, sector_text[4], fontsize=11)
    
    # Add timestamp and disclaimer
    plt.figtext(0.5, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Disclaimer: For informational purposes only. Past performance is not indicative of future results.", 
                ha="center", fontsize=10, style='italic')
    
    # Save the dashboard
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    dashboard_path = os.path.join(output_dir, f'{symbol}_comprehensive_dashboard.png')
    plt.savefig(dashboard_path, dpi=150)
    plt.close()
    
    print(f"Comprehensive dashboard saved to {dashboard_path}")
    
    return dashboard_path

def main():
    """Main function to run the stock analysis from command line."""
    parser = argparse.ArgumentParser(description='Analyze stock data and make predictions')
    parser.add_argument('symbol', type=str, help='Stock symbol to analyze (e.g., AAPL)')
    parser.add_argument('--output', type=str, help='Directory to save results', default='results')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed report')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting of the model trading strategy')
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital for backtesting (default: $10,000)')
    parser.add_argument('--full-dashboard', action='store_true', help='Generate a comprehensive dashboard with analysis and backtest results')
    
    args = parser.parse_args()
    
    output_dir = None if args.no_plots else args.output
    
    try:
        # Run stock analysis
        analysis_results = analyze_stock(args.symbol, output_dir, args.start_date, args.end_date)
        
        # Variables for comprehensive dashboard
        df_original = None
        signals = None
        future_predictions = None
        backtest_metrics = None
        
        if analysis_results:
            # Extract data for dashboard
            if output_dir:
                # Try to load original dataframe and signals from saved files
                df_file = os.path.join(output_dir, f'{args.symbol}_data.csv')
                if os.path.exists(df_file):
                    df_original = pd.read_csv(df_file, index_col=0, parse_dates=True)
                
                signals_file = os.path.join(output_dir, f'{args.symbol}_signals.csv')
                if os.path.exists(signals_file):
                    signals = pd.read_csv(signals_file, index_col=0, parse_dates=True)
                
                future_predictions = analysis_results['future_predictions']['prices']
            
            # Save results to file
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                results_file = os.path.join(args.output, f'{args.symbol}_analysis.txt')
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    # Report header
                    f.write(f"=================================================\n")
                    f.write(f"  STOCK ANALYSIS REPORT FOR {analysis_results['company_name'].upper()} ({args.symbol})\n")
                    f.write(f"=================================================\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Current stock information
                    f.write(f"CURRENT STOCK INFORMATION\n")
                    f.write(f"-------------------------\n")
                    f.write(f"Current Price: ${analysis_results['current_price']:.2f}\n")
                    f.write(f"Volatility: {analysis_results['model_params']['volatility']:.4f}\n")
                    
                    if 'stock_info' in analysis_results and analysis_results['stock_info'] is not None:
                        stock_info = analysis_results['stock_info']
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
                    f.write(f"---------------------\n")
                    
                    recommendation = analysis_results['trading_signals']['recommendation']
                    signal_strength = analysis_results['trading_signals']['current_signal']
                    
                    # Format recommendation with emphasis
                    if recommendation == "Buy":
                        rec_formatted = "BUY"
                    elif recommendation == "Sell":
                        rec_formatted = "SELL"
                    else:
                        rec_formatted = "HOLD"
                        
                    f.write(f"Current Recommendation: {rec_formatted}\n")
                    f.write(f"Signal Strength: {signal_strength:.2f}\n")
                    f.write(f"RSI Signal: {'Bullish' if analysis_results['trading_signals']['rsi_signal'] > 0 else 'Bearish' if analysis_results['trading_signals']['rsi_signal'] < 0 else 'Neutral'}\n")
                    f.write(f"Moving Average Signal: {'Bullish' if analysis_results['trading_signals']['ma_signal'] > 0 else 'Bearish' if analysis_results['trading_signals']['ma_signal'] < 0 else 'Neutral'}\n")
                    f.write(f"MACD Signal: {'Bullish' if analysis_results['trading_signals']['macd_signal'] > 0 else 'Bearish' if analysis_results['trading_signals']['macd_signal'] < 0 else 'Neutral'}\n\n")
                    
                    # Risk assessment
                    f.write(f"RISK ASSESSMENT\n")
                    f.write(f"--------------\n")
                    f.write(f"Volatility Risk: {analysis_results['risk_assessment']['volatility_risk']}\n")
                    f.write(f"Predicted Change: {analysis_results['risk_assessment']['predicted_change']:.2%}\n")
                    f.write(f"Downside Risk: {analysis_results['risk_assessment']['downside_risk']:.2%}\n")
                    f.write(f"Risk-Adjusted Return: {analysis_results['risk_assessment']['risk_adjusted_return']:.2f}\n")
                    f.write(f"Overall Risk Level: {analysis_results['risk_assessment']['overall_risk']}\n\n")
                    
                    # Sector comparison
                    f.write(f"SECTOR COMPARISON\n")
                    f.write(f"-----------------\n")
                    f.write(f"Sector: {analysis_results['sector_analysis']['sector']}\n")
                    f.write(f"Sector ETF: {analysis_results['sector_analysis']['sector_etf']}\n")
                    f.write(f"Stock Return: {analysis_results['sector_analysis']['stock_return']:.2%}\n")
                    f.write(f"Sector Return: {analysis_results['sector_analysis']['sector_return']:.2%}\n")
                    f.write(f"Relative Performance: {analysis_results['sector_analysis']['relative_performance']:.2%}\n")
                    f.write(f"Alpha: {analysis_results['sector_analysis']['alpha']:.2%}\n")
                    f.write(f"Beta: {analysis_results['sector_analysis']['beta']:.2f}\n")
                    if 'stock_sharpe' in analysis_results['sector_analysis']:
                        f.write(f"Stock Sharpe Ratio: {analysis_results['sector_analysis']['stock_sharpe']:.2f}\n")
                        f.write(f"Sector Sharpe Ratio: {analysis_results['sector_analysis']['sector_sharpe']:.2f}\n")
                    f.write("\n")
                    
                    # Model performance metrics
                    f.write(f"MODEL PERFORMANCE METRICS\n")
                    f.write(f"------------------------\n")
                    f.write(f"Mean Absolute Error: {analysis_results['metrics']['mae']:.4f}\n")
                    f.write(f"Mean Squared Error: {analysis_results['metrics']['mse']:.4f}\n")
                    f.write(f"Root Mean Squared Error: {analysis_results['metrics']['rmse']:.4f}\n\n")
                    
                    # Model parameters
                    f.write(f"MODEL PARAMETERS\n")
                    f.write(f"---------------\n")
                    f.write(f"Sequence Length: {analysis_results['model_params']['sequence_length']}\n")
                    f.write(f"Prediction Horizon: {analysis_results['model_params']['prediction_days']} days\n\n")
                    
                    # Future price predictions
                    f.write(f"FUTURE PRICE PREDICTIONS\n")
                    f.write(f"-----------------------\n")
                    f.write("Date            | Price    | 95% Confidence Interval\n")
                    f.write("-------------------------------------------------\n")
                    
                    for date, price, lower, upper in zip(
                        analysis_results['future_predictions']['dates'],
                        analysis_results['future_predictions']['prices'],
                        analysis_results['future_predictions']['lower_bound'],
                        analysis_results['future_predictions']['upper_bound']):
                        f.write(f"{date.strftime('%Y-%m-%d')} | ${price:.2f}  | (${lower:.2f} - ${upper:.2f})\n")
                    f.write("\n")
                    
                    # Analysis conclusion
                    f.write(f"ANALYSIS CONCLUSION\n")
                    f.write(f"------------------\n")
                    
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
                    risk_level = analysis_results['risk_assessment']['overall_risk']
                    if risk_level == "Low":
                        conclusion.append(f"The stock shows low risk characteristics with a volatility of {analysis_results['model_params']['volatility']:.4f} and a positive risk-adjusted return of {analysis_results['risk_assessment']['risk_adjusted_return']:.2f}.")
                    elif risk_level == "Medium":
                        conclusion.append(f"The stock shows moderate risk with a volatility of {analysis_results['model_params']['volatility']:.4f} and a risk-adjusted return of {analysis_results['risk_assessment']['risk_adjusted_return']:.2f}.")
                    else:
                        conclusion.append(f"The stock shows high risk characteristics with a volatility of {analysis_results['model_params']['volatility']:.4f} and a risk-adjusted return of {analysis_results['risk_assessment']['risk_adjusted_return']:.2f}.")
                    
                    # Sector comparison conclusion
                    rel_perf = analysis_results['sector_analysis']['relative_performance']
                    if rel_perf > 0:
                        conclusion.append(f"{args.symbol} has outperformed its sector by {rel_perf:.2%} with a beta of {analysis_results['sector_analysis']['beta']:.2f} and alpha of {analysis_results['sector_analysis']['alpha']:.2%}.")
                    else:
                        conclusion.append(f"{args.symbol} has underperformed its sector by {-rel_perf:.2%} with a beta of {analysis_results['sector_analysis']['beta']:.2f} and alpha of {analysis_results['sector_analysis']['alpha']:.2%}.")
                    
                    # Future prediction conclusion
                    future_prices = analysis_results['future_predictions']['prices']
                    current_price = analysis_results['current_price']
                    price_change = (future_prices[-1] - current_price) / current_price
                    
                    if price_change > 0:
                        conclusion.append(f"The model predicts a {price_change:.2%} increase in price over the next {analysis_results['model_params']['prediction_days']} days, with an upward trend.")
                    else:
                        conclusion.append(f"The model predicts a {-price_change:.2%} decrease in price over the next {analysis_results['model_params']['prediction_days']} days, with a downward trend.")
                    
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
                    f.write("----------\n")
                    f.write("This analysis is for informational purposes only and not a recommendation to buy or sell any securities.\n")
                    f.write("Past performance does not guarantee future results. All investments involve risk.\n")
                    f.write("Always conduct your own research and consider seeking advice from a financial professional.\n")
                
                print(f"\nAnalysis results saved to {results_file}")
                print(f"Visualizations saved to {args.output} directory")
        
        # Run backtesting if requested
        if args.backtest:
            print(f"\nRunning backtesting for {args.symbol}...")
            backtest_metrics = backtest_strategy(
                args.symbol, 
                args.start_date, 
                args.end_date, 
                output_dir, 
                args.initial_capital
            )
            
            if backtest_metrics:
                print("\nBacktest Summary:")
                print(f"Total Return: {backtest_metrics['total_return']:.2%}")
                print(f"Buy & Hold Return: {backtest_metrics['benchmark_return']:.2%}")
                print(f"Excess Return: {backtest_metrics['excess_return']:.2%}")
                print(f"Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}")
                print(f"Win Rate: {backtest_metrics['win_rate']:.2%}")
                print(f"Maximum Drawdown: {backtest_metrics['max_drawdown']:.2%}")
                print(f"\nFull backtest results saved to {args.output} directory")
        
        # Create comprehensive dashboard if both analysis and backtest were performed
        if (args.full_dashboard or args.backtest) and analysis_results and backtest_metrics and output_dir:
            if df_original is not None and signals is not None and future_predictions is not None:
                print("\nGenerating comprehensive dashboard...")
                create_comprehensive_dashboard(
                    args.symbol,
                    analysis_results,
                    backtest_metrics,
                    df_original,
                    signals,
                    future_predictions,
                    output_dir
                )
            else:
                print("\nCannot generate comprehensive dashboard: missing required data files.")
                print("Try re-running the analysis without --no-plots option to generate all required files.")
                
    except Exception as e:
        print(f"Error analyzing stock: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 