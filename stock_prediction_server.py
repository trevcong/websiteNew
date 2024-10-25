from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import ta
import datetime
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for model persistence
MODEL_PATH = 'stock_prediction_model.joblib'
scaler = StandardScaler()


class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.performance_metrics = {
            'accuracy': None,
            'risk_score': None
        }

    def fit_scaler(self, X):
        """Fit the scaler with data"""
        self.scaler.fit(X)
        self.is_fitted = True
        joblib.dump(self.scaler, 'scaler.joblib')

    def transform_data(self, X):
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise Exception("Scaler not fitted yet")
        return self.scaler.transform(X)

    def calculate_metrics(self, X, y):
        """Calculate model performance metrics"""
        try:
            X_scaled = self.transform_data(X)
            predictions = self.model.predict(X_scaled)
            
            # Calculate accuracy
            accuracy = (predictions == y).mean()
            
            # Calculate risk score based on prediction consistency
            recent_predictions = predictions[-20:]
            prediction_changes = np.diff(recent_predictions)
            consistency = 1 - (np.abs(prediction_changes).sum() / len(prediction_changes))
            
            # Calculate volatility
            volatility = np.std(self.model.predict_proba(X_scaled)[:, 1])
            
            # Combined risk score (lower is better)
            risk_score = (1 - accuracy) + (1 - consistency) + volatility
            
            self.performance_metrics = {
                'accuracy': float(accuracy),
                'risk_score': float(risk_score),
                'consistency': float(consistency),
                'volatility': float(volatility)
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            self.performance_metrics = {
                'accuracy': None,
                'risk_score': None
            }

    def save_model(self):
        """Save the model and scaler"""
        if self.model is not None and self.is_fitted:
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, 'scaler.joblib')
            print("Model and scaler saved successfully")
            return True
        return False

    def load_model(self):
        """Load saved model and scaler"""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists('scaler.joblib'):
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load('scaler.joblib')
                self.is_fitted = True
                print("Model and scaler loaded successfully")
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def train_model(self, X, y):
        """Train and save the prediction model"""
        try:
            if not self.is_fitted:
                self.fit_scaler(X)
            
            X_scaled = self.transform_data(X)
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            # Calculate metrics after training
            self.calculate_metrics(X, y)
            
            # Save the model and scaler
            self.save_model()
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False

    def predict(self, X, y=None):
        """Make prediction using model"""
        if not self.is_fitted or self.model is None:
            raise Exception("Model or scaler not ready")
            
        X_scaled = self.transform_data(X)
        prediction = self.model.predict(X_scaled)
        probability = self.model.predict_proba(X_scaled)
        
        # Update metrics if y is provided
        if y is not None:
            self.calculate_metrics(X, y)
            
        return prediction, probability

    def get_model_status(self):
        """Get current status of the model"""
        return {
            'is_fitted': self.is_fitted,
            'has_model': self.model is not None,
            'metrics': self.performance_metrics
        }
    
model_manager = ModelManager()    

def calculate_technical_indicators(df):
    """Calculate technical indicators for prediction"""
    try:
        # Trend Indicators
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

        # Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['Close'])
        df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        df['willr'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])

        # Volatility Indicators
        df['bb_high'] = ta.volatility.bollinger_hband(df['Close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['Close'])
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Volume Indicators
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])

        # Additional Features
        df['daily_return'] = df['Close'].pct_change()
        df['volatility'] = df['daily_return'].rolling(window=20).std()
        
        return df
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        raise

def prepare_features(df):
    """Prepare features for the model"""
    features = [
        'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'adx',
        'rsi', 'stoch', 'cci', 'willr', 'bb_high', 'bb_low',
        'atr', 'obv', 'mfi', 'volatility'
    ]
    
    X = df[features].fillna(0)
    y = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return X[:-1], y[:-1]

def train_model(X, y):
    """Train and save the prediction model"""
    try:
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        joblib.dump(model, MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

def get_detailed_analysis(df, ticker):
    """Get detailed analysis of why the prediction was made"""
    try:
        current_price = df['Close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        volume_trend = df['obv'].iloc[-1] > df['obv'].iloc[-5]
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        signals = {
            "trend_signals": {
                "price_vs_sma20": "bearish" if current_price < sma_20 else "bullish",
                "price_vs_sma50": "bearish" if current_price < sma_50 else "bullish",
                "sma_crossover": "bearish" if sma_20 < sma_50 else "bullish",
                "macd_signal": "bearish" if macd < 0 else "bullish",
            },
            "momentum_signals": {
                "rsi_status": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral",
                "volume_trend": "up" if volume_trend else "down",
            },
            "fundamental_signals": {
                "pe_ratio": info.get('forwardPE', None),
                "market_cap": info.get('marketCap', None),
                "beta": info.get('beta', None)
            }
        }
        
        confidence_factors = []
        
        if current_price < sma_20 and current_price < sma_50:
            confidence_factors.append("Price below both SMAs")
        if sma_20 < sma_50:
            confidence_factors.append("Bearish SMA crossover")
        if macd < 0:
            confidence_factors.append("Negative MACD")
        if rsi > 70:
            confidence_factors.append("Overbought RSI")
        if not volume_trend:
            confidence_factors.append("Declining volume")
        
        volatility = df['Close'].pct_change().std() * np.sqrt(252)
        if volatility > 0.4:
            confidence_factors.append("High volatility")
        
        return {
            "signals": signals,
            "confidence_factors": confidence_factors,
            "key_levels": {
                "support": float(min(df['Low'][-20:])),
                "resistance": float(max(df['High'][-20:])),
                "current_price": float(current_price)
            }
        }
    except Exception as e:
        print(f"Error in detailed analysis: {str(e)}")
        raise

@app.route('/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return jsonify({"error": "Invalid ticker symbol"}), 404
        
        prices = [{
            'date': date.strftime('%Y-%m-%d'),
            'price': price
        } for date, price in zip(hist.index, hist['Close'])]
        
        current_price = hist['Close'][-1]
        previous_price = hist['Close'][-2]
        daily_change = ((current_price - previous_price) / previous_price) * 100
        
        info = stock.info
        stats = {
            'current_price': current_price,
            'daily_change': daily_change,
            'volume': int(hist['Volume'][-1]),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('forwardPE', None),
            'dividend_yield': info.get('dividendYield', None),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', None)
        }
        
        return jsonify({
            "prices": prices,
            "stats": stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<ticker>', methods=['GET'])
def predict_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return jsonify({"error": "Invalid ticker symbol"}), 404
            
        df = calculate_technical_indicators(hist)
        analysis = get_detailed_analysis(df, ticker)
        
        X, y = prepare_features(df)
        
        # Ensure model is ready
        if not model_manager.is_fitted:
            print("Training new model...")
            model_manager.train_model(X, y)
        
        # Get prediction and update metrics
        prediction, probability = model_manager.predict(X, y)
        prediction = prediction[-1]
        probability = probability[-1]
        
        validation_score = len(analysis['confidence_factors'])
        adjusted_probability = (probability[prediction] + (validation_score / 10)) / 2
        
        return jsonify({
            "prediction": "up" if prediction == 1 else "down",
            "probability": float(adjusted_probability * 100),
            "raw_probability": float(probability[prediction] * 100),
            "confidence_score": validation_score,
            "analysis": analysis,
            "signals": {
                "technical": analysis['signals']['trend_signals'],
                "momentum": analysis['signals']['momentum_signals'],
                "fundamental": analysis['signals']['fundamental_signals']
            },
            "supporting_factors": analysis['confidence_factors'],
            "key_levels": analysis['key_levels'],
            "metrics": model_manager.performance_metrics,
            "recommendation": {
                "action": "strong_sell" if adjusted_probability > 0.7 else 
                         "sell" if adjusted_probability > 0.6 else
                         "hold" if adjusted_probability > 0.4 else
                         "buy" if adjusted_probability > 0.3 else
                         "strong_buy",
                "stop_loss": float(analysis['key_levels']['support']),
                "take_profit": float(analysis['key_levels']['resistance'])
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Stock Prediction Server...")
    print("Loading dependencies and initializing models...")
    
    try:
        # Initialize model manager
        model_manager = ModelManager()
        
        # Initialize with some training data
        print("Initializing model with training data...")
        test_ticker = "AAPL"
        stock = yf.Ticker(test_ticker)
        hist = stock.history(period="1y")
        
        if not hist.empty:
            df = calculate_technical_indicators(hist)
            X, y = prepare_features(df)
            
            # Try to load existing model
            if not model_manager.load_model():
                print("No existing model found. Training new model...")
                if model_manager.train_model(X, y):
                    print("Model trained successfully!")
                else:
                    raise Exception("Failed to train model")
            
            # Verify model status
            status = model_manager.get_model_status()
            print(f"Model Status: {status}")
            
            print("\nServer ready! Starting on port 5002...")
            app.run(host='0.0.0.0', port=5002, debug=True)
        else:
            raise Exception("Failed to get initial training data")
            
    except Exception as e:
        print(f"Error starting server: {str(e)}")