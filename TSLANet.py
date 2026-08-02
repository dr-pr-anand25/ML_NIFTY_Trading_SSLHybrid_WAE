# ============================================
# TSLANet: Machine Learning Strategy for NIFTY 50
# ============================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TSLANET: NIFTY 50 TRADING STRATEGY")
print("="*60)

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("\n[1] Loading NIFTY 5-minute data...")

def load_nifty_data():
    """Download NIFTY 5-minute data from Yahoo Finance"""
    df = yf.download('^NSEI', period='60d', interval='5m', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = ['close', 'high', 'low', 'open', 'volume']
    df = df.between_time('03:45', '10:00')
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    return df

df = load_nifty_data()

# ============================================
# STEP 2: CALCULATE SSL HYBRID
# ============================================
print("\n[2] Calculating SSL Hybrid indicator...")

def calculate_ssl_hybrid(df, len_period=20, len2=5, len3=10, multy=0.2, atr_len=14):
    """SSL Hybrid Indicator"""
    df = df.copy()
    
    def hma(series, period):
        half_period = max(period // 2, 1)
        sqrt_period = max(int(np.sqrt(period)), 1)
        wma1 = series.rolling(half_period).mean() * 2
        wma2 = series.rolling(period).mean()
        return (wma1 - wma2).rolling(sqrt_period).mean()
    
    # Baseline
    df['BBMC'] = hma(df['close'], len_period)
    
    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift()))
    )
    df['atr'] = df['tr'].rolling(atr_len).mean()
    df['upper_channel'] = df['BBMC'] + multy * df['atr']
    df['lower_channel'] = df['BBMC'] - multy * df['atr']
    
    # SSL1
    ema_high = hma(df['high'], len_period)
    ema_low = hma(df['low'], len_period)
    
    df['Hlv1'] = 0
    df.loc[df['close'] > ema_high, 'Hlv1'] = 1
    df.loc[df['close'] < ema_low, 'Hlv1'] = -1
    df['Hlv1'] = df['Hlv1'].replace(0, np.nan).ffill().fillna(0)
    df['ssl1'] = np.where(df['Hlv1'] < 0, ema_high, ema_low)
    
    # SSL2
    df['Hlv2'] = 0
    df.loc[df['close'] > hma(df['high'], len2), 'Hlv2'] = 1
    df.loc[df['close'] < hma(df['low'], len2), 'Hlv2'] = -1
    df['Hlv2'] = df['Hlv2'].replace(0, np.nan).ffill().fillna(0)
    df['ssl2'] = np.where(df['Hlv2'] < 0, hma(df['high'], len2), hma(df['low'], len2))
    
    # SSL3 (Exit)
    df['Hlv3'] = 0
    df.loc[df['close'] > hma(df['high'], len3), 'Hlv3'] = 1
    df.loc[df['close'] < hma(df['low'], len3), 'Hlv3'] = -1
    df['Hlv3'] = df['Hlv3'].replace(0, np.nan).ffill().fillna(0)
    df['ssl_exit'] = np.where(df['Hlv3'] < 0, hma(df['high'], len3), hma(df['low'], len3))
    
    # Entry signals
    df['exit_long'] = ((df['close'].shift(1) <= df['ssl_exit'].shift(1)) & (df['close'] > df['ssl_exit']))
    df['exit_short'] = ((df['close'].shift(1) >= df['ssl_exit'].shift(1)) & (df['close'] < df['ssl_exit']))
    
    return df

df = calculate_ssl_hybrid(df)
print(f"✅ SSL Hybrid added. Shape: {df.shape}")

# ============================================
# STEP 3: CALCULATE WAE
# ============================================
print("\n[3] Calculating Waddah Attar Explosion V3...")

def calculate_wae(df, macd_fast=12, macd_slow=26, macd_signal=9,
                  bb_period=20, bb_mult=2.0, dead_zone_mult=3.7):
    """Waddah Attar Explosion V3"""
    df = df.copy()
    
    exp1 = df['close'].ewm(span=macd_fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd_line'] = exp1 - exp2
    df['signal_line'] = df['macd_line'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_histogram'] = (df['macd_line'] - df['signal_line']) * 150
    
    df['bb_middle'] = df['close'].rolling(bb_period).mean()
    bb_std = df['close'].rolling(bb_period).std()
    df['bb_width'] = df['bb_middle'] + bb_mult * bb_std - (df['bb_middle'] - bb_mult * bb_std)
    
    df['true_range'] = df['high'] - df['low']
    df['dead_zone_upper'] = df['true_range'].rolling(bb_period).mean() * dead_zone_mult
    df['dead_zone_lower'] = -df['dead_zone_upper']
    
    df['bar_inside_dead'] = (df['macd_histogram'] <= df['dead_zone_upper']) & (df['macd_histogram'] >= df['dead_zone_lower'])
    df['dead_zone_reentry'] = (df['bar_inside_dead'].shift(1) == False) & df['bar_inside_dead']
    df['trend_direction'] = np.where(df['macd_histogram'] > 0, 1, -1)
    df['trend_strength'] = abs(df['macd_histogram']) / (df['dead_zone_upper'] + 1e-8)
    
    return df

df = calculate_wae(df)
print(f"✅ WAE added. Shape: {df.shape}")

# ============================================
# STEP 4: FEATURE ENGINEERING
# ============================================
print("\n[4] Creating features and target...")

def create_features(df, lookahead=6, threshold=0.005):
    """Create features and target"""
    df = df.copy()
    
    for period in [1, 3, 5, 10, 20]:
        df[f'return_{period}'] = df['close'].pct_change(period)
        df[f'volatility_{period}'] = df['return_1'].rolling(period).std()
    
    ssl_cols = ['BBMC', 'upper_channel', 'lower_channel', 'atr', 'ssl1', 'ssl2', 
                'ssl_exit', 'exit_long', 'exit_short']
    wae_cols = ['macd_histogram', 'macd_line', 'signal_line', 'bb_width',
                'dead_zone_upper', 'dead_zone_lower', 'bar_inside_dead', 
                'dead_zone_reentry', 'trend_strength', 'trend_direction']
    
    available_ssl = [col for col in ssl_cols if col in df.columns]
    available_wae = [col for col in wae_cols if col in df.columns]
    
    future_close = df['close'].shift(-lookahead)
    df['target'] = ((future_close / df['close']) - 1) > threshold
    df['target'] = df['target'].astype(int)
    
    df = df[df['target'].notna()]
    feature_cols = available_ssl + available_wae + ['target']
    df = df[feature_cols].dropna()
    
    print(f"SSL features: {len(available_ssl)}")
    print(f"WAE features: {len(available_wae)}")
    print(f"Total features: {len(available_ssl) + len(available_wae)}")
    print(f"Final rows: {len(df)}")
    
    return df, available_ssl, available_wae

df, ssl_features, wae_features = create_features(df)

# ============================================
# STEP 5: TRAIN MODEL
# ============================================
print("\n[5] Training TSLANet model...")

all_features = ssl_features + wae_features
X = df[all_features].values
y = df['target'].values

total_rows = len(df)
train_size = int(total_rows * 0.7)
val_size = int(total_rows * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
val_acc = model.score(X_val_scaled, y_val)
test_acc = model.score(X_test_scaled, y_test)

print(f"\n📊 Model Performance:")
print(f"  Training Accuracy: {train_acc:.3f}")
print(f"  Validation Accuracy: {val_acc:.3f}")
print(f"  Test Accuracy: {test_acc:.3f}")

# ============================================
# STEP 6: TRADING STRATEGY
# ============================================
print("\n[6] Running TSLANet Direct strategy...")

def trade_tslanet_direct(df, model, scaler, all_features, threshold=0.5):
    """Trade directly on TSLANet predictions"""
    trades = []
    
    for i in range(100, len(df) - 10):
        try:
            features = df.iloc[i][all_features].values.reshape(1, -1)
            features_scaled = scaler.transform(features)
            prob = model.predict_proba(features_scaled)[0, 1]
            
            if prob > threshold:
                entry_price = df['close'].iloc[i]
                exit_price = df['close'].iloc[i + 6]
                trade_return = (exit_price - entry_price) / entry_price
                
                trades.append({
                    'prob': prob,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'win': trade_return > 0
                })
        except:
            continue
    
    return trades

trades = trade_tslanet_direct(df, model, scaler, all_features, threshold=0.5)

# Calculate metrics
if trades:
    returns = [t['return'] for t in trades]
    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    
    win_rate = len(wins) / len(trades)
    avg_return = np.mean(returns)
    avg_win = np.mean([t['return'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['return'] for t in losses]) if losses else 0
    
    total_wins = sum([t['return'] for t in wins]) if wins else 0
    total_losses = abs(sum([t['return'] for t in losses])) if losses else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    total_return = np.prod(1 + np.array(returns)) - 1
    
    print(f"\n📊 TSLANet Direct (0.5 Threshold):")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total Return: {total_return:.2%}")

# ============================================
# STEP 7: SAVE RESULTS
# ============================================
print("\n[7] Saving results...")

import pickle
with open('tslanet_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Results saved!")
print("\n" + "="*60)
print("TSLANET PIPELINE COMPLETE!")
print("="*60)