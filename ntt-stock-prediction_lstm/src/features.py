#NTT株価データの特徴量エンジニアリング


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import ks_2samp
import talib
import warnings

warnings.filterwarnings('ignore')

# グラフ設定
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


class StockFeatureEngineer:
    """株価データの特徴量エンジニアリングクラス"""
    
    def __init__(self, data_path: str):
        """
        初期化
        
        Args:
            data_path (str): CSVファイルのパス
        """
        self.data_path = data_path
        self.df = None
        self.df_features = None
        
    def load_data(self) -> pd.DataFrame:
        """データの読み込み"""
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        return self.df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間関連の特徴量を生成"""
        # 基本的な時間特徴量
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['Week_of_year'] = df['Date'].dt.isocalendar().week
        df['Is_month_start'] = df['Date'].dt.is_month_start.astype(int)
        df['Is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        
        # 周期性を考慮したsin/cos変換
        df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
        df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
        df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格関連の基本特徴量を生成（未来リーク防止）"""
        # 基本リターン系（過去情報のみ使用）
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # ギャップ（前日終値と当日始値の差）
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # 同日価格情報は未来リークのため除外
        # Range, Body, Shadow系は使用しない
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """TALibを使用したテクニカル指標を生成"""
        high = df['High'].astype(np.float64).values
        low = df['Low'].astype(np.float64).values
        close = df['Close'].astype(np.float64).values
        open_price = df['Open'].astype(np.float64).values
        volume = df['Volume'].astype(np.float64).values
        
        # 移動平均・EMA（ラグ版のみ使用で未来リーク防止）
        for period in [5, 10, 20, 50, 100, 200]:
            # 移動平均を一日シフトして未来リークを防ぐ
            df[f'SMA_{period}_Lag1'] = pd.Series(talib.SMA(close, timeperiod=period)).shift(1)
            if period <= 50:
                df[f'EMA_{period}_Lag1'] = pd.Series(talib.EMA(close, timeperiod=period)).shift(1)
        
        # RSI
        for period in [9, 14, 21]:
            df[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # ストキャスティクス
        slowk, slowd = talib.STOCH(high, low, close, 14, 3, 3)
        df['STOCH_K'] = slowk
        df['STOCH_D'] = slowd
        
        # その他のオシレーター
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        
        # モメンタム系
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = talib.ROC(close, timeperiod=period)
            df[f'MOM_{period}'] = talib.MOM(close, timeperiod=period)
        
        # ボラティリティ系（ラグ版で未来リーク防止）
        df['ATR_14_Lag1'] = pd.Series(talib.ATR(high, low, close, timeperiod=14)).shift(1)
        df['NATR_Lag1'] = pd.Series(talib.NATR(high, low, close, timeperiod=14)).shift(1)
        
        # ボリンジャーバンド（ラグ版で未来リーク防止）
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
        df['BB_Upper_Lag1'] = pd.Series(bb_upper).shift(1)
        df['BB_Middle_Lag1'] = pd.Series(bb_middle).shift(1)
        df['BB_Lower_Lag1'] = pd.Series(bb_lower).shift(1)
        df['BB_Width_Lag1'] = ((pd.Series(bb_upper).shift(1) - pd.Series(bb_lower).shift(1)) / 
                               pd.Series(bb_middle).shift(1))
        
        # 出来高系
        df['AD'] = talib.AD(high, low, close, volume)
        df['ADOSC'] = talib.ADOSC(high, low, close, volume, 3, 10)
        df['OBV'] = talib.OBV(close, volume)
        df['OBV_Change'] = df['OBV'].pct_change()
        
        # ローソク足パターン
        df['CDLDOJI'] = talib.CDLDOJI(open_price, high, low, close)
        df['CDLHAMMER'] = talib.CDLHAMMER(open_price, high, low, close)
        df['CDLENGULFING'] = talib.CDLENGULFING(open_price, high, low, close)
        
        # 統計系
        for period in [5, 10, 20]:
            df[f'STDDEV_{period}'] = talib.STDDEV(close, timeperiod=period)
        
        df['VAR_20'] = talib.VAR(close, timeperiod=20)
        df['LINEARREG_14'] = talib.LINEARREG(close, timeperiod=14)
        df['LINEARREG_SLOPE_14'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
        df['TSF_14'] = talib.TSF(close, timeperiod=14)
        
        # 価格系
        df['TYPPRICE'] = talib.TYPPRICE(high, low, close)
        df['WCLPRICE'] = talib.WCLPRICE(high, low, close)
        df['MEDPRICE'] = talib.MEDPRICE(high, low)
        
        # アルーン系
        aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
        df['AROON_DOWN'] = aroon_down
        df['AROON_UP'] = aroon_up
        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
        
        # ADX系
        df['DX'] = talib.DX(high, low, close, timeperiod=14)
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ラグ特徴量を生成"""
        close = df['Close'].astype(np.float64).values
        
        # 過去の値（Daily_Returnを使用）
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            
            # ラグ版テクニカル指標（さらにラグを加えて未来リーク防止）
            if lag <= 5:
                df[f'SMA_5_Lag_{lag+1}'] = pd.Series(talib.SMA(close, timeperiod=5)).shift(lag+1)
                df[f'EMA_5_Lag_{lag+1}'] = pd.Series(talib.EMA(close, timeperiod=5)).shift(lag+1)
            if lag <= 3:
                df[f'SMA_20_Lag_{lag+1}'] = pd.Series(talib.SMA(close, timeperiod=20)).shift(lag+1)
                df[f'RSI_14_Lag_{lag+1}'] = pd.Series(talib.RSI(close, timeperiod=14)).shift(lag+1)
        
        # 未来の値（予測用ラベルのみ、特徴量としては使用しない）
        for forward in [1, 3, 5, 10]:
            df[f'Forward_Return_{forward}d'] = df['Close'].pct_change(periods=forward).shift(-forward)
            df[f'Forward_Direction_{forward}d'] = np.where(df[f'Forward_Return_{forward}d'] > 0, 1, 0)
        
        return df
    
    def _create_extended_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """拡張特徴量を生成"""
        close = df['Close'].astype(np.float64).values
        volume = df['Volume'].astype(np.float64).values
        
        # 出来高関連
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Ratio_5'] = df['Volume'] / df['Volume'].rolling(5).mean()
        
        # ゴールデンクロス・デッドクロス（ラグ版）
        sma_50_lag = pd.Series(talib.SMA(close, timeperiod=50)).shift(1)
        sma_200_lag = pd.Series(talib.SMA(close, timeperiod=200)).shift(1)
        df['SMA_Cross_50_200_Lag1'] = (sma_50_lag > sma_200_lag).astype(int)
        
        # MACD差分
        macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD_Diff'] = macd - macd_signal
        
        # 加速度的特徴量（適切なラグで修正）
        return_5d_lag = df['Close'].pct_change(5).shift(6)  # 5日リターンをさらに1日シフト
        return_10d_lag = df['Close'].pct_change(10).shift(11)  # 10日リターンをさらに1日シフト
        df['Acceleration_5_10'] = return_5d_lag - return_10d_lag
        
        # イベントフラグ
        df['Is_Earnings_Season'] = df['Date'].dt.month.isin([3, 6, 9, 12]).astype(int)
        df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
        
        return df
    
    def generate_features(self) -> pd.DataFrame:
        """全ての特徴量を生成"""
        if self.df is None:
            raise ValueError("データが読み込まれていません。load_data()を先に実行してください。")
        
        df_feat = self.df.copy()
        
        # 各種特徴量を順次生成
        df_feat = self._create_time_features(df_feat)
        df_feat = self._create_price_features(df_feat)
        df_feat = self._create_technical_indicators(df_feat)
        df_feat = self._create_lag_features(df_feat)
        df_feat = self._create_extended_features(df_feat)
        
        # 欠損値を除去
        self.df_features = df_feat.dropna()
        
        return self.df_features
    
    def evaluate_feature_importance(self, target_col: str = 'Close', n_splits: int = 5) -> pd.DataFrame:
        """
        時系列対応の特徴量重要度評価（終値をターゲット）
        
        Args:
            target_col (str): 目的変数のカラム名（デフォルト: 'Close'）
            n_splits (int): 時系列分割数
            
        Returns:
            pd.DataFrame: 特徴量重要度のデータフレーム
        """
        if self.df_features is None:
            raise ValueError("特徴量が生成されていません。generate_features()を先に実行してください。")
        
        # 目的変数との相関でデータリーケージを自動検出
        target_series = self.df_features[target_col]
        high_corr_features = []
        
        for col in self.df_features.columns:
            if col != target_col and pd.api.types.is_numeric_dtype(self.df_features[col]):
                corr = abs(target_series.corr(self.df_features[col]))
                if corr > 0.95:  # 95%以上の相関でリーケージと判定
                    high_corr_features.append((col, corr))
        
        leakage_features = [col for col, _ in high_corr_features]
        
        print(f"\n高相関特徴量（リーケージの可能性）:")
        for col, corr in sorted(high_corr_features, key=lambda x: x[1], reverse=True):
            print(f"  {col}: {corr:.3f}")
        
        # 目的変数とターゲット列を除外する列を定義
        exclude_cols = [
            'Date', target_col,
            'Forward_Return_1d', 'Forward_Return_3d', 'Forward_Return_5d', 'Forward_Return_10d',
            'Forward_Direction_1d', 'Forward_Direction_3d', 'Forward_Direction_5d', 'Forward_Direction_10d'
        ] + leakage_features
        
        print(f"\n予測対象: {target_col}")
        print(f"除外された特徴量数: {len(leakage_features)}")
        print(f"使用される特徴量数: {len([col for col in self.df_features.columns if col not in exclude_cols])}")
        
        # 存在する列のみを除外
        exclude_cols = [col for col in exclude_cols if col in self.df_features.columns]
        
        # 相関チェック後の特徴量数を表示
        remaining_features = [col for col in self.df_features.columns if col not in exclude_cols]
        print(f"最終的に使用される特徴量数: {len(remaining_features)}")
        if len(remaining_features) < 10:
            print(f"使用特徴量: {remaining_features}")
        
        y = self.df_features[target_col].values
        X = self.df_features.drop(exclude_cols, axis=1)
        
        # カテゴリカル変数をダミー変数に変換
        categorical_cols = ['Weekday', 'Month', 'Quarter']
        existing_categorical_cols = [col for col in categorical_cols if col in X.columns]
        
        if existing_categorical_cols:
            X = pd.get_dummies(X, columns=existing_categorical_cols, drop_first=True)
        
        # 数値型のみを選択
        X = X.select_dtypes(include=[np.number])
        X_values = X.values
        feature_names = X.columns
        
        # 時系列分割で特徴量重要度を評価
        tscv = TimeSeriesSplit(n_splits=n_splits)
        importances = []
        
        for train_index, test_index in tscv.split(X_values):
            X_train, X_test = X_values[train_index], X_values[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model = RandomForestRegressor(n_estimators=20, random_state=42)
            model.fit(X_train, y_train)
            importances.append(model.feature_importances_)
        
        # 平均重要度を計算
        mean_importance = np.mean(importances, axis=0)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_importance
        }).sort_values('Importance', ascending=False)
        
        return feature_importance
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, top_n: int = 30):
        """特徴量重要度をプロット"""
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
        plt.title(f'Top {top_n} Important Features for Close Price Prediction')
        plt.tight_layout()
        plt.show()
    
    def run_drift_analysis(self, feature_importance: pd.DataFrame, test_size: int = 30) -> pd.DataFrame:
        """
        データドリフト分析を実行
        
        Args:
            feature_importance (pd.DataFrame): 特徴量重要度
            test_size (int): テストデータのサイズ
            
        Returns:
            pd.DataFrame: ドリフト分析結果
        """
        if self.df_features is None:
            raise ValueError("特徴量が生成されていません。")
        
        # データ分割
        df_future = self.df_features.tail(test_size)
        df_train = self.df_features.iloc[:-test_size]
        
        # KS検定実行
        ks_results = []
        common_cols = set(df_train.columns).intersection(set(df_future.columns))
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(df_train[col]):
                stat, p = ks_2samp(df_train[col].dropna(), df_future[col].dropna())
                ks_results.append({
                    'Feature': col,
                    'ks_stat': round(stat, 4),
                    'p_value': round(p, 4),
                    'drift_detected': p < 0.05
                })
        
        ks_result = pd.DataFrame(ks_results).sort_values('ks_stat', ascending=False)
        
        # 重要度とマージ
        combined_result = pd.merge(feature_importance, ks_result, on='Feature', how='inner')
        combined_result = combined_result.sort_values(['drift_detected', 'Importance'], ascending=[False, False])
        
        return combined_result
    
    def save_results(self, output_dir: str):
        """結果を保存"""
        if self.df_features is None:
            raise ValueError("特徴量が生成されていません。")
        
        # 特徴量データを保存
        features_path = f"{output_dir}\\features_talib.csv"
        self.df_features.to_csv(features_path, index=False)
        print(f"特徴量データを保存しました: {features_path}")
        
        return features_path


def main():
    """メイン処理"""
    # データパス
    data_path = r"C:\Users\Take\python\ntt-stock-prediction_test\data\data.csv"
    output_dir = r"C:\Users\Take\python\ntt-stock-prediction_test\data"
    
    # 特徴量エンジニアリング実行
    engineer = StockFeatureEngineer(data_path)
    
    # データ読み込み
    df = engineer.load_data()
    print(f"データ読み込み完了: {df.shape}")
    
    # 特徴量生成
    df_features = engineer.generate_features()
    print(f"生成された特徴量の数: {df_features.shape[1] - 1}")
    print(f"データ行数: {df_features.shape[0]}")
    
    # TALib特徴量の確認
    talib_features = [col for col in df_features.columns if any(x in col for x in 
                     ['RSI', 'MACD', 'STOCH', 'WILLR', 'CCI', 'ROC', 'MOM', 'ATR', 'NATR', 
                      'BB_', 'AD', 'OBV', 'CDL', 'STDDEV', 'VAR', 'LINEARREG', 'TSF',
                      'TYPPRICE', 'WCLPRICE', 'MEDPRICE', 'AROON', 'DX', 'ADX', 'PLUS_DI', 'MINUS_DI'])]
    
    print(f"\nTALibで生成された特徴量数: {len(talib_features)}")
    
    # 特徴量重要度評価（終値をターゲット）
    feature_importance = engineer.evaluate_feature_importance(target_col='Close')
    print("\n上位の特徴量（Top 30）:")
    print(feature_importance.head(30))
    
    # 重要度プロット
    engineer.plot_feature_importance(feature_importance)
    
    # ドリフト分析
    combined_result = engineer.run_drift_analysis(feature_importance)
    print("\n★ 重要度 X ドリフト検出 結果（上位）:")
    print(combined_result.head(30))
    
    # 結果保存
    engineer.save_results(output_dir)
    
    # 上位特徴量保存
    top_30 = feature_importance.head(35)
    top_30_path = f"{output_dir}\\top_30_feature_importance.csv"
    top_30.to_csv(top_30_path, index=False)
    print(f"上位30特徴量を保存しました: {top_30_path}")
    
    # ドリフト分析結果保存
    drift_path = f"{output_dir}\\feature_drift_analysis.csv"
    combined_result.to_csv(drift_path, index=False)
    print(f"分析結果を保存しました: {drift_path}")
    
    return df_features, feature_importance, combined_result


if __name__ == "__main__":
    df_features, feature_importance, combined_result = main()