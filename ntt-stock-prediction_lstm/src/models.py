#NTT株価予測モデルの訓練と評価

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings

warnings.filterwarnings('ignore')

# グラフ設定
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


class StockPredictionModel:
    """株価予測モデルクラス（終値予測対応）"""
    
    def __init__(self, data_path: str, target_col: str = 'Close'):
        """
        初期化
        
        Args:
            data_path (str): 特徴量データのパス
            target_col (str): 予測対象列名（デフォルト: 'Close'）
        """
        self.data_path = data_path
        self.target_col = target_col
        self.df = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))  # 終値は正の値
        self.model = None
        self.history = None
        self.seq_length = 30
        
    def load_and_prepare_data(self):
        """データの読み込みと前処理"""
        print("データの読み込みと前処理を開始...")
        
        # データ読み込み
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        print(f"データ形状: {self.df.shape}")
        print(f"予測対象: {self.target_col}")
        
        # 欠損値チェック
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"欠損値を検出: {missing_data[missing_data > 0]}")
            self.df = self.df.dropna()
            print(f"欠損値除去後のデータ形状: {self.df.shape}")
        
        return self.df
    
    def select_features_by_importance_and_drift(self, stage=None):
        """
        重要度とドリフト分析結果に基づく特徴量選択（スコア計算版）

        Args:
            stage (str or None): 特徴量の段階選択。'step1', 'step2', 'step3', 'step4' または None（全部）

        Returns:
            list: 選択された特徴量リスト
        """
        print("\n特徴量選択（重要度 ÷ (1 + ドリフトスコア) ベース）...")
        
        drift_analysis_path = r"C:\Users\Take\python\ntt-stock-prediction_test\data\feature_drift_analysis.csv"
        try:
            drift_df = pd.read_csv(drift_analysis_path)
            print(f"ドリフト分析データを読み込み: {len(drift_df)}特徴量")
        except FileNotFoundError:
            print("ドリフト分析ファイルが見つかりません。デフォルト特徴量を使用します。")
            return self._get_default_features()

        # 段階的特徴量セット定義
        step_features = {
            'step1': [
                'Open', 'High', 'Low',
                'Daily_Return', 'Log_Return', 'Gap',
                'ATR_14_Lag1', 'RSI_14',
                'SMA_5_Lag1', 'SMA_20_Lag1', 'EMA_5_Lag1', 'EMA_20_Lag1',
                'Volume', 'Close_Lag_1', 'Close_Lag_2',
                'Return_Lag_1', 'Return_Lag_2',
                'Weekday_sin', 'Month_sin', 'Is_month_end'
            ],
            'step2': [
                'MACD', 'MACD_Signal', 'MACD_Hist',
                'CCI', 'ADX', 'PLUS_DI', 'MINUS_DI',
                'BB_Upper_Lag1', 'BB_Middle_Lag1', 'BB_Lower_Lag1', 'BB_Width_Lag1',
                'MOM_20', 'ATR_14_Lag1', 'Volume_Change', 'Volume_Ratio_5'
            ],
            'step3': [
                'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3',
                'Close_Lag_3', 'Return_Lag_3',
                'SMA_5_Lag_2', 'SMA_5_Lag_3',
                'EMA_5_Lag_2', 'EMA_5_Lag_3',
                'RSI_14_Lag_2', 'RSI_14_Lag_3',
                'Weekday_cos', 'Month_cos', 'Quarter_sin', 'Quarter_cos'
            ],
            'step4': [
                'AD', 'OBV', 'OBV_Change', 'Is_Earnings_Season', 'Is_Month_End'
            ]
        }

        # ドリフトと重要度ベースの候補抽出
        importance_threshold = 0.0001
        max_features = 30
        drift_threshold = 0.7

        filtered_df = drift_df[drift_df['ks_stat'] <= drift_threshold].copy()
        filtered_df['score'] = filtered_df['Importance'] / (1 + filtered_df['ks_stat'])
        filtered_df = filtered_df[filtered_df['Importance'] >= importance_threshold]
        filtered_df = filtered_df.sort_values('score', ascending=False)
        top_features = filtered_df.head(max_features)
        candidate_features = [f for f in top_features['Feature'] if f in self.df.columns]

        # 人力で追加したいステップをここで指定（明示的に step1, step2 を使用）
        selected_features = candidate_features.copy()
        selected_features += step_features['step1']
        selected_features += step_features['step2']
        selected_features += step_features['step3']
        #selected_features += step_features['step4']

        # 重複除去（順序保持）
        selected_features = list(dict.fromkeys(selected_features))

        # デバッグ表示
        print(f"\nデバッグ情報:")
        print(f"  - ドリフト閾値: {drift_threshold}")
        print(f"  - 重要度閾値: {importance_threshold}")
        print(f"  - フィルタ後特徴量数: {len(filtered_df)}")
        print(f"  - スコア上位候補数: {len(top_features)}")
        print(f"  - 最終選択数: {len(selected_features)}")

        # 選択特徴量のスコア情報表示（存在する場合のみ）
        if selected_features:
            print("\n選択された上位特徴量:")
            for i, feature in enumerate(selected_features, 1):
                if feature in top_features['Feature'].values:
                    row = top_features[top_features['Feature'] == feature].iloc[0]
                    score = row['score']
                    importance = row['Importance']
                    ks_stat = row['ks_stat']
                    print(f"  {i:2d}. {feature:<20} (スコア: {score:.6f}, 重要度: {importance:.6f}, ks_stat: {ks_stat:.3f})")
                else:
                    print(f"  {i:2d}. {feature:<20} (← スコア情報なし。手動追加特徴量)")
        else:
            print("\n一致する特徴量がありません。")
            print(f"CSV候補例: {top_features['Feature'].head().tolist()}")
            print(f"DF列例: {list(self.df.columns)[:5]}")

        if len(selected_features) == 0:
            print("選択された特徴量がありません。デフォルト特徴量を使用します。")
            return self._get_default_features()

        return selected_features



    
    def _get_default_features(self):
        """デフォルト特徴量（フォールバック用）"""
        default_features = [
            'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3',
            'Volume_Change', 'Gap', 'OBV_Change',
            'RSI_14', 'MACD', 'MACD_Hist',
            'Weekday_sin', 'Weekday_cos', 'Month_sin', 'Month_cos'
        ]
        return [col for col in default_features if col in self.df.columns]
    
    def prepare_sequences(self, feature_cols: list, holdout_days=30):
        """
        時系列シーケンスデータの準備
        
        Args:
            feature_cols (list): 使用する特徴量リスト
            holdout_days (int): 未来何日分をホールドアウト（テスト用に使わないか別に保持）
            
        Returns:
            tuple: 訓練データX_train, 検証データX_val, 訓練ラベルy_train, 検証ラベルy_val
        """
        print(f"\n時系列シーケンスデータの準備（シーケンス長: {self.seq_length}）...")

        total_len = len(self.df)
        
        # データ検証
        if total_len <= holdout_days + self.seq_length:
            raise ValueError(f"データが不足しています。必要: {holdout_days + self.seq_length + 1}, 実際: {total_len}")

        # 未来holdout_daysは学習・検証に含めず別に保持
        df_train_val = self.df.iloc[:total_len - holdout_days]
        df_holdout = self.df.iloc[total_len - holdout_days:]

        # 特徴量とターゲット分離（学習・検証用）
        X = df_train_val[feature_cols].values
        y = df_train_val[self.target_col].values
        
        # ターゲット値の検証
        if np.any(y <= 0):
            print(f"警告: ターゲット値に0以下の値があります。最小値: {y.min()}")
            # 0以下の値を最小正値で置き換え
            min_positive = y[y > 0].min() if np.any(y > 0) else 0.01
            y = np.maximum(y, min_positive)
            print(f"0以下の値を {min_positive} で置き換えました")

        # スケーリング（fitは学習・検証用だけに）
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # シーケンス作成
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        if len(X_seq) == 0:
            raise ValueError("シーケンスデータが作成できませんでした。データ長またはシーケンス長を確認してください。")
            
        print(f"シーケンスデータ形状: X={X_seq.shape}, y={y_seq.shape}")

        # 学習:検証 = 7:3（割合は調整可）
        train_size = max(1, int(len(X_seq) * 0.7))
        
        X_train = X_seq[:train_size]
        y_train = y_seq[:train_size]

        X_val = X_seq[train_size:]
        y_val = y_seq[train_size:]

        # ホールドアウトデータの処理
        if len(df_holdout) > self.seq_length:
            X_holdout = df_holdout[feature_cols].values
            y_holdout = df_holdout[self.target_col].values
            
            # ホールドアウトのターゲット値検証
            if np.any(y_holdout <= 0):
                min_positive = y_holdout[y_holdout > 0].min() if np.any(y_holdout > 0) else 0.01
                y_holdout = np.maximum(y_holdout, min_positive)

            X_holdout_scaled = self.feature_scaler.transform(X_holdout)
            y_holdout_scaled = self.target_scaler.transform(y_holdout.reshape(-1, 1)).flatten()

            X_test, y_test = self._create_sequences(X_holdout_scaled, y_holdout_scaled)
        else:
            # ホールドアウトデータが不足している場合は検証データの一部を使用
            print("警告: ホールドアウトデータが不足しています。検証データの一部をテストに使用します。")
            val_split = len(X_val) // 2
            X_test = X_val[val_split:]
            y_test = y_val[val_split:]
            X_val = X_val[:val_split]
            y_val = y_val[:val_split]

        print(f"訓練データ: {X_train.shape}")
        print(f"検証データ: {X_val.shape}")
        print(f"テストデータ: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test


    
    def _create_sequences(self, X, y):
        """シーケンスデータ作成のヘルパー関数"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
            y_seq.append(y[i + self.seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape: tuple, model_type: str = 'lstm'):
        """
        モデル構築
        
        Args:
            input_shape (tuple): 入力データの形状
            model_type (str): モデルタイプ ('lstm', 'gru', 'bidirectional')
            
        Returns:
            keras.Model: 構築されたモデル
        """
        print(f"\n{model_type.upper()}モデルを構築中...")
        
        if model_type == 'lstm':
            model = self._build_lstm_model(input_shape)
        elif model_type == 'gru':
            model = self._build_gru_model(input_shape)
        elif model_type == 'bidirectional':
            model = self._build_bidirectional_lstm_model(input_shape)
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")
        
        print("モデル構築完了")
        model.summary()
        
        return model
    
    def _build_lstm_model(self, input_shape):
        """LSTM モデル構築（終値予測用）"""
        model = Sequential([
            LSTM(units=128, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=l2(0.0001)),
            Dropout(0.3),
            LSTM(units=64, return_sequences=True,
                 kernel_regularizer=l2(0.0001)),
            Dropout(0.3),
            LSTM(units=32, kernel_regularizer=l2(0.0001)),
            Dropout(0.3),
            Dense(units=16, activation='relu', kernel_regularizer=l2(0.0001)),
            Dense(units=1, activation='linear')  # 終値予測は線形出力
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',  # 終値予測にはMSE
            metrics=['mae']
        )
        
        return model
    
    def _build_gru_model(self, input_shape):
        """GRU モデル構築"""
        model = Sequential([
            GRU(units=128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            GRU(units=64, return_sequences=True),
            Dropout(0.3),
            GRU(units=32),
            Dropout(0.3),
            Dense(units=16, activation='relu'),
            Dense(units=1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_bidirectional_lstm_model(self, input_shape):
        """双方向LSTM モデル構築"""
        model = Sequential([
            Bidirectional(LSTM(units=64, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(units=32, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(units=16)),
            Dropout(0.3),
            Dense(units=8, activation='relu'),
            Dense(units=1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=100):
        """
        モデル訓練
        
        Args:
            X_train, X_val, y_train, y_val: 訓練・検証データ
            epochs (int): エポック数
            
        Returns:
            History: 訓練履歴
        """
        print(f"\nモデル訓練開始（エポック数: {epochs}）...")
        
        # データ検証
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("訓練または検証データが空です")
            
        # バッチサイズを調整（データサイズに応じて）
        batch_size = min(32, max(1, len(X_train) // 10))
        print(f"バッチサイズ: {batch_size}")
        
        # コールバック設定
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=2,
                min_delta=1e-6,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        try:
            # 訓練実行
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
            print(f"訓練中にエラーが発生しました: {e}")
            # より小さなバッチサイズで再試行
            print("バッチサイズを1に変更して再試行します...")
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=min(epochs, 50),
                batch_size=1,
                verbose=1
            )
        
        print("モデル訓練完了")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        モデル評価
        
        Args:
            X_test, y_test: テストデータ
            
        Returns:
            dict: 評価指標
        """
        print("\nモデル評価中...")
        
        # 予測実行
        y_pred_scaled = self.model.predict(X_test).flatten()
        
        # スケール復元
        y_test_orig = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # 負の値を0に置き換え（株価は負になれない）
        y_pred_orig = np.maximum(y_pred_orig, 0.01)
        
        # 評価指標計算
        metrics = self._calculate_metrics(y_test_orig, y_pred_orig)
        
        # 結果表示
        print("\n評価結果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        
        return metrics, y_test_orig, y_pred_orig
    
    def _calculate_metrics(self, y_true, y_pred):
        """評価指標計算（終値予測用）"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE（平均絶対パーセント誤差）
        # 0での除算を防ぐ
        y_true_safe = np.maximum(np.abs(y_true), 0.01)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        
        # 方向性精度（価格変化の方向）
        price_changes_true = np.diff(y_true)
        price_changes_pred = np.diff(y_pred)
        direction_accuracy = np.mean(np.sign(price_changes_true) == np.sign(price_changes_pred)) * 100
        
        return {
            'RMSE': f"{rmse:.2f}",
            'MAE': f"{mae:.2f}",
            'R²': f"{r2:.4f}",
            'MAPE': f"{mape:.2f}%",
            'Direction Accuracy': f"{direction_accuracy:.2f}%"
        }
    
    def plot_results(self, y_test, y_pred, history=None):
        """結果可視化"""
        print("\n結果可視化中...")
        
        # 学習曲線
        if history is not None:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history.history['loss'], label='Training Loss', alpha=0.8)
            plt.plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
            plt.title('Learning Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 予測 vs 実際
        plt.subplot(1, 3, 2)
        plt.scatter(y_test, y_pred, alpha=0.6, s=20)
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('Actual Close Price')
        plt.ylabel('Predicted Close Price')
        plt.title('Predicted vs Actual')
        plt.grid(True, alpha=0.3)
        
        # 予測誤差分布
        plt.subplot(1, 3, 3)
        errors = y_test - y_pred
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 日付インデックスを取得（dfの末尾からy_testの分だけ取る）
        self.df.set_index('Date', inplace=True)
        test_dates = self.df.index[-len(y_test):]  # self.dfは元データ
        
        # 時系列プロット
        plt.figure(figsize=(14, 6))
        plt.plot(test_dates, y_test, label='Actual Close Price', alpha=0.8, linewidth=1)
        plt.plot(test_dates, y_pred, label='Predicted Close Price', color='red', linestyle='--', alpha=0.8, linewidth=1)
        plt.title('Close Price Prediction Results')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    
    def run_full_pipeline(self, model_type='lstm', epochs=100):
        """
        完全なパイプライン実行
        
        Args:
            model_type (str): モデルタイプ
            epochs (int): エポック数
            
        Returns:
            dict: 評価結果
        """
        print("株価予測モデル パイプライン開始")
        print("=" * 50)
        
        # 1. データ準備
        self.load_and_prepare_data()
        
        # 2. 特徴量選択
        feature_cols = self.select_features_by_importance_and_drift()
        
        # 3. シーケンスデータ準備
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_sequences(feature_cols)
        
        # 4. モデル構築
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape, model_type)
        
        # 5. モデル訓練
        history = self.train_model(X_train, X_val, y_train, y_val, epochs)
        
        # 6. モデル評価
        metrics, y_test_orig, y_pred_orig = self.evaluate_model(X_test, y_test)
        
        # 7. 結果可視化
        self.plot_results(y_test_orig, y_pred_orig, history)
        
        print("\nパイプライン完了")
        print("=" * 50)
        
        return {
            'metrics': metrics,
            'model': self.model,
            'history': history,
            'predictions': (y_test_orig, y_pred_orig)
        }


def main():
    """メイン実行関数"""
    # データパス
    data_path = r"C:\Users\Take\python\ntt-stock-prediction_test\data\features_talib.csv"
    
    # モデル初期化（終値予測）
    predictor = StockPredictionModel(data_path, target_col='Close')
    
    # パイプライン実行
    results = predictor.run_full_pipeline(
        model_type='lstm',  # 'lstm', 'gru', 'bidirectional'から選択
        epochs=20
    )
    
    # 結果サマリー
    print("\n最終結果サマリー:")
    print("-" * 30)
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value}")
    
    return results


if __name__ == "__main__":
    results = main()