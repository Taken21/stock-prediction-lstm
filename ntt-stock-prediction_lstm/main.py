from src.data_loader import load_data
from src.features import create_features
from src.models import StockPredictionModel

# ファイルパス
data_path = "data/stock_data.csv"

# 1. データ読み込み
df = load_data(data_path)

# 2. 特徴量エンジニアリング
df = create_features(df)

# 3. モデル構築と予測
predictor = StockPredictionModel(df, target_col="Close")
predictor.run_full_pipeline(model_type='lstm', epochs=100)
