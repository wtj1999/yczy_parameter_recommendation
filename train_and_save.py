import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
# from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, TimeSeriesSplit
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 全局配置
DATA_DIR = r'D:\xz2_yczy_data'
MODELS_DIR = r'./models_by_file'
FEATURE_COLS = ['ryczy004', 'ryczy205']
TARGET_COL = 'ryczy608'
TOLERANCE = 3  # ±3
TEST_SIZE = 0.2
TIME_SERIES = False
RANDOM_STATE = 42
MIN_SAMPLES = 200  # 训练前样本最少阈值

# 标称容量映射（文件名里含 '123Ah' 或 '133Ah'）
NOMINAL_MAP = {
    '123': 330,
    '133': 411
}

# 匹配 merged 文件
MERGED_RE = re.compile(r"merged_.*?(\d{3}Ah).*?\.csv", flags=re.IGNORECASE)  # 捕获 123Ah/133Ah 等


def data_split(X, y, test_size, time_series, random_state):
    if time_series:
        # 使用 TimeSeriesSplit：取最后一折作为测试集
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))
        train_index, test_index = splits[-1]
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]
    else:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
    print(f"样本数量 — 训练: {len(X_train_full)}, 测试: {len(X_test)}")
    return X_train_full, X_test, y_train_full, y_test


def train_on_file(file_path,
                  feature_cols=FEATURE_COLS,
                  target_col=TARGET_COL,
                  nominal_map=NOMINAL_MAP,
                  tolerance=TOLERANCE,
                  models_dir=MODELS_DIR,
                  test_size=TEST_SIZE,
                  time_series=TIME_SERIES,
                  random_state=RANDOM_STATE,
                  min_samples=MIN_SAMPLES):
    basename = os.path.basename(file_path)
    print(f"\n=== 处理文件: {basename} ===")
    # 从文件名提取规格 (123Ah / 133Ah 等)
    m = MERGED_RE.search(basename)
    if not m:
        print("无法从文件名推断规格 (例如 123Ah/133Ah)，跳过:", basename)
        return

    spec_str = m.group(1)  # e.g. '133Ah'
    spec_digits = re.search(r'(\d{3})', spec_str).group(1) if re.search(r'(\d{3})', spec_str) else None
    if spec_digits not in nominal_map:
        print(f"未在 nominal_map 中找到规格 {spec_digits}，跳过")
        return
    nominal = nominal_map[spec_digits]
    low = nominal - tolerance
    high = nominal + tolerance
    print(f"推断为 {spec_digits}Ah -> 标称 {nominal}, 过滤范围 {low} .. {high}")

    # 读取 CSV
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception as e:
        print("读取失败:", e)
        return

    # 取出目标列并数值化
    if target_col not in df.columns:
        print(f"文件没有目标列 {target_col}, 跳过")
        return
    # 先把 target 转为数值
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    # 过滤目标列在 nominal ± tolerance
    df = df[(df[target_col] > low) & (df[target_col] < high)].reset_index(drop=True)
    print(f"过滤后样本数: {len(df)}")
    if len(df) < min_samples:
        print(f"样本数量少于最小阈值 {min_samples}，跳过训练")
        return

    # 检查特征列是否存在
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        print("缺少以下特征列，跳过:", missing_features)
        return

    # 准备 X, y，并丢弃在 feature 或 target 上为 NaN 的行
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    valid_mask = X.notna().all(axis=1) & y.notna()
    removed = len(df) - valid_mask.sum()
    if removed > 0:
        print(f"去除 {removed} 行（feature/target 存在 NaN）")
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    # 训练/测试划分
    X_train, X_test, y_train, y_test = data_split(X, y, test_size=test_size, time_series=time_series, random_state=random_state)

    # 构建数值 pipeline（可空值中位数填充 + 标准化）
    # num_pipeline = Pipeline([
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('scaler', StandardScaler())
    # ])

    # 拟合 pipeline（注意：只对训练集 fit）
    # X_train[num_features := feature_cols] = num_pipeline.fit_transform(X_train[num_features])
    # X_test[num_features] = num_pipeline.transform(X_test[num_features])

    # 构建并训练 CatBoost
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=4,
        loss_function='RMSE',
        verbose=100,
        random_seed=random_state
    )

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print("模型训练失败:", e)
        return

    # 评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"评估结果 — MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")

    # 保存模型/ pipeline / metadata / 特征重要性图
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", os.path.splitext(basename)[0])
    model_dir = os.path.join(models_dir, safe_name)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "catboost_model.cbm")
    pipeline_path = os.path.join(model_dir, "num_pipeline.joblib")
    meta_path = os.path.join(model_dir, "metadata.json")
    fi_path = os.path.join(model_dir, "feature_importance.png")

    try:
        model.save_model(model_path)
        # joblib.dump(num_pipeline, pipeline_path)
    except Exception as e:
        print("保存模型或 pipeline 失败:", e)

    meta = {
        'source_file': basename,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'nominal': nominal,
        'tolerance': tolerance,
        'n_samples_filtered': len(df),
        'n_samples_used': len(X_train) + len(X_test),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'model_path': model_path,
        'pipeline_path': pipeline_path
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 特征重要性图
    # try:
    #     fi = model.get_feature_importance()
    #     feature_names = model.feature_names_ if hasattr(model, 'feature_names_') else feature_cols
    #     fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
    #     fi_df = fi_df.sort_values("importance", ascending=True)
    #     plt.figure(figsize=(8, 6))
    #     plt.barh(fi_df["feature"], fi_df["importance"])
    #     plt.xlabel("Importance")
    #     plt.title(f"Feature Importance ({safe_name})")
    #     plt.tight_layout()
    #     plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    # except Exception as e:
    #     print("绘制/保存特征重要性失败:", e)

    pred_out = os.path.join(model_dir, "predictions_sample.csv")
    pd.DataFrame({
        'y_test': y_test.reset_index(drop=True),
        'y_pred': y_pred
    }).to_csv(pred_out, index=False, encoding='utf-8-sig')

    print(f"模型与文件已保存到: {model_dir}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    merged_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().startswith('merged_') and f.lower().endswith('.csv')]
    if not merged_files:
        print("未找到 merged_*.csv 文件，检查 DATA_DIR")
        return

    for fp in merged_files:
        train_on_file(fp,
                      feature_cols=FEATURE_COLS,
                      target_col=TARGET_COL,
                      nominal_map=NOMINAL_MAP,
                      tolerance=TOLERANCE,
                      models_dir=MODELS_DIR,
                      test_size=TEST_SIZE,
                      time_series=TIME_SERIES,
                      random_state=RANDOM_STATE,
                      min_samples=MIN_SAMPLES)


if __name__ == "__main__":
    # main()
    import requests

    url = "http://10.2.128.43:8010/recommend"
    payload = {
        "technics_line_name": "133Ah",
        "device_code": "XZ2-02-03-ZY-YC-01",
        "ryczy407": 3,
        "ryczy004": 110.23,
        "ryczy205": 400.56
    }
    r = requests.post(url, json=payload, timeout=10)
    print(r.status_code, r.json())
