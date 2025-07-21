import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Setup
st.set_page_config(page_title="Demand Forecast", layout="wide")
st.title("📦 Demand Forecasting App")

# Session State Initialization
for key in ['file_uploaded', 'file_processed', 'forecast_ready', 'trained', 'forecast_df',
            'preprocessed_df', 'outliers_df', 'outlier_summary', 'features_df', 'model_ready_df']:
    if key not in st.session_state:
        st.session_state[key] = False if key in ['file_uploaded', 'file_processed', 'forecast_ready', 'trained'] else None

os.makedirs("saved_models", exist_ok=True)

# File Upload
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    st.session_state.file_uploaded = True
    st.success("CSV uploaded. Click below to process.")
    if st.button("🚀 Process File"):
        st.session_state.raw_data = pd.read_csv(uploaded_file)[['SKU', 'Week', 'Units']]
        st.session_state.file_processed = True
        st.rerun()

# Main Logic Block
if st.session_state.file_processed and not st.session_state.trained:
    data = st.session_state.raw_data.copy()

    st.subheader("Week Range")
    range_type = st.radio("Select Range", ['Auto-detect', 'Manual'])
    if range_type == 'Auto-detect':
        start_week, end_week = int(data['Week'].min()), int(data['Week'].max())
    else:
        start_week = st.number_input("Start Week", min_value=1, value=1)
        end_week = st.number_input("End Week", min_value=start_week, value=start_week + 5)

    # Fill missing weeks
    full_weeks = list(range(start_week, end_week + 1))
    data['SKU'] = data['SKU'].astype(str)
    full_index = pd.MultiIndex.from_product([data['SKU'].unique(), full_weeks], names=['SKU', 'Week'])
    data = data.set_index(['SKU', 'Week']).reindex(full_index).reset_index()
    data.fillna(0, inplace=True)
    data['Units'] = data['Units'].apply(lambda x: max(x, 0))
    data = data.sort_values(by=['SKU', 'Week'])
    st.session_state.preprocessed_df = data.copy()

    # Capping outliers
    sku_outlier_info = []
    for sku in data['SKU'].unique():
        sku_mask = data['SKU'] == sku
        sku_data = data[sku_mask]
        Q1 = sku_data['Units'].quantile(0.25)
        Q3 = sku_data['Units'].quantile(0.75)
        IQR = Q3 - Q1
        lower = max(Q1 - 1.5 * IQR, 0)
        upper = Q3 + 1.5 * IQR
        outliers = sku_data[(sku_data['Units'] < lower) | (sku_data['Units'] > upper)]
        sku_outlier_info.append({
            "SKU": sku,
            "Outlier_Count": len(outliers),
            "Outlier_Weeks": ",".join(str(w) for w in outliers['Week'].tolist()),
            "Outlier_Values": ",".join(str(v) for v in outliers['Units'].tolist()),
            "Lower_Cap": round(lower, 2),
            "Upper_Cap": round(upper, 2)
        })
        data.loc[sku_mask, 'Units'] = sku_data['Units'].clip(lower=lower, upper=upper)
    st.session_state.outliers_df = data.copy()
    st.session_state.outlier_summary = pd.DataFrame(sku_outlier_info)

    # Encoding and feature engineering
    le = LabelEncoder()
    data['SKU_encoded'] = le.fit_transform(data['SKU']).astype(int)
    for i in range(1, 5):
        data[f'lag_{i}'] = data.groupby('SKU')['Units'].shift(i)
    data['rolling_mean_2'] = data.groupby('SKU')['Units'].shift(1).rolling(2).mean().reset_index(0, drop=True)
    data['rolling_mean_3'] = data.groupby('SKU')['Units'].shift(1).rolling(3).mean().reset_index(0, drop=True)
    data['rolling_mean_4'] = data.groupby('SKU')['Units'].shift(1).rolling(4).mean().reset_index(0, drop=True)
    data.fillna(0, inplace=True)
    data = data[data['Week'] >= (start_week + 4)]
    st.session_state.features_df = data.copy()

    features = ['Week', 'SKU_encoded', 'lag_1', 'lag_2', 'lag_3','lag_4',
                'rolling_mean_2', 'rolling_mean_3', 'rolling_mean_4']
    target = 'Units'
    features_to_scale = [f for f in features if f != 'SKU_encoded']

    # Scaling
    scaler = RobustScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    joblib.dump(scaler, 'saved_models/robust_scaler.save')
    st.session_state.model_ready_df = data.copy()

    # Train-test split
    all_weeks = sorted(data['Week'].unique())
    split_point = int(len(all_weeks) * 0.8)
    train_weeks = all_weeks[:split_point]
    test_weeks = all_weeks[split_point:]
    train = data[data['Week'].isin(train_weeks)]
    test = data[data['Week'].isin(test_weeks)]
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # Model training
    models = {
        'CatBoostRegressor': CatBoostRegressor(verbose=0, iterations=200, depth=6, learning_rate=0.1),
        'XGB Regressor': XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6,
                                      subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                                      gamma=0.1, min_child_weight=3, verbosity=0, random_state=42),
        'Light GBM Regressor': LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                                             reg_lambda=1.0, random_state=42, verbosity=-1),
        'Gradient Boosting': GradientBoostingRegressor(learning_rate=0.1, random_state=42),
    }
    results = []
    for name, model in models.items():
        st.write("Training:", name)
        if name == 'CatBoostRegressor':
            model.fit(X_train, y_train, cat_features=['SKU_encoded'])
        else:
            model.fit(X_train, y_train)
        joblib.dump(model, f"saved_models/{name.lower()}.joblib")
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R² Score': r2_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
    st.dataframe(results_df)

    # Forecasting
    best_model_name = results_df.loc[0, 'Model']
    joblib.dump(best_model_name, 'saved_models/best_model.txt')
    model = joblib.load(f"saved_models/{best_model_name.lower()}.joblib")
    scaler = joblib.load("saved_models/robust_scaler.save")

    st.write("Forecasting using best model:", best_model_name)
    forecast_weeks = 6
    forecast_dict = {}
    data = data.sort_values(['SKU', 'Week'])
    inverse_map = data[['SKU', 'SKU_encoded']].drop_duplicates().set_index('SKU_encoded')['SKU'].to_dict()

    for sku in data['SKU_encoded'].unique():
        sku_data = data[data['SKU_encoded'] == sku].sort_values('Week')
        if sku_data.empty:
            continue
        sku_original = inverse_map.get(sku)
        last_row = sku_data.iloc[-1]
        lags = [last_row[f'lag_{i}'] for i in range(1, 5)]
        forecasts = []
        for _ in range(forecast_weeks):
            input_row = {
                'SKU_encoded': sku,
                'Week': last_row['Week'] + 1,
                'lag_1': lags[0], 'lag_2': lags[1], 'lag_3': lags[2], 'lag_4': lags[3],
                'rolling_mean_2': np.mean(lags[:2]),
                'rolling_mean_3': np.mean(lags[:3]),
                'rolling_mean_4': np.mean(lags),
            }
            input_df = pd.DataFrame([input_row])
            scaled_numeric = scaler.transform(input_df[features_to_scale])
            scaled_df = pd.DataFrame(scaled_numeric, columns=features_to_scale)
            scaled_df.insert(0, 'SKU_encoded', sku)
            pred = model.predict(scaled_df)[0]
            pred = max(0, pred)
            forecasts.append(round(pred, 2))
            lags = [pred] + lags[:3]
            last_row['Week'] += 1
        forecast_dict[sku] = forecasts

    forecast_df = pd.DataFrame.from_dict(forecast_dict, orient='index')
    forecast_df.columns = [f"Week_{i+1}" for i in range(forecast_weeks)]
    forecast_df['SKU'] = forecast_df.index.map(inverse_map)
    forecast_df = forecast_df[['SKU'] + [f"Week_{i+1}" for i in range(forecast_weeks)]]
    st.session_state.forecast_df = forecast_df
    st.session_state.trained = True
    st.session_state.forecast_ready = True
    st.rerun()

# Forecast Display
if st.session_state.forecast_ready:
    st.subheader("📈 Forecast of Next 6 Weeks")
    st.dataframe(st.session_state.forecast_df)
    csv = st.session_state.forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast", data=csv, file_name="forecast.csv")
    if st.button("🔄 Reset Forecast View"):
        st.session_state.clear()
        st.rerun()

# Sidebar Downloads
st.sidebar.subheader("📥 Download Files")
with st.sidebar.expander("Downloads"):
    if st.session_state.preprocessed_df is not None:
        st.sidebar.download_button("Preprocessed CSV", data=st.session_state.preprocessed_df.to_csv(index=False).encode('utf-8'), file_name="preprocessed_data.csv")
    if st.session_state.outlier_summary is not None:
        st.sidebar.download_button("Outlier Summary", data=st.session_state.outlier_summary.to_csv(index=False).encode('utf-8'), file_name="outliers_summary.csv")
    if st.session_state.outliers_df is not None:
        st.sidebar.download_button("Outliers Preprocessed", data=st.session_state.outliers_df.to_csv(index=False).encode('utf-8'), file_name="outliers_preprocessed.csv")
    if st.session_state.features_df is not None:
        st.sidebar.download_button("Features CSV", data=st.session_state.features_df.to_csv(index=False).encode('utf-8'), file_name="features.csv")
    if st.session_state.model_ready_df is not None:
        st.sidebar.download_button("Model Ready CSV", data=st.session_state.model_ready_df.to_csv(index=False).encode('utf-8'), file_name="model_ready.csv")

