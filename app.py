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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
import logging

# --- Setup ---
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="Demand Forecast", layout="wide")
st.title("📦 Demand Forecasting App")

# --- Session State ---
if "trained" not in st.session_state:
    st.session_state.trained = False
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None

# --- Features ---
all_features = ['Week', 'SKU_encoded', 'lag_1', 'lag_2', 'lag_3', 'lag_4',
                'rolling_mean_2', 'rolling_mean_3', 'rolling_mean_4',
                'sku_mean', 'cumulative_units', 'units_to_sku_mean']

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)[['SKU', 'Week', 'Units']]
    st.success("CSV loaded successfully.")
    st.write("Preprocessing file...")

    st.subheader("Week Range")
    range_type = st.radio("Select Range", ['Auto-detect', 'Manual'])
    if range_type == 'Auto-detect':
        start_week, end_week = int(data['Week'].min()), int(data['Week'].max())
    else:
        start_week = st.number_input("Start Week", min_value=1, value=1)
        end_week = st.number_input("End Week", min_value=start_week, value=start_week + 5)

    full_weeks = list(range(start_week, end_week + 1))
    data['SKU'] = data['SKU'].astype(str)
    full_index = pd.MultiIndex.from_product([data['SKU'].unique(), full_weeks], names=['SKU', 'Week'])
    data = data.set_index(['SKU', 'Week']).reindex(full_index).reset_index()
    data['Units'] = data['Units'].apply(lambda x: max(x, 0))
    data.fillna(0, inplace=True)

    Q1, Q3 = data['Units'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = data[(data['Units'] < lower_bound) | (data['Units'] > upper_bound)]
    st.write("Number of outliers in 'Units'", len(outliers))

    if st.checkbox("Apply Outlier Capping"):
        q10, q90 = data['Units'].quantile(0.1), data['Units'].quantile(0.9)
        data['Units'] = data['Units'].clip(lower=q10, upper=q90)

    le = LabelEncoder()
    data['SKU_encoded'] = le.fit_transform(data['SKU']).astype(int)
    os.makedirs("saved_data", exist_ok=True)
    data.to_csv("saved_data/preprocessed.csv", index=False)

    for i in range(1, 5):
        data[f'lag_{i}'] = data.groupby('SKU')['Units'].shift(i)
    data['rolling_mean_2'] = data.groupby('SKU')['Units'].shift(1).rolling(2).mean().reset_index(0, drop=True)
    data['rolling_mean_3'] = data.groupby('SKU')['Units'].shift(1).rolling(3).mean().reset_index(0, drop=True)
    data['rolling_mean_4'] = data.groupby('SKU')['Units'].shift(1).rolling(4).mean().reset_index(0, drop=True)
    data.fillna(0, inplace=True)
    data = data[data['Week'] >= (start_week + 4)]

    all_weeks = sorted(data['Week'].unique())
    split_point = int(len(all_weeks) * 0.8)
    train_weeks = all_weeks[:split_point]
    test_weeks = all_weeks[split_point:]
    train = data[data['Week'].isin(train_weeks)]

    sku_cumsum = train.groupby('SKU')['Units'].sum().to_dict()
    sku_mean_map = train.groupby('SKU')['Units'].mean().to_dict()
    data['cumulative_units'] = data['SKU'].map(sku_cumsum)
    data['sku_mean'] = data['SKU'].map(sku_mean_map)
    data['units_to_sku_mean'] = data['Units'] / (data['sku_mean'] + 1e-5)

    features = all_features
    target = 'Units'
    features_to_scale = [f for f in features if f != 'SKU_encoded']

    train = data[data['Week'].isin(train_weeks)]
    test = data[data['Week'].isin(test_weeks)]

    scaler = RobustScaler()
    scaler.fit(train[features_to_scale])
    joblib.dump(scaler, 'saved_models/robust_scaler.save')
    data[features_to_scale] = scaler.transform(data[features_to_scale])

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    X_train['SKU_encoded'] = X_train['SKU_encoded'].astype(str)
    X_test['SKU_encoded'] = X_test['SKU_encoded'].astype(str)
    cat_features = ['SKU_encoded']

    if st.button("Train Models and Forecast"):
        models = {
            'CatBoostRegressor': CatBoostRegressor(cat_features=cat_features, verbose=0, iterations=200, depth=6, learning_rate=0.1, random_strength=0.01, l2_leaf_reg=2),
            'XGB Regressor': XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, gamma=0.1, min_child_weight=3, verbosity=0, random_state=42),
            'Light GBM Regressor': LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=-1),
            'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_split=10, min_samples_leaf=5, max_features=0.6, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(learning_rate=0.1, random_state=42),
        }
        results = []
        for name, model in models.items():
            st.write("Running", name)

            if name == 'CatBoostRegressor':
                train_pool = Pool(X_train, y_train, cat_features=['SKU_encoded'])
                test_pool = Pool(X_test, y_test, cat_features=['SKU_encoded'])
                model.fit(train_pool)
                joblib.dump(model, f"saved_models/{name.lower()}.joblib")
                y_pred = model.predict(test_pool)
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

        best_model_name = results_df.loc[0, 'Model']
        best_model_file = best_model_name.lower()
        with open("saved_models/best_model.txt", "w") as f:
            f.write(best_model_file)

        model = joblib.load(f"saved_models/{best_model_file}.joblib")
        scaler = joblib.load("saved_models/robust_scaler.save")
        st.write("Forecasting using best model: ", best_model_name)

        forecast_weeks = 6
        data = data.sort_values(['SKU', 'Week'])
        data['SKU_encoded'] = data['SKU_encoded'].astype(int).astype(str)
        inverse_map = data[['SKU', 'SKU_encoded']].drop_duplicates().set_index('SKU_encoded')['SKU'].to_dict()
        train = data[data['Week'].isin(train_weeks)]
        sku_mean_map = train.groupby('SKU')['Units'].mean().to_dict()

        forecast_dict = {}
        for sku in data['SKU_encoded'].unique():
            sku_data = data[data['SKU_encoded'] == sku].sort_values('Week')
            if sku_data.empty:
                continue
            sku_original = inverse_map.get(sku)
            if sku_original is None:
                continue

            last_row = sku_data.iloc[-1].copy()
            lags = [last_row[f'lag_{i}'] for i in range(1, 5)]
            cumulative_units = train[train['SKU'] == sku_original]['Units'].sum()
            forecasts = []

            for _ in range(forecast_weeks):
                sku_mean = sku_mean_map.get(sku_original, 0)
                input_row = {
                    'SKU_encoded': sku,
                    'Week': last_row['Week'] + 1,
                    'lag_1': lags[0], 'lag_2': lags[1], 'lag_3': lags[2], 'lag_4': lags[3],
                    'rolling_mean_2': np.mean(lags[:2]),
                    'rolling_mean_3': np.mean(lags[:3]),
                    'rolling_mean_4': np.mean(lags),
                    'sku_mean': sku_mean,
                    'cumulative_units': cumulative_units,
                    'units_to_sku_mean': last_row['Units'] / (sku_mean + 1e-5)
                }

                input_df = pd.DataFrame([input_row])
                scaled_numeric = scaler.transform(input_df[features_to_scale])
                scaled_df = pd.DataFrame(scaled_numeric, columns=features_to_scale)
                scaled_df.insert(0, 'SKU_encoded', sku)
                X_final = scaled_df[features]

                pred = model.predict(X_final)[0]
                pred = max(0, pred)
                forecasts.append(round(pred, 2))
                lags = [pred] + lags[:3]
                last_row['Units'] = pred
                last_row['Week'] += 1
                cumulative_units += pred

            forecast_dict[sku] = forecasts

        forecast_df = pd.DataFrame.from_dict(forecast_dict, orient='index')
        forecast_df.columns = [f"Week_{i+1}" for i in range(forecast_weeks)]
        forecast_df['SKU'] = forecast_df.index.map(inverse_map)
        forecast_df = forecast_df[['SKU'] + [f"Week_{i+1}" for i in range(forecast_weeks)]]

        st.session_state.forecast_df = forecast_df

#  Display forecast
if st.session_state.forecast_df is not None:
    st.subheader("📈 Forecast of Next 6 Weeks")
    st.dataframe(st.session_state.forecast_df)
    csv = st.session_state.forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast", data=csv, file_name="forecast.csv")
    if st.button("🔄 Reset Forecast View"):
        st.session_state.forecast_df = None
        st.session_state.trained = False
