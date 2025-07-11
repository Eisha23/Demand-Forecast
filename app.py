import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import logging

os.makedirs("saved_models", exist_ok=True)
# Configure once, at the top of your app
logging.basicConfig(
    filename='app.log',   # log file (can omit to log only to console)
    level=logging.INFO,   # can be DEBUG, INFO, WARNING, ERROR
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(page_title="Demand Forecast", layout="wide")
st.title("📦 Demand Forecasting App")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    logging.info("Reading CSV file.")
    data = pd.read_csv(uploaded_file)
    data = data[['SKU', 'Week', 'Units']]
    st.success("CSV loaded successfully.")

    # Week range
    st.subheader("Week Range")
    range_type = st.radio("Select Range", ['Auto-detect', 'Manual'])
    if range_type == 'Auto-detect':
        start_week, end_week = int(data['Week'].min()), int(data['Week'].max())
    else:
        start_week = st.number_input("Start Week", min_value=1, value=1)
        end_week = st.number_input("End Week", min_value=start_week, value=start_week + 5)

    # Fill missing weeks
    logging.info("Preprocessing the dataset.")
    full_weeks = list(range(start_week, end_week + 1))
    data['SKU'] = data['SKU'].astype(str)
    full_index = pd.MultiIndex.from_product([data['SKU'].unique(), full_weeks], names=['SKU', 'Week'])
    data = data.set_index(['SKU', 'Week']).reindex(full_index).reset_index()

    # Clean
    data['Units'] = data['Units'].apply(lambda x: max(x, 0))
    data.fillna(0, inplace=True)
    
    #Check outliers
    Q1 = data['Units'].quantile(0.25)
    Q3 = data['Units'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Get outliers
    outliers = data[(data['Units'] < lower_bound) | (data['Units'] > upper_bound)]
    st.write("Number of outliers in 'Units'", len(outliers))


    if st.checkbox("Apply Outlier Capping"):
        q10, q90 = data['Units'].quantile(0.1), data['Units'].quantile(0.9)
        data['Units'] = data['Units'].clip(lower=q10, upper=q90)

    # Label Encoding
    le = LabelEncoder()
    data['SKU_encoded'] = le.fit_transform(data['SKU']).astype(int)
    os.makedirs("saved_data", exist_ok=True)
    data.to_csv("saved_data/preprocessed.csv", index=False)
    logging.info("Data Preprocessed")


    # Lag features
    logging.info("Feature Engineering")
    for i in range(1, 5):
        data[f'lag_{i}'] = data.groupby('SKU')['Units'].shift(i)
    data.fillna(0, inplace=True)

    # Rolling means
    data['rolling_mean_2'] = data.groupby('SKU')['Units'].shift(1).rolling(2).mean().reset_index(0, drop=True)
    data['rolling_mean_3'] = data.groupby('SKU')['Units'].shift(1).rolling(3).mean().reset_index(0, drop=True)
    data['rolling_mean_4'] = data.groupby('SKU')['Units'].shift(1).rolling(4).mean().reset_index(0, drop=True)
    data.fillna(0, inplace=True)

    # Remove early rows for lag validity
    data = data[data['Week'] >= (start_week + 4)]

    # Train-test split weeks
    all_weeks = sorted(data['Week'].unique())
    split_point = int(len(all_weeks) * 0.8)
    train_weeks = all_weeks[:split_point]
    train = data[data['Week'].isin(train_weeks)]

    test_weeks = all_weeks[split_point:]

    # Additional features
    # create "cumulative_units", "sku mean" and "units to sku mean" features
    sku_cumsum = train.groupby('SKU')['Units'].sum().to_dict()
    sku_mean_map = train.groupby('SKU')['Units'].mean().to_dict()

    data['cumulative_units'] = data['SKU'].map(sku_cumsum)
    data['sku_mean'] = data['SKU'].map(sku_mean_map)
    data['units_to_sku_mean'] = data['Units'] / (data['sku_mean'] + 1e-5)
    data.to_csv("saved_data/features.csv", index=False)

    # Features
    features = ['Week', 'SKU_encoded', 'lag_1', 'lag_2', 'lag_3', 'lag_4',
                'rolling_mean_2', 'rolling_mean_3', 'rolling_mean_4',
                'sku_mean', 'cumulative_units', 'units_to_sku_mean']
    target = 'Units'
    

    # Scale numeric columns
    features_to_scale = ['Week','Units','lag_1', 'lag_2', 'lag_3', 'lag_4',
                     'rolling_mean_2', 'rolling_mean_3', 'rolling_mean_4','sku_mean',
                     'cumulative_units', 'units_to_sku_mean']
    #split train test data
    all_weeks = sorted(data['Week'].unique())
    split_point = int(len(all_weeks) * 0.8)

    train_weeks = all_weeks[:split_point]
    test_weeks = all_weeks[split_point:]

    train = data[data['Week'].isin(train_weeks)]
    test = data[data['Week'].isin(test_weeks)]
    logging.info("Scaling the data.")
    scaler = RobustScaler()
    scaler.fit(data[data['Week'].isin(train_weeks)][features_to_scale])
    joblib.dump(scaler, 'saved_models/robust_scaler.save')
    data[features_to_scale] = scaler.transform(data[features_to_scale])

    # Final splits
     # Final training and testing sets
    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]
    X_train['SKU_encoded'] = X_train['SKU_encoded'].astype(int)

    X_test['SKU_encoded'] = X_test['SKU_encoded'].astype(int)
    
    # Models
    if st.button("Train Models and Forecast"):
        models = {
           # 'CatBoost': CatBoostRegressor(cat_features=['SKU_encoded'], verbose=0),
           # 'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8),
            'LightGBM': LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=4),
            'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=8),
            'GradientBoosting': GradientBoostingRegressor()
        }
       

        results = []
        for name, model in models.items():
            logging.info("Running", name)
            st.write("Running", name)
            model.fit(X_train, y_train)
            file_name = name.replace(' ', '_').lower()
            joblib.dump(model, f"saved_models/{file_name}.joblib")
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

        #  Save the name of the best model
        
        best_model_name = results_df.loc[0, 'Model']
        file_safe_model_name = best_model_name.replace(' ', '_').lower()
        with open("saved_models/best_model.txt", "w") as f:
            f.write(file_safe_model_name)

        st.success(f"✅ Best model: {best_model_name}")
        logging.info("Best model saved.")

        # Save best model name to file for later use
        with open("saved_models/best_model.txt", "w") as f:
            f.write(best_model_name.replace(' ', '_').lower())

        # Forecast next 6 weeks
        logging.info("Forecasting next 6 weeks...")
        st.subheader("📈 Forecast Next 6 Weeks (CatBoost)")
        forecast_weeks = 6
        with open("saved_models/best_model.txt", "r") as f:
            best_model_file = f.read().strip()  

        model = joblib.load(f"saved_models/{best_model_file}.joblib")
        scaler = joblib.load("saved_models/robust_scaler.save")

        forecast_dict = {}
        inverse_map = data[['SKU', 'SKU_encoded']].drop_duplicates().set_index('SKU_encoded')['SKU'].to_dict()
        sku_mean_map = train.groupby('SKU')['Units'].mean().to_dict()

        for sku in data['SKU_encoded'].unique():
            sku_data = data[data['SKU_encoded'] == sku].sort_values('Week')
            if sku_data.empty: continue

            sku_original = inverse_map[sku]
            last_row = sku_data.iloc[-1].copy()
            lags = [last_row[f'lag_{i}'] for i in range(1, 5)]
            cumulative_units = train[train['SKU'] == sku_original]['Units'].sum()
            forecasts = []

            for _ in range(forecast_weeks):
                sku_mean = sku_mean_map.get(sku_original, 0)
                input_row = pd.DataFrame([{
                    'SKU_encoded': sku,
                    'Week': last_row['Week'] + 1,
                    'Units': last_row['Units'],
                    'lag_1': lags[0], 'lag_2': lags[1], 'lag_3': lags[2], 'lag_4': lags[3],
                    'rolling_mean_2': np.mean(lags[:2]),
                    'rolling_mean_3': np.mean(lags[:3]),
                    'rolling_mean_4': np.mean(lags),
                    'sku_mean': sku_mean,
                    'cumulative_units': cumulative_units,
                    'units_to_sku_mean': last_row['Units'] / (sku_mean + 1e-5)
                }])

                scaled = scaler.transform(input_row[features_to_scale])
                scaled_df = pd.DataFrame(scaled, columns=features_to_scale)
                scaled_df.insert(0, 'SKU_encoded', sku)
                pool = Pool(scaled_df, cat_features=['SKU_encoded'])

                pred = model.predict(pool)[0]
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

        st.dataframe(forecast_df)
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast", data=csv, file_name="forecast.csv")
