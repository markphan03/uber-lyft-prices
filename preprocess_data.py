import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


# =========================
# Load Data
# =========================
def load_data(weather_path, cab_path):
    df_weather = pd.read_csv(weather_path, on_bad_lines="skip", low_memory=False)
    df_cab = pd.read_csv(cab_path, on_bad_lines="skip", low_memory=False)
    return df_weather, df_cab


# =========================
# Fill Missing Rain (KNN)
# =========================
def impute_rain_knn(df_weather):
    df = df_weather.copy()
    target = "rain"

    # One-hot encode location
    df = pd.get_dummies(df, columns=["location"], drop_first=False)

    feature_cols = [col for col in df.columns if col not in [target, "time_stamp"]]

    train_df = df[df[target].notna()]
    missing_df = df[df[target].isna()]

    if missing_df.empty:
        return df

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_missing = missing_df[feature_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_missing_scaled = scaler.transform(X_missing)

    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
    knn.fit(X_train_scaled, y_train)

    df.loc[df[target].isna(), target] = knn.predict(X_missing_scaled)

    return df


# =========================
# Preprocess Timestamps
# =========================
def preprocess_timestamps(df_cab, df_weather):
    # Convert timestamps (assuming milliseconds)
    df_cab['time_stamp'] = pd.to_datetime(df_cab['time_stamp'], unit='ms').astype('datetime64[ns]')
    df_weather['time_stamp'] = pd.to_datetime(df_weather['time_stamp'], unit='s').astype('datetime64[ns]')

    return (
        df_cab.sort_values("time_stamp"),
        df_weather.sort_values("time_stamp"),
    )


# =========================
# Merge Data
# =========================
def merge_data(df_cab, df_weather):
    merged = pd.merge_asof(
        df_cab,
        df_weather,
        on="time_stamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=15),
    )

    return merged.dropna()


# =========================
# Feature Engineering
# =========================
def engineer_features(df):
    df = df.copy()

    # Time features
    df["hour"] = df["time_stamp"].dt.hour
    df["day"] = df["time_stamp"].dt.day
    df["weekday"] = df["time_stamp"].dt.weekday

    # One-hot encode categorical variables
    df = pd.get_dummies(
        df,
        columns=["cab_type", "source", "destination", "name"],
        drop_first=False,
    )

    # Drop unused columns
    df = df.drop(columns=["time_stamp", "id", "product_id"])

    return df


# =========================
# Prepare Train/Test
# =========================
def prepare_train_test(df):
    y = df["price"]
    X = df.drop(columns=["price"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )


# =========================
# Main Pipeline
# =========================
def main(weather_path, cab_path):
    df_weather, df_cab = load_data(weather_path, cab_path)

    df_weather = impute_rain_knn(df_weather)
    df_cab, df_weather = preprocess_timestamps(df_cab, df_weather)

    merged = merge_data(df_cab, df_weather)
    merged = engineer_features(merged)

    X_train, X_test, y_train, y_test = prepare_train_test(merged)

    np.savez(
        "train_test_split.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )



# =========================
# Run
# =========================
if __name__ == "__main__":
    weather_path = "data/weather.csv"
    cab_path = "data/cab_rides.csv"

    main(weather_path, cab_path)