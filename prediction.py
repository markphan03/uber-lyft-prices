import numpy as np
import torch

from sklearn.linear_model import LinearRegression
from xgboost import XGBRFRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

from nn_model import CabPriceModel


# =========================================================
# LOAD TRAIN/TEST DATA
# =========================================================
def load_data(path="train_test_split.npz"):
    data = np.load(path, allow_pickle=True)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()


# =========================================================
# LINEAR REGRESSION
# =========================================================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


# =========================================================
# XGBOOST RANDOM FOREST
# =========================================================
rf = XGBRFRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    device="cuda",
    tree_method="hist"
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# =========================================================
# NEURAL NETWORK (FROM CHECKPOINT)
# =========================================================
def load_nn_predictions(
    checkpoint_path="nn_checkpoint.pth",
    model=None,
    X_test_t=None,
    y_test_t=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(X_test_t).detach().cpu().numpy()
        true = y_test_t.detach().cpu().numpy()

    return true, preds




# =========================================================
# TENSOR PREPARATION (TRAIN / TEST SPLIT)
# =========================================================
def to_tensor(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    return X_t, y_t

X_test_t, y_test_t = to_tensor(X_test, y_test)

# Define NN model config:
# model, X_test_t, y_test_t
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CabPriceModel(X_train.shape[1]).to(device)
y_true_nn, y_preds_nn = load_nn_predictions(
    model=model,
    X_test_t=X_test_t,
    y_test_t=y_test_t
)


# =========================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =========================================================
def bootstrap_metric_ci_pm(y_true, y_pred, metric_func, n_bootstrap=500, random_state=42):
    rng = np.random.default_rng(random_state)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        score = metric_func(y_true[idx], y_pred[idx])
        scores.append(score)

    scores = np.array(scores)

    mean_score = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    pm = (upper - lower) / 2

    return mean_score, pm


# =========================================================
# MODEL + METRIC SETUP
# =========================================================
models = {
    "Linear Regression": (y_test, y_pred_lr),
    "Random Forest (XGBRF)": (y_test, y_pred_rf),
    "Neural Network": (y_true_nn, y_preds_nn),
}

metrics = {
    "RMSE": root_mean_squared_error,
    "R2": r2_score,
    "MAE": mean_absolute_error,
}


# =========================================================
# EVALUATION LOOP
# =========================================================
def evaluate_models(models, metrics):
    results = {}

    for model_name, (y_true, y_pred) in models.items():
        results[model_name] = {}

        for metric_name, metric_func in metrics.items():
            mean_val, pm_val = bootstrap_metric_ci_pm(
                y_true,
                y_pred,
                metric_func,
                n_bootstrap=10_000,
                random_state=42
            )

            results[model_name][metric_name] = (mean_val, pm_val)

    return results


results = evaluate_models(models, metrics)


# =========================================================
# PRINT RESULTS
# =========================================================
print("\nBootstrap results (95% CI as ± half-width)\n")

for model_name, metric_results in results.items():
    print(f"{model_name}")

    for metric_name, (mean_val, pm_val) in metric_results.items():
        print(f"  {metric_name}: {mean_val:.4f} ± {pm_val:.4f}")

    print()