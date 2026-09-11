import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

TICKERS = ["AAPL"]

FEATURE_PATH = "./save/features/{}_features.csv"

# IMPORTANT:
# Use dates instead of [:1000].
#
# Train:      2017 - 2021
# Validation: 2022
# Test:       2023 - 2026
#
# We do NOT use the test set during training / early stopping.

TRAIN_START = "2017-04-01"
TRAIN_END   = "2021-12-31"

VALID_START = "2022-01-01"
VALID_END   = "2022-12-31"

TEST_START  = "2023-01-01"
TEST_END    = "2026-09-01"


# ============================================================
# 2. LOAD DATA
# ============================================================

total_x_train = []
total_y_train = []

total_x_valid = []
total_y_valid = []

total_x_test = []
total_y_test = []

for ticker in TICKERS:

    path = FEATURE_PATH.format(ticker)

    df = pd.read_csv(path, index_col=0)

    print(df.index)
    # Make sure index is datetime
    df.index = pd.to_datetime(df.index)
    # Sort chronologically
    df = df.sort_index()

    # --------------------------------------------------------
    # Select by DATE, NOT by number of rows
    # --------------------------------------------------------

    train_df = df.loc[TRAIN_START:TRAIN_END]
    valid_df = df.loc[VALID_START:VALID_END]
    test_df  = df.loc[TEST_START:TEST_END]

    print("\n" + "=" * 60)
    print(f"TICKER: {ticker}")
    print("=" * 60)

    print(
        f"Train: {train_df.index.min().date()} -> "
        f"{train_df.index.max().date()} "
        f"({len(train_df)} rows)"
    )

    print(
        f"Valid: {valid_df.index.min().date()} -> "
        f"{valid_df.index.max().date()} "
        f"({len(valid_df)} rows)"
    )

    print(
        f"Test : {test_df.index.min().date()} -> "
        f"{test_df.index.max().date()} "
        f"({len(test_df)} rows)"
    )

    # --------------------------------------------------------
    # Convert to numpy
    # Last column = Return
    # Everything before last column = features
    # --------------------------------------------------------

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    X_valid = valid_df.iloc[:, :-1].values
    y_valid = valid_df.iloc[:, -1].values

    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    total_x_train.append(X_train)
    total_y_train.append(y_train)

    total_x_valid.append(X_valid)
    total_y_valid.append(y_valid)

    total_x_test.append(X_test)
    total_y_test.append(y_test)


# Combine tickers
X_train = np.concatenate(total_x_train, axis=0)
y_train = np.concatenate(total_y_train, axis=0)

X_valid = np.concatenate(total_x_valid, axis=0)
y_valid = np.concatenate(total_y_valid, axis=0)

X_test = np.concatenate(total_x_test, axis=0)
y_test = np.concatenate(total_y_test, axis=0)


print("\n" + "=" * 60)
print("DATA SHAPES")
print("=" * 60)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print(f"X_valid: {X_valid.shape}")
print(f"y_valid: {y_valid.shape}")

print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")


# ============================================================
# 3. BASIC DATA SANITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("TARGET SANITY CHECK")
print("=" * 60)

print(f"Train mean: {np.mean(y_train):.8f}")
print(f"Train std : {np.std(y_train):.8f}")

print(f"Valid mean: {np.mean(y_valid):.8f}")
print(f"Valid std : {np.std(y_valid):.8f}")

print(f"Test mean : {np.mean(y_test):.8f}")
print(f"Test std  : {np.std(y_test):.8f}")

print(f"NaN in X_train: {np.isnan(X_train).sum()}")
print(f"NaN in y_train: {np.isnan(y_train).sum()}")

print(f"Inf in X_train: {np.isinf(X_train).sum()}")
print(f"Inf in y_train: {np.isinf(y_train).sum()}")


# ============================================================
# 4. LIGHTGBM DATASETS
# ============================================================

train_data = lgb.Dataset(
    X_train,
    label=y_train
)

valid_data = lgb.Dataset(
    X_valid,
    label=y_valid,
    reference=train_data
)


# ============================================================
# 5. NORMAL LIGHTGBM MODEL
# ============================================================

# This is intentionally NOT extremely powerful.
# The goal here is to test actual out-of-sample performance.

params = {
    "objective": "regression",
    "metric": "mse",

    "learning_rate": 0.03,

    "num_leaves": 31,
    "max_depth": -1,

    "min_data_in_leaf": 30,

    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,

    "lambda_l1": 0.0,
    "lambda_l2": 0.0,

    "verbosity": -1,
    "seed": 42,
}


print("\n" + "=" * 60)
print("TRAINING NORMAL LIGHTGBM")
print("=" * 60)


model = lgb.train(
    params,
    train_data,
    num_boost_round=2000,

    # IMPORTANT:
    # Only validation set is used for early stopping.
    # Test set is completely untouched.
    valid_sets=[valid_data],

    valid_names=["validation"],

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=100,
            verbose=True
        )
    ]
)


# ============================================================
# 6. PREDICTIONS
# ============================================================

y_pred_train = model.predict(X_train)
y_pred_valid = model.predict(X_valid)
y_pred_test = model.predict(X_test)


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate(name, y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    correlation = np.corrcoef(
        y_true,
        y_pred
    )[0, 1]

    prediction_std = np.std(y_pred)
    actual_std = np.std(y_true)

    # IC = Pearson correlation between
    # predicted and realized returns.
    ic = correlation

    print(f"\n{name}")
    print("-" * 40)

    print(f"RMSE          : {rmse:.8f}")
    print(f"R²            : {r2:.8f}")
    print(f"Correlation   : {correlation:.8f}")
    print(f"Actual std    : {actual_std:.8f}")
    print(f"Prediction std: {prediction_std:.8f}")
    print(f"IC            : {ic:.8f}")

    return {
        "RMSE": rmse,
        "R2": r2,
        "Correlation": correlation,
        "ActualStd": actual_std,
        "PredictionStd": prediction_std,
        "IC": ic
    }


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

train_result = evaluate(
    "TRAIN",
    y_train,
    y_pred_train
)

valid_result = evaluate(
    "VALIDATION",
    y_valid,
    y_pred_valid
)

test_result = evaluate(
    "TEST",
    y_test,
    y_pred_test
)


# ============================================================
# 8. MEAN BASELINE
# ============================================================

# A very important baseline:
# Predict the mean training return for every observation.

baseline_value = np.mean(y_train)

baseline_test_pred = np.full_like(
    y_test,
    baseline_value
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_test_pred
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_test_pred
)


print("\n" + "=" * 60)
print("BASELINE")
print("=" * 60)

print(f"Baseline prediction: {baseline_value:.8f}")
print(f"Baseline RMSE      : {baseline_rmse:.8f}")
print(f"Baseline R²        : {baseline_r2:.8f}")


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 FEATURE IMPORTANCES")
print("=" * 60)

feature_names = pd.read_csv(
    FEATURE_PATH.format(TICKERS[0]),
    index_col=0,
    nrows=1
).columns[:-1]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importance(
        importance_type="gain"
    )
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance.head(20).to_string(index=False))


# ============================================================
# 10. HIGH-CAPACITY OVERFITTING SANITY TEST
# ============================================================
#
# THIS IS VERY IMPORTANT.
#
# We want to answer:
#
# "Can LightGBM actually memorize my training data?"
#
# We deliberately make the model extremely powerful.
#
# If this model still cannot get a very high TRAIN R²,
# something is wrong with:
#
#   - data
#   - target
#   - feature/target alignment
#   - model configuration
#   - preprocessing
#
# It does NOT mean the real financial signal is absent.
#

print("\n" + "=" * 60)
print("HIGH-CAPACITY OVERFITTING SANITY TEST")
print("=" * 60)

overfit_params = {
    "objective": "regression",

    "learning_rate": 0.05,

    "num_leaves": 128,
    "max_depth": -1,

    "min_data_in_leaf": 1,

    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,

    "lambda_l1": 0.0,
    "lambda_l2": 0.0,

    "verbosity": -1,
    "seed": 42,
}


overfit_model = lgb.train(
    overfit_params,
    train_data,

    # No early stopping.
    num_boost_round=3000
)

overfit_pred = overfit_model.predict(
    X_train
)

overfit_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        overfit_pred
    )
)

overfit_r2 = r2_score(
    y_train,
    overfit_pred
)

overfit_corr = np.corrcoef(
    y_train,
    overfit_pred
)[0, 1]


print("\nOVERFIT TEST RESULTS")
print("-" * 40)

print(f"Train RMSE      : {overfit_rmse:.10f}")
print(f"Train R²        : {overfit_r2:.10f}")
print(f"Train correlation: {overfit_corr:.10f}")
print(f"Prediction std  : {np.std(overfit_pred):.10f}")


# ============================================================
# 11. SHUFFLED-TARGET SANITY TEST
# ============================================================
#
# Even stronger diagnostic.
#
# We destroy the relationship between X and y.
#
# A sufficiently powerful model should STILL be able to
# memorize the shuffled training labels.
#
# If it cannot, the model/pipeline has a problem.
#

print("\n" + "=" * 60)
print("SHUFFLED-TARGET SANITY TEST")
print("=" * 60)

rng = np.random.default_rng(42)

y_train_shuffled = rng.permutation(y_train)

shuffled_train_data = lgb.Dataset(
    X_train,
    label=y_train_shuffled
)

shuffled_model = lgb.train(
    overfit_params,
    shuffled_train_data,
    num_boost_round=3000
)

shuffled_pred = shuffled_model.predict(
    X_train
)

shuffled_r2 = r2_score(
    y_train_shuffled,
    shuffled_pred
)

shuffled_rmse = np.sqrt(
    mean_squared_error(
        y_train_shuffled,
        shuffled_pred
    )
)

print(f"Shuffled Train RMSE: {shuffled_rmse:.10f}")
print(f"Shuffled Train R²  : {shuffled_r2:.10f}")


# ============================================================
# 12. PLOT TEST PREDICTION
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    y_test,
    label="Actual Return",
    alpha=0.6
)

plt.plot(
    y_pred_test,
    label="LightGBM Prediction",
    linestyle="--"
)

plt.title(
    "AAPL Next-Day Return Prediction"
)

plt.xlabel("Test Days")
plt.ylabel("Return")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 13. PREDICTION VS ACTUAL
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_pred_test,
    alpha=0.3,
    s=10
)

plt.xlabel("Actual Return")
plt.ylabel("Predicted Return")

plt.title(
    "Actual vs Predicted Return"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 14. FIRST 20 PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 20 TEST PREDICTIONS")
print("=" * 60)

check = pd.DataFrame({
    "Actual": y_test[:20],
    "Predicted": y_pred_test[:20]
})

print(check.to_string(index=False))


# ============================================================
# 15. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)

print("""
1. Check OVERFIT TEST first.

   If Train R² is very high (e.g. > 0.95):
       LightGBM has enough capacity to memorize the data.
       Your pipeline is probably functioning.

   If Train R² is still very low:
       DO NOT continue feature engineering yet.
       Investigate the data/target/model pipeline.

2. Check SHUFFLED-TARGET TEST.

   It should also achieve a high Train R² with this
   deliberately overpowered model.

   If it cannot:
       something is wrong with the pipeline or
       model configuration.

3. Then look at the NORMAL MODEL.

   Train R² high + Test R² near 0/negative:
       Features may not generalize out-of-sample.

   Train R² low + Test R² near 0:
       Model may be underfitting or features contain
       very weak information.

   Test correlation / IC slightly positive:
       There may still be weak predictive information
       even if R² is very small.

4. Compare against the mean baseline.

   If LightGBM cannot beat the baseline out-of-sample,
   that is an important result.

5. DO NOT judge the project from the test result alone.

   First establish that the pipeline can overfit.
   Then investigate whether the signal survives
   out-of-sample.
""")