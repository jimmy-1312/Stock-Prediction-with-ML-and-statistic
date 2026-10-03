import lightgbm as lgb

def train_lightgbm_model(X_train,Y_train):
    lightgbm_data = lgb.Dataset(X_train, label=Y_train)

    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.01, #0.05
        "num_leaves": 20, #20
        # "num_threads":20,
        "max_depth": 3,
        "min_data_in_leaf": 200, #200
        # "lambda_l1": 0.5,
        # "lambda_l2": 1,
        "seed": 43,
        'verbosity': -1,
    }

    # callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)]

    model = lgb.train(
        params,
        lightgbm_data,
        num_boost_round=100,           # 最大迭代次數
        # valid_sets=[train_data, valid_data], # 同時監控訓練集與測試集的 Loss
        # callbacks=callbacks
    )

    return model